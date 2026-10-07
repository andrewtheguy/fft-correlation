//! FFT-based correlation for 1D real-valued signals
//!
//! Provides efficient cross-correlation using FFT with configurable output modes (Full, Same, Valid)
//! matching scipy/numpy conventions. Uses a bounded thread-local FFT plan cache for optimal performance.
//!
//! [`fft_correlate_1d`] is the one-shot entry point. To correlate one signal against several
//! templates, or the same templates against many signals, use [`CorrelationWorkspace`] with
//! [`CorrelationTemplate`]: the signal is transformed once per FFT size, each template's spectrum
//! is cached, and all FFT buffers are reused.
//!
//! # FFT Sizing
//!
//! The FFT size is the smallest `m * 2^k` (`m` one of 1, 3, 5, 9, 15) that covers the full
//! correlation length `N + M - 1`, so it is at most 25% above that length instead of up to twice
//! it with a plain next power of two. It depends only on the signal and template lengths.
//!
//! # Mode Semantics and Indexing
//!
//! The three correlation modes follow the scipy.signal.correlate conventions:
//!
//! - **Full**: Returns complete correlation result with length `N + M - 1` where N is signal length
//!   and M is template length. Output index k corresponds to the lag where template[M-1] aligns
//!   with signal[k].
//!
//! - **Same**: Returns centered output with length equal to the signal. The center of the Full
//!   result is extracted to produce output of the same size as the input signal.
//!
//! - **Valid**: Returns only indices where the template fully overlaps the signal, with length
//!   `N - M + 1` (or empty if M > N). These represent fully-overlapping windows.
//!
//! # References
//!
//! - scipy.signal.correlate: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
//! - numpy.correlate: https://numpy.org/doc/stable/reference/generated/numpy.correlate.html

use realfft::{num_complex::Complex, ComplexToReal, RealFftPlanner, RealToComplex};
use std::{cell::RefCell, collections::VecDeque, sync::Arc};

pub mod error;
pub use error::{FftCorrelationError, Result};

#[cfg(feature = "python")]
mod python;

const FFT_PLAN_CACHE_CAPACITY: usize = 8;

/// Forward and inverse plans for one FFT size.
type FftPlans = (Arc<dyn RealToComplex<f32>>, Arc<dyn ComplexToReal<f32>>);

struct CachedFftPlans {
    fft_size: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
}

struct BoundedFftPlanCache {
    entries: VecDeque<CachedFftPlans>,
}

impl BoundedFftPlanCache {
    fn new() -> Self {
        Self {
            entries: VecDeque::with_capacity(FFT_PLAN_CACHE_CAPACITY),
        }
    }

    fn get_or_insert(&mut self, fft_size: usize) -> FftPlans {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.fft_size == fft_size)
        {
            let entry = self
                .entries
                .remove(index)
                .expect("cache entry should exist at the located index");
            let r2c = Arc::clone(&entry.r2c);
            let c2r = Arc::clone(&entry.c2r);
            self.entries.push_front(entry);
            return (r2c, c2r);
        }

        // Use a fresh planner per cache miss so planner-internal maps do not grow unbounded.
        let mut planner = RealFftPlanner::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);

        if self.entries.len() == FFT_PLAN_CACHE_CAPACITY {
            self.entries.pop_back();
        }

        self.entries.push_front(CachedFftPlans {
            fft_size,
            r2c: Arc::clone(&r2c),
            c2r: Arc::clone(&c2r),
        });

        (r2c, c2r)
    }
}

// Thread-local bounded FFT plan cache for optimal performance
thread_local! {
    static FFT_PLAN_CACHE: RefCell<BoundedFftPlanCache> = RefCell::new(BoundedFftPlanCache::new());
}

fn get_fft_plans(fft_size: usize) -> FftPlans {
    FFT_PLAN_CACHE.with(|cache_cell| cache_cell.borrow_mut().get_or_insert(fft_size))
}

/// Output mode for correlation, matching scipy/numpy conventions
///
/// Determines the size of the correlation output. The indexing convention follows
/// scipy.signal.correlate: in Full mode, index k represents the lag where the
/// template's last sample aligns with signal[k].
///
/// - `Full`: Complete correlation (length = signal.len() + template.len() - 1)
/// - `Same`: Centered output matching signal size (length = signal.len())
/// - `Valid`: Only fully-overlapping region (length = signal.len() - template.len() + 1)
///
/// # Indexing Details
///
/// In Full mode, output[i + template.len() - 1] contains the correlation value
/// for a window starting at signal[i]. For Same mode, the center index is
/// (output_len - signal.len()) / 2, providing a symmetric view. For Valid mode,
/// only indices where the template fully overlaps the signal are returned.
///
/// # References
///
/// scipy.signal.correlate documentation:
/// https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Full correlation output (signal.len() + template.len() - 1 samples)
    Full,
    /// Centered output matching first input size (signal.len() samples)
    Same,
    /// Only fully-overlapping region (signal.len() - template.len() + 1 samples)
    Valid,
}

/// Odd factors of the FFT sizes [`fast_fft_size`] picks from.
const FAST_FFT_SIZE_FACTORS: [usize; 5] = [1, 3, 5, 9, 15];

/// Smallest FFT size of the form `m * 2^k` (`m` one of [`FAST_FFT_SIZE_FACTORS`]) that is at
/// least `min_size`.
///
/// A plain next power of two can nearly double the transform; these sizes are about as fast per
/// sample and at most 25% above `min_size`. The steps are kept coarse on purpose, so templates of
/// similar length still land on the same size and share the signal's forward FFT.
fn fast_fft_size(min_size: usize) -> usize {
    FAST_FFT_SIZE_FACTORS
        .iter()
        .map(|&factor| factor * min_size.div_ceil(factor).next_power_of_two())
        .min()
        .expect("there is at least one factor")
}

/// Number of FFT sizes a [`CorrelationTemplate`] keeps spectra for.
const TEMPLATE_SPECTRUM_CACHE_CAPACITY: usize = 4;

/// A correlation template with its time-reversed spectrum cached per FFT size, so repeated
/// correlations against new signals only transform the signal.
///
/// Spectra for the most recent [`TEMPLATE_SPECTRUM_CACHE_CAPACITY`] FFT sizes are kept.
pub struct CorrelationTemplate {
    reversed: Vec<f32>,
    spectra: Vec<(usize, Vec<Complex<f32>>)>,
}

impl CorrelationTemplate {
    pub fn new(template: &[f32]) -> Self {
        Self {
            // Reverse template for correlation via the Correlation Theorem:
            // For real-valued signals, correlation(x, y) = IFFT(FFT(x) * conj(FFT(y)))
            // Time-reversing y achieves the same effect as frequency-domain conjugation,
            // which is equivalent to: signal * reverse(template) in time domain.
            // Reference: Oppenheim & Schafer, "Discrete-Time Signal Processing"
            reversed: template.iter().rev().copied().collect(),
            spectra: Vec::new(),
        }
    }

    /// Number of samples in the template.
    pub fn len(&self) -> usize {
        self.reversed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reversed.is_empty()
    }

    /// Spectrum of the zero-padded, reversed template at `fft_size`.
    fn spectrum(&mut self, fft_size: usize) -> Result<&[Complex<f32>]> {
        if let Some(index) = self.spectra.iter().position(|(size, _)| *size == fft_size) {
            return Ok(&self.spectra[index].1);
        }
        debug_assert!(
            fft_size >= self.reversed.len(),
            "FFT size must cover the template"
        );

        let (r2c, _) = get_fft_plans(fft_size);
        let mut padded = vec![0.0_f32; fft_size];
        padded[..self.reversed.len()].copy_from_slice(&self.reversed);
        let mut spectrum = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();
        r2c.process_with_scratch(&mut padded, &mut spectrum, &mut scratch)
            .map_err(|e| {
                FftCorrelationError::FftProcessing(format!(
                    "FFT forward process failed for template: {:?}",
                    e
                ))
            })?;

        if self.spectra.len() == TEMPLATE_SPECTRUM_CACHE_CAPACITY {
            self.spectra.remove(0);
        }
        self.spectra.push((fft_size, spectrum));
        Ok(&self.spectra.last().expect("spectrum was just pushed").1)
    }
}

/// Reusable buffers for FFT cross-correlation of one signal against any number of templates.
///
/// The signal is set by [`load_signal`] and transformed once per FFT size, so each [`correlate`]
/// call with a template needing the same size costs a spectrum product and one inverse FFT.
/// Buffers are kept between signals of the same FFT size.
///
/// The FFT size of a correlation depends only on the signal and that template, never on the
/// other templates, so the result is bit-identical to [`fft_correlate_1d`] on the pair.
///
/// ```
/// use fft_correlation::{CorrelationTemplate, CorrelationWorkspace, Mode};
///
/// let mut workspace = CorrelationWorkspace::new();
/// let mut template = CorrelationTemplate::new(&[1.0, 0.5]);
/// let mut output = Vec::new();
///
/// workspace.load_signal(&[1.0, 2.0, 3.0, 4.0]);
/// workspace.correlate(&mut template, Mode::Full, &mut output).unwrap();
/// assert_eq!(output.len(), 5);
/// ```
///
/// [`load_signal`]: CorrelationWorkspace::load_signal
/// [`correlate`]: CorrelationWorkspace::correlate
#[derive(Default)]
pub struct CorrelationWorkspace {
    signal: Vec<f32>,
    /// FFT size the buffers are allocated for (0 before the first transform).
    fft_size: usize,
    /// Whether `signal_spectrum` is the transform of `signal` at `fft_size`.
    spectrum_loaded: bool,
    plans: Option<FftPlans>,
    padded_signal: Vec<f32>,
    signal_spectrum: Vec<Complex<f32>>,
    product: Vec<Complex<f32>>,
    forward_scratch: Vec<Complex<f32>>,
    inverse_scratch: Vec<Complex<f32>>,
    time_domain: Vec<f32>,
}

impl CorrelationWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the signal for the following [`correlate`](Self::correlate) calls.
    pub fn load_signal(&mut self, signal: &[f32]) {
        self.signal.clear();
        self.signal.extend_from_slice(signal);
        self.spectrum_loaded = false;
    }

    /// Transform the loaded signal at `fft_size` unless that is already done.
    fn transform_signal(&mut self, fft_size: usize) -> Result<()> {
        if self.spectrum_loaded && self.fft_size == fft_size {
            return Ok(());
        }
        self.spectrum_loaded = false;
        if self.fft_size != fft_size || self.plans.is_none() {
            let plans = get_fft_plans(fft_size);
            let (r2c, c2r) = &plans;
            self.signal_spectrum = r2c.make_output_vec();
            self.product = r2c.make_output_vec();
            self.forward_scratch = r2c.make_scratch_vec();
            self.inverse_scratch = c2r.make_scratch_vec();
            self.time_domain = c2r.make_output_vec();
            self.plans = Some(plans);
            self.fft_size = fft_size;
        }

        self.padded_signal.clear();
        self.padded_signal.resize(fft_size, 0.0);
        self.padded_signal[..self.signal.len()].copy_from_slice(&self.signal);

        let (r2c, _) = self.plans.as_ref().expect("plans were just set");
        r2c.process_with_scratch(
            &mut self.padded_signal,
            &mut self.signal_spectrum,
            &mut self.forward_scratch,
        )
        .map_err(|e| {
            FftCorrelationError::FftProcessing(format!(
                "FFT forward process failed for signal: {:?}",
                e
            ))
        })?;
        self.spectrum_loaded = true;
        Ok(())
    }

    /// Cross-correlation of the loaded signal with `template`, written to `output` (replacing
    /// its contents).
    ///
    /// Modes and indexing are those of [`fft_correlate_1d`]. `output` is left empty if the
    /// signal or the template is empty, or if Valid mode is used with a signal shorter than the
    /// template.
    ///
    /// # Errors
    ///
    /// Returns `FftCorrelationError::FftProcessing` if FFT processing fails.
    pub fn correlate(
        &mut self,
        template: &mut CorrelationTemplate,
        mode: Mode,
        output: &mut Vec<f32>,
    ) -> Result<()> {
        output.clear();
        if self.signal.is_empty() || template.is_empty() {
            return Ok(());
        }

        let signal_len = self.signal.len();
        // Covers the full correlation without wrap-around.
        let output_len = signal_len + template.len() - 1;
        let (trim_start, trim_len) = match mode {
            Mode::Full => (0, output_len),
            Mode::Same => ((output_len - signal_len) / 2, signal_len),
            Mode::Valid => {
                if signal_len < template.len() {
                    return Ok(());
                }
                (template.len() - 1, signal_len - template.len() + 1)
            }
        };
        self.transform_signal(fast_fft_size(output_len))?;

        // Frequency domain multiplication (element-wise). The template is already reversed.
        let template_spectrum = template.spectrum(self.fft_size)?;
        for ((p, s), t) in self
            .product
            .iter_mut()
            .zip(&self.signal_spectrum)
            .zip(template_spectrum)
        {
            *p = *s * *t;
        }

        let (_, c2r) = self
            .plans
            .as_ref()
            .expect("the signal was just transformed");
        c2r.process_with_scratch(
            &mut self.product,
            &mut self.time_domain,
            &mut self.inverse_scratch,
        )
        .map_err(|e| {
            FftCorrelationError::FftProcessing(format!("FFT inverse process failed: {:?}", e))
        })?;

        // Normalize only the output window we keep.
        let normalization = 1.0 / self.fft_size as f32;
        output.extend(
            self.time_domain[trim_start..trim_start + trim_len]
                .iter()
                .map(|x| x * normalization),
        );
        Ok(())
    }
}

/// Correlate two 1D signals using FFT
///
/// Computes cross-correlation efficiently using FFT with O(N log N) complexity.
/// The `mode` parameter controls output size:
/// - `Mode::Full`: Returns complete correlation (signal.len() + template.len() - 1)
/// - `Mode::Same`: Returns centered output matching signal.len()
/// - `Mode::Valid`: Returns only fully-overlapping region (signal.len() - template.len() + 1)
///
/// This is a one-shot convenience over [`CorrelationWorkspace`]; use the workspace directly to
/// reuse the signal's transform across templates.
///
/// # Indexing Convention
///
/// In `Mode::Full`, output index `k` corresponds to the lag where `template[template.len()-1]`
/// aligns with `signal[k]`. Equivalently, a window starting at position `i` in the signal
/// maps to output index `i + template.len() - 1`. This matches the convention used in
/// scipy.signal.correlate and numpy.correlate.
///
/// Returns an empty vector if either input is empty or if Valid mode is used
/// with signal shorter than template.
///
/// # Errors
///
/// Returns `FftCorrelationError::FftProcessing` if FFT processing fails.
///
/// # References
///
/// - scipy.signal.correlate: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
/// - numpy.correlate: https://numpy.org/doc/stable/reference/generated/numpy.correlate.html
pub fn fft_correlate_1d(signal: &[f32], template: &[f32], mode: Mode) -> Result<Vec<f32>> {
    let mut workspace = CorrelationWorkspace::new();
    let mut template = CorrelationTemplate::new(template);
    let mut output = Vec::new();
    workspace.load_signal(signal);
    workspace.correlate(&mut template, mode, &mut output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Naive correlation used for correctness checks in tests.
    fn naive_full_correlation(signal: &[f32], template: &[f32]) -> Vec<f32> {
        let output_len = signal.len() + template.len() - 1;
        let mut result = vec![0.0; output_len];

        for lag in 0..output_len {
            let mut correlation = 0.0;
            for i in 0..template.len() {
                let signal_idx = lag as isize - (template.len() as isize - 1) + i as isize;
                if (0..signal.len() as isize).contains(&signal_idx) {
                    correlation += signal[signal_idx as usize] * template[i];
                }
            }
            result[lag] = correlation;
        }

        result
    }

    #[test]
    fn test_fft_correlate_mode_full_length() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let template = vec![1.0, 0.0, 0.0];
        let result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        assert_eq!(result.len(), signal.len() + template.len() - 1);

        let signal = vec![1.0; 100];
        let template = vec![1.0; 10];
        let result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        assert_eq!(result.len(), 109);
    }

    #[test]
    fn test_fft_plan_cache_is_bounded() {
        FFT_PLAN_CACHE.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();
            cache.entries.clear();

            for power in 3..(3 + FFT_PLAN_CACHE_CAPACITY + 2) {
                let fft_size = 1usize << power;
                let _ = cache.get_or_insert(fft_size);
            }

            let cached_sizes: Vec<_> = cache.entries.iter().map(|entry| entry.fft_size).collect();

            assert_eq!(cached_sizes.len(), FFT_PLAN_CACHE_CAPACITY);
            assert_eq!(
                cached_sizes[0],
                1usize << (3 + FFT_PLAN_CACHE_CAPACITY + 1),
                "most recently inserted size should be kept"
            );
            assert!(
                !cached_sizes.contains(&(1usize << 3)),
                "oldest size should be evicted first"
            );
            assert!(
                !cached_sizes.contains(&(1usize << 4)),
                "second-oldest size should be evicted once capacity is exceeded twice"
            );
        });
    }

    #[test]
    fn test_fft_correlate_mode_same_length() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let template = vec![1.0, 0.0, 0.0];
        let result = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        assert_eq!(result.len(), signal.len());

        let signal = vec![1.0; 100];
        let template = vec![1.0; 10];
        let result = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_fft_correlate_mode_valid_length() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let template = vec![1.0, 0.0, 0.0];
        let result = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();
        assert_eq!(result.len(), signal.len() - template.len() + 1);

        let signal = vec![1.0; 100];
        let template = vec![1.0; 10];
        let result = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();
        assert_eq!(result.len(), 91);

        // Template longer than signal
        let signal = vec![1.0, 2.0];
        let template = vec![1.0; 10];
        let result = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_fft_correlate_mode_full_impulse() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let template = vec![1.0, 0.0, 0.0];
        let result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();

        // Correlation with impulse at position 0 should return signal shifted
        assert_eq!(result.len(), 7);
        assert!((result[2] - 1.0).abs() < 1e-4);
        assert!((result[3] - 2.0).abs() < 1e-4);
        assert!((result[4] - 3.0).abs() < 1e-4);
        assert!((result[5] - 4.0).abs() < 1e-4);
        assert!((result[6] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_fft_correlate_mode_same_centering() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let template = vec![1.0, 0.0, 0.0];

        let full_result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let same_result = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();

        // Same mode should be centered slice of Full mode
        let output_len = full_result.len();
        let start = (output_len - signal.len()) / 2;
        let expected_same = &full_result[start..start + signal.len()];

        assert_eq!(same_result.len(), expected_same.len());
        for (a, b) in same_result.iter().zip(expected_same.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_fft_correlate_mode_valid_no_edges() {
        let signal = vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0];
        let template = vec![1.0, 1.0, 1.0];

        let valid_result = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();

        // Valid mode should have length 5 for this case
        assert_eq!(valid_result.len(), 5);

        // Check that peak is in the center where template fully overlaps with signal
        let max_idx = valid_result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(max_idx >= 1 && max_idx <= 3);
    }

    #[test]
    fn test_fft_correlate_modes_consistency() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let template = vec![0.5, 1.0, 0.5];

        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();

        // Check Same is centered slice of Full
        let output_len = full.len();
        let start = (output_len - signal.len()) / 2;
        for (i, &val) in same.iter().enumerate() {
            assert!((val - full[start + i]).abs() < 1e-4);
        }

        // Check Valid is appropriate slice of Full
        let valid_start = template.len() - 1;
        for (i, &val) in valid.iter().enumerate() {
            assert!((val - full[valid_start + i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_fft_correlate_vs_sliding_window() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let template = vec![0.5, 1.0, 0.5];

        let fft_result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let sliding_result = naive_full_correlation(&signal, &template);

        assert_eq!(fft_result.len(), sliding_result.len());
        for (i, (&fft_val, &sliding_val)) in
            fft_result.iter().zip(sliding_result.iter()).enumerate()
        {
            assert!(
                (fft_val - sliding_val).abs() < 1e-4,
                "Sample {} mismatch: FFT={}, sliding={}",
                i,
                fft_val,
                sliding_val
            );
        }
    }

    #[test]
    fn test_fft_correlate_chirp_signals() {
        use std::f32::consts::PI;

        // Generate chirp from 200-4000 Hz
        fn generate_chirp(samples: usize, f_start: f32, f_end: f32) -> Vec<f32> {
            let sample_rate = 16000.0;
            let duration = samples as f32 / sample_rate;
            let mut signal = vec![0.0; samples];
            for n in 0..samples {
                let t = n as f32 / sample_rate;
                let k = (f_end - f_start) / duration;
                let phase = 2.0 * PI * (f_start * t + k * t * t / 2.0);
                signal[n] = phase.sin();
            }
            signal
        }

        let template = generate_chirp(1600, 200.0, 4000.0);
        let mut signal = vec![0.0; 500];
        signal.extend_from_slice(&template);
        signal.extend_from_slice(&vec![0.0; 500]);

        // Test all three modes
        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();

        // All should find peak
        let full_peak = full
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let same_peak = same
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let valid_peak = valid
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert!(full_peak > 100.0);
        assert!(same_peak > 100.0);
        assert!(valid_peak > 100.0);
    }

    #[test]
    fn test_fft_correlate_empty_inputs() {
        let result1 = fft_correlate_1d(&[], &[1.0, 2.0, 3.0], Mode::Full).unwrap();
        assert_eq!(result1.len(), 0);

        let result2 = fft_correlate_1d(&[1.0, 2.0, 3.0], &[], Mode::Full).unwrap();
        assert_eq!(result2.len(), 0);

        let result3 = fft_correlate_1d(&[], &[], Mode::Full).unwrap();
        assert_eq!(result3.len(), 0);
    }

    #[test]
    fn test_fft_correlate_single_element() {
        let signal = vec![5.0];
        let template = vec![2.0];

        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        assert_eq!(full.len(), 1);
        assert!((full[0] - 10.0).abs() < 1e-4);

        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        assert_eq!(same.len(), 1);
        assert!((same[0] - 10.0).abs() < 1e-4);

        let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();
        assert_eq!(valid.len(), 1);
        assert!((valid[0] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_fft_correlate_time_reversal_equivalence() {
        // Verify that correlation is equivalent to time-reversing template
        // for real-valued signals: correlate(x, y) ≡ convolve(x, reverse(y))
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let template = vec![0.5, 1.0, 1.5];

        // Compute correlation in both directions
        let result_xy = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let result_yx = fft_correlate_1d(&template, &signal, Mode::Full).unwrap();

        // For real signals, correlate(x, y) reversed should equal correlate(y, x) reversed
        // correlate(x,y) has length signal.len() + template.len() - 1
        // correlate(y,x) has length template.len() + signal.len() - 1 (same)
        assert_eq!(result_xy.len(), result_yx.len());

        // Reverse result_yx and compare with result_xy
        let result_yx_rev: Vec<f32> = result_yx.iter().rev().cloned().collect();
        // Due to the way correlation is defined, they should match within tolerance
        for (i, (&val_xy, val_yx_rev)) in result_xy.iter().zip(result_yx_rev.iter()).enumerate() {
            assert!(
                (val_xy - val_yx_rev).abs() < 1e-4,
                "Mismatch at index {}: {} vs {}",
                i,
                val_xy,
                val_yx_rev
            );
        }
    }

    #[test]
    fn test_fft_correlate_equal_length() {
        let signal = vec![1.0, 2.0, 3.0];
        let template = vec![0.5, 1.0, 1.5];

        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        assert_eq!(full.len(), 5);

        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        assert_eq!(same.len(), 3);

        let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();
        assert_eq!(valid.len(), 1);
    }

    #[test]
    fn test_fft_correlate_template_longer() {
        let signal = vec![1.0, 2.0];
        let template = vec![1.0; 10];

        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        assert_eq!(full.len(), 11);

        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        assert_eq!(same.len(), 2);

        let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();
        assert_eq!(valid.len(), 0);
    }

    #[test]
    fn test_fft_correlate_normalization() {
        // Autocorrelation test
        let signal = vec![1.0; 50];
        let template = signal.clone();

        let result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();

        let max_val = result
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // Autocorrelation peak should equal signal length
        assert!((max_val - 50.0).abs() < 0.5);
    }

    #[test]
    fn test_fft_correlate_even_odd_length_combinations() {
        // Comment 3: Test even/odd length combinations for correct Same/Valid centering

        // Case 1: signal 8 (even), template 4 (even)
        let signal = vec![1.0; 8];
        let template = vec![0.5; 4];
        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();

        assert_eq!(full.len(), 11);
        assert_eq!(same.len(), 8);
        assert_eq!(valid.len(), 5);

        // Verify Same is centered slice of Full
        let output_len = full.len();
        let start = (output_len - signal.len()) / 2;
        for (i, &val) in same.iter().enumerate() {
            assert!((val - full[start + i]).abs() < 1e-4);
        }

        // Verify Valid is correct slice of Full
        let valid_start = template.len() - 1;
        for (i, &val) in valid.iter().enumerate() {
            assert!((val - full[valid_start + i]).abs() < 1e-4);
        }

        // Case 2: signal 8 (even), template 3 (odd)
        let signal = vec![1.0; 8];
        let template = vec![0.5; 3];
        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();

        assert_eq!(full.len(), 10);
        assert_eq!(same.len(), 8);
        assert_eq!(valid.len(), 6);

        // Case 3: signal 7 (odd), template 4 (even)
        let signal = vec![1.0; 7];
        let template = vec![0.5; 4];
        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();

        assert_eq!(full.len(), 10);
        assert_eq!(same.len(), 7);
        assert_eq!(valid.len(), 4);
    }

    #[test]
    fn test_fft_correlate_large_randomized_fft_sizing() {
        // Comment 4: Test FFT sizing with lengths straddling powers-of-two
        // 1025 + 64 - 1 = 1088 samples just exceed 1024 and use FFT size 1152 (9 * 128)

        use std::f32::consts::PI;

        let signal_len = 1025;
        let template_len = 64;

        // Generate pseudo-random signal
        let signal: Vec<f32> = (0..signal_len)
            .map(|i| (i as f32 * 0.1).sin() + 0.001 * ((i as f32 * 0.7).cos()))
            .collect();

        let template: Vec<f32> = (0..template_len)
            .map(|i| (i as f32 * 2.0 * PI / template_len as f32).sin())
            .collect();

        // Compute FFT correlation
        let fft_result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();

        // Verify output length
        assert_eq!(fft_result.len(), signal.len() + template.len() - 1);

        // Verify no NaN or Inf values
        for (i, &val) in fft_result.iter().enumerate() {
            assert!(val.is_finite(), "Output should be finite at index {}", i);
        }

        // Verify peak value is reasonable (should be within reasonable bounds for sine wave correlation)
        let max_abs = fft_result
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(
            max_abs < 100.0,
            "Peak value should be reasonable for sine wave, got {}",
            max_abs
        );
    }

    #[test]
    fn test_fft_correlate_all_zero_inputs() {
        // Comment 5: Test behavior with all-zero inputs

        let signal = vec![0.0; 10];
        let template = vec![0.0; 5];

        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
        let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();

        // All should return zeros
        for val in &full {
            assert!(val.abs() < 1e-6, "Full mode should be zero, got {}", val);
        }
        for val in &same {
            assert!(val.abs() < 1e-6, "Same mode should be zero, got {}", val);
        }
        for val in &valid {
            assert!(val.abs() < 1e-6, "Valid mode should be zero, got {}", val);
        }
    }

    #[test]
    fn test_fft_correlate_nan_inf_handling() {
        // Comment 5: Test behavior with NaN and Inf (defines expected behavior)
        // Note: FFT operations may fail with NaN/Inf; we document this behavior

        // Test signal with one very large value (avoiding NaN/Inf which break FFT)
        let mut signal = vec![1.0; 10];
        signal[5] = 1e10; // Use large value instead of NaN (NaN breaks FFT validation)
        let template = vec![0.5; 3];

        let result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();

        // Result should preserve large values
        assert!(
            result.iter().any(|x| *x > 1e8),
            "Large values should be present in correlation"
        );

        // Test with negative values
        let mut signal = vec![1.0; 10];
        signal[3] = -10.0;
        let template = vec![0.5; 3];

        let result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();

        // Result should contain the effect of negative values
        assert!(
            result.iter().any(|x| *x < 0.0),
            "Negative values should propagate through correlation"
        );
    }

    #[test]
    fn test_fft_correlate_same_centering_even_template() {
        // Comment 6: Test Mode::Same centering when template is even (ambiguous center)
        // Verify left-biased centering for consistent alignment

        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // length 6
        let template = vec![0.5, 1.0, 1.5, 2.0]; // length 4 (even)

        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();

        assert_eq!(same.len(), signal.len());

        // Full has length 6 + 4 - 1 = 9
        // Same should extract center: start = (9 - 6) / 2 = 1, so indices 1..7
        let output_len = full.len();
        let start = (output_len - signal.len()) / 2;

        for (i, &val) in same.iter().enumerate() {
            assert!(
                (val - full[start + i]).abs() < 1e-4,
                "Same mode index {} should match Full mode index {}",
                i,
                start + i
            );
        }
    }

    #[test]
    fn test_fft_correlate_autocorrelation_sinusoid() {
        // Comment 8: Test autocorrelation of non-constant signal (sinusoid)

        use std::f32::consts::PI;

        let n = 64;
        // Generate sinusoid
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * (i as f32) / n as f32).sin())
            .collect();

        let autocorr = fft_correlate_1d(&signal, &signal, Mode::Full).unwrap();

        // Autocorrelation peak should be at the center (lag 0) and equal sum of squares
        let sum_sq: f32 = signal.iter().map(|x| x * x).sum();
        let peak = autocorr[n - 1]; // lag 0 is at index n - 1 in Full mode

        assert!(
            (peak - sum_sq).abs() < 0.1,
            "Autocorr peak {} should equal sum_sq {}",
            peak,
            sum_sq
        );

        // Peak should be maximum
        for val in &autocorr {
            assert!(
                *val <= peak + 1e-4,
                "Autocorr value {} exceeds peak {}",
                val,
                peak
            );
        }

        // Test Same mode autocorrelation
        let same_result = fft_correlate_1d(&signal, &signal, Mode::Same).unwrap();
        assert_eq!(same_result.len(), n);

        // Same mode should have peak at center
        let same_peak_idx = same_result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(
            same_peak_idx >= n / 2 - 2 && same_peak_idx <= n / 2 + 2,
            "Same mode peak should be near center, got index {}",
            same_peak_idx
        );
    }

    #[test]
    fn test_fft_correlate_matches_naive_across_modes_small_vectors() {
        for signal_len in 1..=5 {
            for template_len in 1..=6 {
                let signal: Vec<f32> = (0..signal_len)
                    .map(|i| ((i as f32) * 0.7).cos() + (i as f32) * 0.05)
                    .collect();
                let template: Vec<f32> = (0..template_len)
                    .map(|i| ((i as f32) * 0.4).sin() - (i as f32) * 0.03)
                    .collect();

                let naive_full = naive_full_correlation(&signal, &template);
                let fft_full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();

                assert_eq!(
                    fft_full.len(),
                    naive_full.len(),
                    "Full length mismatch for signal {} template {}",
                    signal_len,
                    template_len
                );
                for (idx, (expected, actual)) in naive_full.iter().zip(fft_full.iter()).enumerate()
                {
                    assert!(
                        (expected - actual).abs() < 1e-4,
                        "Full mode mismatch at {} for signal {} template {}: expected {}, got {}",
                        idx,
                        signal_len,
                        template_len,
                        expected,
                        actual
                    );
                }

                let fft_same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
                let same_start = (naive_full.len() - signal.len()) / 2;
                let expected_same = &naive_full[same_start..same_start + signal.len()];
                assert_eq!(
                    fft_same.len(),
                    expected_same.len(),
                    "Same length mismatch for signal {} template {}",
                    signal_len,
                    template_len
                );
                for (idx, (expected, actual)) in
                    expected_same.iter().zip(fft_same.iter()).enumerate()
                {
                    assert!(
                        (expected - actual).abs() < 1e-4,
                        "Same mode mismatch at {} for signal {} template {}: expected {}, got {}",
                        idx,
                        signal_len,
                        template_len,
                        expected,
                        actual
                    );
                }

                let fft_valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();
                if signal.len() < template.len() {
                    assert!(
                        fft_valid.is_empty(),
                        "Valid mode should be empty for signal {} template {}",
                        signal_len,
                        template_len
                    );
                } else {
                    let valid_start = template.len() - 1;
                    let valid_len = signal.len() - template.len() + 1;
                    let expected_valid = &naive_full[valid_start..valid_start + valid_len];
                    assert_eq!(
                        fft_valid.len(),
                        expected_valid.len(),
                        "Valid length mismatch for signal {} template {}",
                        signal_len,
                        template_len
                    );
                    for (idx, (expected, actual)) in
                        expected_valid.iter().zip(fft_valid.iter()).enumerate()
                    {
                        assert!(
                            (expected - actual).abs() < 1e-4,
                            "Valid mode mismatch at {} for signal {} template {}: expected {}, got {}",
                            idx,
                            signal_len,
                            template_len,
                            expected,
                            actual
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_fft_correlate_thread_safety_consistency() {
        use std::thread;

        let signal: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.05).sin()).collect();
        let template: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.2).cos()).collect();

        let expected = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let signal = signal.clone();
                let template = template.clone();
                thread::spawn(move || fft_correlate_1d(&signal, &template, Mode::Same).unwrap())
            })
            .collect();

        for (idx, handle) in handles.into_iter().enumerate() {
            let result = handle.join().expect("thread panicked");
            assert_eq!(
                result.len(),
                expected.len(),
                "Length mismatch in thread {}",
                idx
            );
            for (exp, got) in expected.iter().zip(result.iter()) {
                assert!((exp - got).abs() < 1e-4, "Value mismatch in thread {}", idx);
            }
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    fn test_fast_fft_size() {
        let sizes: Vec<usize> = (0..=20).map(fast_fft_size).collect();
        assert_eq!(
            sizes,
            vec![1, 1, 2, 3, 4, 5, 6, 8, 8, 9, 10, 12, 12, 15, 15, 15, 16, 18, 18, 20, 20]
        );
        // Powers of two are kept, and a size just above one no longer doubles.
        assert_eq!(fast_fft_size(524_288), 524_288);
        assert_eq!(fast_fft_size(524_289), 589_824);
        assert_eq!(fast_fft_size(589_825), 655_360);
        assert_eq!(fast_fft_size(655_361), 786_432);
        assert_eq!(fast_fft_size(786_433), 983_040);
        assert_eq!(fast_fft_size(983_041), 1_048_576);
    }

    #[test]
    fn test_fast_fft_size_is_at_most_a_quarter_above_the_minimum() {
        for min_size in 1..=5000usize {
            let size = fast_fft_size(min_size);
            assert!(size >= min_size, "{size} < {min_size}");
            assert!(
                size * 4 <= min_size * 5 || size - min_size <= 1,
                "{size} for {min_size}"
            );
        }
    }

    #[test]
    fn test_template_len_and_is_empty() {
        let template = CorrelationTemplate::new(&[1.0, 2.0, 3.0]);
        assert_eq!(template.len(), 3);
        assert!(!template.is_empty());
        assert_eq!(template.reversed, vec![3.0, 2.0, 1.0]);

        let empty = CorrelationTemplate::new(&[]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_workspace_reuse_matches_naive() {
        let signal_a: Vec<f32> = (0..50)
            .map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0)
            .collect();
        let signal_b: Vec<f32> = (0..70).map(|i| ((i * 3 % 19) as f32 - 9.0) / 9.0).collect();
        let template_a: Vec<f32> = (0..7).map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0).collect();
        let template_b: Vec<f32> = (0..12)
            .map(|i| ((i * 11 % 13) as f32 - 6.0) / 6.0)
            .collect();

        let mut workspace = CorrelationWorkspace::new();
        let mut cached_a = CorrelationTemplate::new(&template_a);
        let mut cached_b = CorrelationTemplate::new(&template_b);
        let mut output = Vec::new();

        // Two templates against one signal, then a different signal size.
        for signal in [&signal_a, &signal_b, &signal_a] {
            workspace.load_signal(signal);
            for (cached, template) in [(&mut cached_a, &template_a), (&mut cached_b, &template_b)] {
                workspace
                    .correlate(cached, Mode::Full, &mut output)
                    .unwrap();
                assert_close(&output, &naive_full_correlation(signal, template));
            }
        }
        assert_eq!(
            cached_a.spectra.len(),
            2,
            "one cached spectrum per FFT size"
        );
    }

    #[test]
    fn test_workspace_modes_are_windows_of_full() {
        let signal: Vec<f32> = (0..37).map(|i| ((i * 7 % 11) as f32 - 5.0) / 5.0).collect();
        let template: Vec<f32> = (0..8).map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0).collect();

        let mut workspace = CorrelationWorkspace::new();
        let mut cached = CorrelationTemplate::new(&template);
        let (mut full, mut same, mut valid) = (Vec::new(), Vec::new(), Vec::new());
        workspace.load_signal(&signal);
        workspace
            .correlate(&mut cached, Mode::Full, &mut full)
            .unwrap();
        workspace
            .correlate(&mut cached, Mode::Same, &mut same)
            .unwrap();
        workspace
            .correlate(&mut cached, Mode::Valid, &mut valid)
            .unwrap();

        assert_close(&full, &naive_full_correlation(&signal, &template));
        // Same starts (44 - 37) / 2 = 3 samples in, Valid skips the 7 partial overlaps.
        assert_eq!(same, full[3..40]);
        assert_eq!(valid, full[7..37]);
        for mode in [Mode::Full, Mode::Same, Mode::Valid] {
            let mut output = Vec::new();
            workspace.correlate(&mut cached, mode, &mut output).unwrap();
            assert_eq!(output, fft_correlate_1d(&signal, &template, mode).unwrap());
        }
    }

    #[test]
    fn test_workspace_empty_inputs_clear_the_output() {
        let mut workspace = CorrelationWorkspace::new();
        let mut template = CorrelationTemplate::new(&[1.0, 2.0]);
        let mut output = vec![1.0];
        workspace.load_signal(&[]);
        workspace
            .correlate(&mut template, Mode::Full, &mut output)
            .unwrap();
        assert!(output.is_empty());

        let mut empty = CorrelationTemplate::new(&[]);
        output.push(1.0);
        workspace.load_signal(&[1.0, 2.0, 3.0]);
        workspace
            .correlate(&mut empty, Mode::Full, &mut output)
            .unwrap();
        assert!(output.is_empty());

        // Valid mode with a template longer than the signal.
        let mut long = CorrelationTemplate::new(&[1.0, 2.0, 3.0, 4.0]);
        output.push(1.0);
        workspace
            .correlate(&mut long, Mode::Valid, &mut output)
            .unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_workspace_output_is_replaced_not_appended() {
        let mut workspace = CorrelationWorkspace::new();
        let mut template = CorrelationTemplate::new(&[1.0, 0.5]);
        let mut output = vec![9.0; 20];
        workspace.load_signal(&[1.0, 2.0, 3.0, 4.0]);
        workspace
            .correlate(&mut template, Mode::Full, &mut output)
            .unwrap();
        // scipy.signal.correlate([1, 2, 3, 4], [1, 0.5], mode="full")
        assert_close(&output, &[0.5, 2.0, 3.5, 5.0, 4.0]);
    }

    #[test]
    fn test_cached_spectrum_gives_identical_output() {
        let signal: Vec<f32> = (0..90)
            .map(|i| ((i * 29 % 31) as f32 - 15.0) / 15.0)
            .collect();
        let template: Vec<f32> = (0..11).map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0).collect();

        let mut workspace = CorrelationWorkspace::new();
        let mut cached = CorrelationTemplate::new(&template);
        let mut first = Vec::new();
        let mut second = Vec::new();

        workspace.load_signal(&signal);
        workspace
            .correlate(&mut cached, Mode::Full, &mut first)
            .unwrap();
        assert_eq!(cached.spectra.len(), 1);
        // Same signal again: the cached spectrum is used (no new entry) and the result is
        // bit-identical to the first pass and to the one-shot.
        workspace.load_signal(&signal);
        workspace
            .correlate(&mut cached, Mode::Full, &mut second)
            .unwrap();
        assert_eq!(cached.spectra.len(), 1);
        assert_eq!(first, second);
        assert_eq!(
            first,
            fft_correlate_1d(&signal, &template, Mode::Full).unwrap()
        );
    }

    #[test]
    fn test_template_spectrum_cache_evicts_oldest() {
        let template: Vec<f32> = (0..5).map(|i| ((i * 3 % 5) as f32 - 2.0) / 2.0).collect();
        let mut cached = CorrelationTemplate::new(&template);
        let mut workspace = CorrelationWorkspace::new();
        let mut output = Vec::new();

        // Six signal lengths, each needing a different FFT size.
        let lengths = [4, 12, 28, 60, 124, 252];
        let mut fft_sizes = Vec::new();
        for &len in &lengths {
            let signal: Vec<f32> = (0..len)
                .map(|i| ((i * 17 % 23) as f32 - 11.0) / 11.0)
                .collect();
            workspace.load_signal(&signal);
            workspace
                .correlate(&mut cached, Mode::Full, &mut output)
                .unwrap();
            assert_close(&output, &naive_full_correlation(&signal, &template));
            fft_sizes.push(workspace.fft_size);
            assert!(cached.spectra.len() <= TEMPLATE_SPECTRUM_CACHE_CAPACITY);
        }
        assert_eq!(fft_sizes, vec![8, 16, 32, 64, 128, 256]);

        // The oldest two sizes were evicted, the newest four kept in order.
        let kept: Vec<usize> = cached.spectra.iter().map(|(size, _)| *size).collect();
        assert_eq!(kept, vec![32, 64, 128, 256]);

        // An evicted size is recomputed and still correct.
        let signal: Vec<f32> = (0..4).map(|i| i as f32 - 1.5).collect();
        workspace.load_signal(&signal);
        workspace
            .correlate(&mut cached, Mode::Full, &mut output)
            .unwrap();
        assert_close(&output, &naive_full_correlation(&signal, &template));
        let kept: Vec<usize> = cached.spectra.iter().map(|(size, _)| *size).collect();
        assert_eq!(kept, vec![64, 128, 256, 8]);
    }

    #[test]
    fn test_fft_size_is_independent_of_other_templates() {
        let signal: Vec<f32> = (0..40)
            .map(|i| ((i * 19 % 29) as f32 - 14.0) / 14.0)
            .collect();
        let short: Vec<f32> = (0..6).map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0).collect();
        let long: Vec<f32> = (0..30)
            .map(|i| ((i * 11 % 17) as f32 - 8.0) / 8.0)
            .collect();
        let expected_short = fft_correlate_1d(&signal, &short, Mode::Full).unwrap();
        let expected_long = fft_correlate_1d(&signal, &long, Mode::Full).unwrap();

        let mut workspace = CorrelationWorkspace::new();
        let mut output = Vec::new();
        workspace.load_signal(&signal);
        // Each template gets the FFT size it needs on its own, in any order, so the output is
        // bit-identical to the one-shot correlation.
        let mut fft_sizes = Vec::new();
        for (template, expected) in [
            (&long, &expected_long),
            (&short, &expected_short),
            (&long, &expected_long),
        ] {
            workspace
                .correlate(
                    &mut CorrelationTemplate::new(template),
                    Mode::Full,
                    &mut output,
                )
                .unwrap();
            fft_sizes.push(workspace.fft_size);
            assert_eq!(&output, expected);
            assert_close(&output, &naive_full_correlation(&signal, template));
        }
        // 40 + 30 - 1 = 69 -> 72 (9 * 8); 40 + 6 - 1 = 45 -> 48 (3 * 16).
        assert_eq!(fft_sizes, vec![72, 48, 72]);
    }

    #[test]
    fn test_non_power_of_two_fft_sizes_match_naive() {
        // Output lengths whose FFT size uses each odd factor: 3, 5, 9 and 15.
        for (signal_len, template_len, fft_size) in
            [(40, 6, 48), (70, 7, 80), (60, 10, 72), (100, 15, 120)]
        {
            let signal: Vec<f32> = (0..signal_len)
                .map(|i| ((i * 19 % 29) as f32 - 14.0) / 14.0)
                .collect();
            let template: Vec<f32> = (0..template_len)
                .map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0)
                .collect();
            assert_eq!(fast_fft_size(signal_len + template_len - 1), fft_size);
            let result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
            assert_close(&result, &naive_full_correlation(&signal, &template));
        }
    }
}
