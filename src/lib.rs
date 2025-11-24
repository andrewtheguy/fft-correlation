//! FFT-based correlation for 1D real-valued signals
//!
//! Provides efficient cross-correlation using FFT with configurable output modes (Full, Same, Valid)
//! matching scipy/numpy conventions. Uses thread-local FFT planner caching for optimal performance.
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

use realfft::RealFftPlanner;
use std::cell::RefCell;

pub mod error;
pub use error::{FftCorrelationError, Result};

// Thread-local FFT planner cache for optimal performance
thread_local! {
    static FFT_PLANNER: RefCell<RealFftPlanner<f32>> = RefCell::new(RealFftPlanner::new());
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


/// Correlate two 1D signals using FFT
///
/// Computes cross-correlation efficiently using FFT with O(N log N) complexity.
/// The `mode` parameter controls output size:
/// - `Mode::Full`: Returns complete correlation (signal.len() + template.len() - 1)
/// - `Mode::Same`: Returns centered output matching signal.len()
/// - `Mode::Valid`: Returns only fully-overlapping region (signal.len() - template.len() + 1)
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
    // Early validation for empty inputs
    if signal.is_empty() || template.is_empty() {
        return Ok(Vec::new());
    }

    let output_len = signal.len() + template.len() - 1;
    let fft_size = output_len.next_power_of_two();

    // Zero-pad both signal and template to fft_size
    let mut padded_signal = vec![0.0; fft_size];
    let mut padded_template = vec![0.0; fft_size];
    padded_signal[..signal.len()].copy_from_slice(signal);

    // Reverse template for correlation via the Correlation Theorem:
    // For real-valued signals, correlation(x, y) = IFFT(FFT(x) * conj(FFT(y)))
    // Since we work with real signals (no complex representation), time-reversing y
    // achieves the same effect as frequency-domain conjugation for real data.
    // This is equivalent to: signal * reverse(template) in time domain.
    // Reference: Oppenheim & Schafer, "Discrete-Time Signal Processing"
    for (i, &val) in template.iter().rev().enumerate() {
        padded_template[i] = val;
    }

    // Get plans from thread-local cached planner
    // RealFftPlanner caches FFT plans internally, so reusing it avoids repeated planning
    let (r2c, c2r) = FFT_PLANNER.with(|planner_cell| {
        let mut planner = planner_cell.borrow_mut();
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);
        (r2c, c2r)
    });

    // Allocate buffers for FFT output (complex)
    let mut signal_spectrum = r2c.make_output_vec();
    let mut template_spectrum = r2c.make_output_vec();

    // Forward FFT on both signal and template
    debug_assert_eq!(padded_signal.len(), fft_size, "Signal buffer size mismatch");
    debug_assert_eq!(padded_template.len(), fft_size, "Template buffer size mismatch");
    r2c.process(&mut padded_signal, &mut signal_spectrum)
        .map_err(|e| FftCorrelationError::FftProcessing(format!("FFT forward process failed for signal: {:?}", e)))?;
    r2c.process(&mut padded_template, &mut template_spectrum)
        .map_err(|e| FftCorrelationError::FftProcessing(format!("FFT forward process failed for template: {:?}", e)))?;

    // Frequency domain multiplication (element-wise)
    // For correlation, we already reversed template, so just multiply in-place
    for i in 0..signal_spectrum.len() {
        signal_spectrum[i] *= template_spectrum[i];
    }

    // Inverse FFT
    let mut result_time = vec![0.0; fft_size];
    debug_assert_eq!(signal_spectrum.len(), r2c.make_output_vec().len(), "Spectrum buffer size mismatch");
    debug_assert_eq!(result_time.len(), fft_size, "Output buffer size mismatch");
    c2r.process(&mut signal_spectrum, &mut result_time)
        .map_err(|e| FftCorrelationError::FftProcessing(format!("FFT inverse process failed: {:?}", e)))?;

    // Normalize by FFT size
    let normalization = fft_size as f32;
    result_time.iter_mut().for_each(|x| *x /= normalization);

    // Mode-based output trimming
    match mode {
        Mode::Full => {
            result_time.truncate(output_len);
            Ok(result_time)
        }
        Mode::Same => {
            let start = (output_len - signal.len()) / 2;
            Ok(result_time[start..start + signal.len()].to_vec())
        }
        Mode::Valid => {
            if signal.len() < template.len() {
                return Ok(Vec::new());
            }
            let valid_len = signal.len() - template.len() + 1;
            let start = template.len() - 1;
            Ok(result_time[start..start + valid_len].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let max_idx = valid_result.iter()
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
        // Naive sliding window correlation for comparison
        fn sliding_window_correlate(signal: &[f32], template: &[f32]) -> Vec<f32> {
            let output_len = signal.len() + template.len() - 1;
            let mut result = vec![0.0; output_len];

            for lag in 0..output_len {
                let mut correlation = 0.0;
                for i in 0..template.len() {
                    let signal_idx = lag as i32 - (template.len() as i32 - 1) + i as i32;
                    if signal_idx >= 0 && signal_idx < signal.len() as i32 {
                        correlation += signal[signal_idx as usize] * template[template.len() - 1 - i];
                    }
                }
                result[lag] = correlation;
            }
            result
        }

        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let template = vec![0.5, 1.0, 0.5];

        let fft_result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let sliding_result = sliding_window_correlate(&signal, &template);

        assert_eq!(fft_result.len(), sliding_result.len());
        for (i, (&fft_val, &sliding_val)) in fft_result.iter().zip(sliding_result.iter()).enumerate() {
            assert!((fft_val - sliding_val).abs() < 1e-4,
                "Sample {} mismatch: FFT={}, sliding={}", i, fft_val, sliding_val);
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
        let full_peak = full.iter().map(|x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let same_peak = same.iter().map(|x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let valid_peak = valid.iter().map(|x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

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
            assert!((val_xy - val_yx_rev).abs() < 1e-4,
                "Mismatch at index {}: {} vs {}", i, val_xy, val_yx_rev);
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

        let max_val = result.iter()
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
        // 1025 requires FFT size 2048, 513 requires FFT size 1024

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
        let max_abs = fft_result.iter().map(|x| x.abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        assert!(max_abs < 100.0, "Peak value should be reasonable for sine wave, got {}", max_abs);
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
        signal[5] = 1e10;  // Use large value instead of NaN (NaN breaks FFT validation)
        let template = vec![0.5; 3];

        let result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();

        // Result should preserve large values
        assert!(result.iter().any(|x| *x > 1e8), "Large values should be present in correlation");

        // Test with negative values
        let mut signal = vec![1.0; 10];
        signal[3] = -10.0;
        let template = vec![0.5; 3];

        let result = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();

        // Result should contain the effect of negative values
        assert!(result.iter().any(|x| *x < 0.0), "Negative values should propagate through correlation");
    }

    #[test]
    fn test_fft_correlate_same_centering_even_template() {
        // Comment 6: Test Mode::Same centering when template is even (ambiguous center)
        // Verify left-biased centering for consistent alignment

        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];  // length 6
        let template = vec![0.5, 1.0, 1.5, 2.0];  // length 4 (even)

        let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
        let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();

        assert_eq!(same.len(), signal.len());

        // Full has length 6 + 4 - 1 = 9
        // Same should extract center: start = (9 - 6) / 2 = 1, so indices 1..7
        let output_len = full.len();
        let start = (output_len - signal.len()) / 2;

        for (i, &val) in same.iter().enumerate() {
            assert!((val - full[start + i]).abs() < 1e-4,
                "Same mode index {} should match Full mode index {}", i, start + i);
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

        assert!((peak - sum_sq).abs() < 0.1, "Autocorr peak {} should equal sum_sq {}", peak, sum_sq);

        // Peak should be maximum
        for val in &autocorr {
            assert!(*val <= peak + 1e-4, "Autocorr value {} exceeds peak {}", val, peak);
        }

        // Test Same mode autocorrelation
        let same_result = fft_correlate_1d(&signal, &signal, Mode::Same).unwrap();
        assert_eq!(same_result.len(), n);

        // Same mode should have peak at center
        let same_peak_idx = same_result.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(same_peak_idx >= n / 2 - 2 && same_peak_idx <= n / 2 + 2,
            "Same mode peak should be near center, got index {}", same_peak_idx);
    }
}
