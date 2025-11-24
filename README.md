# fft-correlation

A Rust library for efficient FFT-based cross-correlation of 1D real-valued signals, matching scipy/numpy conventions.

## Overview

`fft-correlation` provides fast cross-correlation computation using the Fast Fourier Transform (FFT) with O(N log N) complexity. The library follows the conventions established by `scipy.signal.correlate` and `numpy.correlate`, making it a drop-in replacement for scientific computing workflows that need Rust's performance and safety guarantees.

## Features

- **Three output modes** matching scipy/numpy:
  - `Full`: Complete correlation result (length = N + M - 1)
  - `Same`: Centered output matching signal size (length = N)
  - `Valid`: Only fully-overlapping region (length = N - M + 1)

- **High performance**:
  - Thread-local FFT planner caching for optimal performance
  - O(N log N) complexity vs O(N*M) for naive sliding window
  - Zero-copy where possible

- **Correct indexing**: Follows scipy.signal.correlate convention where output index k corresponds to the lag where `template[M-1]` aligns with `signal[k]`

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fft-correlation = "0.1.0"
```

## Usage

```rust
use fft_correlation::{fft_correlate_1d, Mode};

// Create signal and template
let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let template = vec![1.0, 0.0, 0.0];

// Compute correlation with different modes
let full = fft_correlate_1d(&signal, &template, Mode::Full).unwrap();
let same = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();
let valid = fft_correlate_1d(&signal, &template, Mode::Valid).unwrap();

println!("Full mode output length: {}", full.len());   // 7 = 5 + 3 - 1
println!("Same mode output length: {}", same.len());   // 5 (matches signal)
println!("Valid mode output length: {}", valid.len()); // 3 = 5 - 3 + 1
```

### Finding peaks in signals

```rust
use fft_correlation::{fft_correlate_1d, Mode};

// Signal with embedded template
let template = vec![0.5, 1.0, 0.5];
let mut signal = vec![0.0; 100];
signal[50..53].copy_from_slice(&template);

// Correlate to find template location
let result = fft_correlate_1d(&signal, &template, Mode::Same).unwrap();

// Find peak location, filtering out non-finite values (NaN/Inf)
let peak_idx = result.iter()
    .enumerate()
    .filter(|(_, v)| v.is_finite())
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(i, _)| i);

match peak_idx {
    Some(idx) => println!("Template found at index: {}", idx),
    None => println!("No valid peak found (all values are NaN/Inf)"),
}
```

## Mode Semantics

### Full Mode
Returns complete correlation result with length `N + M - 1` where N is signal length and M is template length. Output index k corresponds to the lag where `template[M-1]` aligns with `signal[k]`.

### Same Mode
Returns centered output with length equal to the signal. The center of the Full result is extracted to produce output of the same size as the input signal.

### Valid Mode
Returns only indices where the template fully overlaps the signal, with length `N - M + 1` (or empty if M > N). These represent fully-overlapping windows.

## Performance

The library uses thread-local FFT planner caching to avoid repeated planning overhead. For correlation of signals of length N and M:

- Time complexity: O((N+M) log(N+M))
- Space complexity: O(N+M)
- Naive sliding window: O(N*M)

For large signals or templates, FFT-based correlation is significantly faster than direct convolution.

## Testing

Run the test suite:

```bash
cargo test
```

The test suite includes:
- Output length validation for all modes
- Correctness verification against naive sliding window implementation
- Edge cases (empty inputs, single elements, template longer than signal)
- Signal processing tests (chirp signals, sinusoids, autocorrelation)
- Numerical accuracy tests

## References

- [scipy.signal.correlate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)
- [numpy.correlate](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html)
- Oppenheim & Schafer, "Discrete-Time Signal Processing" (Correlation Theorem)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
