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
  - Bounded thread-local FFT plan caching for optimal performance
  - O(N log N) complexity vs O(N*M) for naive sliding window
  - Zero-copy where possible

- **Python bindings**:
  - Optional PyO3 extension module
  - Buildable with `maturin`
  - Python API mirrors the Rust `fft_correlate_1d` entrypoint
  - Ships a `.pyi` stub for Pyright and other type checkers

- **Correct indexing**: Follows scipy.signal.correlate convention where output index k corresponds to the lag where `template[M-1]` aligns with `signal[k]`

## Installation

### Rust

Add this to your `Cargo.toml` (replace tag with a github version):

```toml
[dependencies]
fft-correlation = { git = "https://github.com/andrewtheguy/fft-correlation", tag = "0.0.0" }
```

### Python

Install from the GitHub Pages package index (replace tag with a github version):

```bash
pip install --extra-index-url https://andrewtheguy.github.io/fft-correlation/simple/ fft-correlation==0.0.0
```

Or with `uv`:

```bash
uv pip install --extra-index-url https://andrewtheguy.github.io/fft-correlation/simple/ fft-correlation==0.0.0
```

#### Development install

To build and install the extension module from source with `maturin`:

```bash
python -m pip install maturin numpy
maturin develop
```

That installs a module named `fft_correlation`.

## Usage

### Rust

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

### Python

```python
import fft_correlation
import numpy as np

signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
template = np.array([1.0, 0.0, 0.0], dtype=np.float32)

full = fft_correlation.fft_correlate_1d(signal, template, mode="full")
same = fft_correlation.fft_correlate_1d(signal, template, mode=fft_correlation.SAME)
valid = fft_correlation.fft_correlate_1d(signal, template, mode="valid")

print(type(full))  # numpy.ndarray
print(full.dtype)  # float32
print(len(full))   # 7
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

### Rust

Run the Rust test suite:

```bash
cargo test
```

### Python bindings

The Python test loads the compiled extension module directly from `target/debug` or `target/release`, so you do not need to install a wheel just to verify the binding.

Build the extension and run the Python test:

```bash
python3 -m pip install numpy
cargo build --features python --lib
python3 -m unittest tests.test_python_bindings
```

If the extension lives somewhere else, point the test runner at it with `FFT_CORRELATION_PYTHON_MODULE=/path/to/module.so`.

### Coverage

The test suite includes:
- Output length validation for all modes
- Correctness verification against naive sliding window implementation
- Edge cases (empty inputs, single elements, template longer than signal)
- Signal processing tests (chirp signals, sinusoids, autocorrelation)
- Numerical accuracy tests
- Python binding smoke coverage for exported constants, mode parsing, result correctness, and error translation

## References

- [scipy.signal.correlate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)
- [numpy.correlate](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html)
- Oppenheim & Schafer, "Discrete-Time Signal Processing" (Correlation Theorem)

## License

This project is licensed under the MIT License
