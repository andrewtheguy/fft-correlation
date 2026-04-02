import importlib.util
import os
import pathlib
import unittest


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _module_path() -> pathlib.Path:
    override = os.environ.get("FFT_CORRELATION_PYTHON_MODULE")
    if override:
        path = pathlib.Path(override)
        if path.exists():
            return path
        raise FileNotFoundError(
            f"FFT_CORRELATION_PYTHON_MODULE points to a missing file: {path}"
        )

    candidates = [
        _repo_root() / "target" / "debug" / "libfft_correlation.so",
        _repo_root() / "target" / "release" / "libfft_correlation.so",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the built fft_correlation extension. "
        "Build it first with `cargo build --features python --lib` or `maturin develop`."
    )


def _load_module():
    module_path = _module_path()
    spec = importlib.util.spec_from_file_location("fft_correlation", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _naive_full_correlation(signal, template):
    output_len = len(signal) + len(template) - 1
    result = [0.0] * output_len

    for lag in range(output_len):
        total = 0.0
        for i, template_value in enumerate(template):
            signal_idx = lag - (len(template) - 1) + i
            if 0 <= signal_idx < len(signal):
                total += signal[signal_idx] * template_value
        result[lag] = total

    return result


class PythonBindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fft_correlation = _load_module()

    def test_exports_mode_constants(self):
        self.assertEqual(self.fft_correlation.FULL, "full")
        self.assertEqual(self.fft_correlation.SAME, "same")
        self.assertEqual(self.fft_correlation.VALID, "valid")

    def test_fft_correlate_matches_naive_full(self):
        signal = [1.0, -2.0, 3.5, 0.5]
        template = [0.5, 1.5, -1.0]

        expected = _naive_full_correlation(signal, template)
        actual = self.fft_correlation.fft_correlate_1d(signal, template, mode="full")

        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=5)

    def test_fft_correlate_same_and_valid_lengths(self):
        signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        template = [1.0, 0.0, 0.0]

        same = self.fft_correlation.fft_correlate_1d(
            signal, template, mode=self.fft_correlation.SAME
        )
        valid = self.fft_correlation.fft_correlate_1d(signal, template, mode="valid")

        self.assertEqual(len(same), len(signal))
        self.assertEqual(len(valid), len(signal) - len(template) + 1)
        self.assertAlmostEqual(same[1], 1.0, places=5)
        self.assertAlmostEqual(valid[0], 1.0, places=5)

    def test_fft_correlate_empty_inputs(self):
        self.assertEqual(self.fft_correlation.fft_correlate_1d([], [1.0]), [])
        self.assertEqual(self.fft_correlation.fft_correlate_1d([1.0], []), [])

    def test_invalid_mode_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            self.fft_correlation.fft_correlate_1d([1.0], [1.0], mode="bogus")

    def test_non_finite_values_raise_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "FFT inverse process failed"):
            self.fft_correlation.fft_correlate_1d([float("nan")], [1.0], mode="full")


if __name__ == "__main__":
    unittest.main()
