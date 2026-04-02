use crate::{fft_correlate_1d as rust_fft_correlate_1d, Mode};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

fn parse_mode(mode: &str) -> PyResult<Mode> {
    if mode.eq_ignore_ascii_case("full") {
        Ok(Mode::Full)
    } else if mode.eq_ignore_ascii_case("same") {
        Ok(Mode::Same)
    } else if mode.eq_ignore_ascii_case("valid") {
        Ok(Mode::Valid)
    } else {
        Err(PyValueError::new_err(
            "mode must be one of: 'full', 'same', 'valid'",
        ))
    }
}

#[pyfunction(name = "fft_correlate_1d", signature = (signal, template, mode = "full"))]
fn fft_correlate_1d_py<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f32>,
    template: PyReadonlyArray1<'py, f32>,
    mode: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let mode = parse_mode(mode)?;
    let signal = signal.as_slice().map_err(|_| {
        PyValueError::new_err("signal must be a contiguous 1D numpy.ndarray with dtype float32")
    })?;
    let template = template.as_slice().map_err(|_| {
        PyValueError::new_err("template must be a contiguous 1D numpy.ndarray with dtype float32")
    })?;

    let result = rust_fft_correlate_1d(signal, template, mode)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    Ok(result.into_pyarray(py))
}

#[pymodule]
#[pyo3(name = "fft_correlation")]
fn fft_correlation(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__all__", vec!["fft_correlate_1d", "FULL", "SAME", "VALID"])?;
    module.add("FULL", "full")?;
    module.add("SAME", "same")?;
    module.add("VALID", "valid")?;
    module.add_function(wrap_pyfunction!(fft_correlate_1d_py, module)?)?;
    Ok(())
}
