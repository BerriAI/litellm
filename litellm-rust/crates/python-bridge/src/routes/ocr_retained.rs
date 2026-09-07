use pyo3::prelude::*;

use super::retained_http;

#[pyfunction]
fn ocr_retained(boundary: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    retained_http::run_sync(boundary)
}

#[pyfunction]
fn aocr_retained(boundary: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    retained_http::run_async(boundary)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr_retained, module)?)?;
    module.add_function(wrap_pyfunction!(aocr_retained, module)?)
}
