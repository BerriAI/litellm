use pyo3::exceptions::PyBaseException;
use pyo3::prelude::*;

use litellm_core::ocr::LiteLLMOcrResponse;
use litellm_python_interop::to_py;

pub(super) fn response(py: Python<'_>, response: &LiteLLMOcrResponse) -> PyResult<Py<PyAny>> {
    py.import("litellm.rust_bridge.ocr.callbacks")?
        .getattr("response")?
        .call1((to_py(py, response)?,))
        .map(Bound::unbind)
}

pub(super) fn map_failure(
    py: Python<'_>,
    error: &Py<PyBaseException>,
    request: &Bound<'_, PyAny>,
    provider: &str,
) -> PyResult<Py<PyBaseException>> {
    Ok(py
        .import("litellm.rust_bridge.ocr.callbacks")?
        .getattr("map_failure")?
        .call1((error, request, provider))?
        .extract()?)
}
