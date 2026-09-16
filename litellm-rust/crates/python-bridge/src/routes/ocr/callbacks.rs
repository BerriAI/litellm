use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use litellm_core::ocr::LiteLLMOcrResponse;
use litellm_core::ocr::hooks::OcrDuringCallRequest;
use litellm_python_interop::to_py_preserving_errors as to_py;

use super::host::PythonPayload;
use crate::lifecycle::PythonLogger;

pub(super) fn update_logging(
    py: Python<'_>,
    logger: &PythonLogger,
    kwargs: &Py<PyDict>,
    request: &OcrDuringCallRequest,
    secret_fields: &[&str],
) -> PyResult<()> {
    py.import("litellm.rust_bridge.ocr")?
        .getattr("update_logging")?
        .call1((
            logger.object(py),
            kwargs,
            &request.model,
            &request.custom_llm_provider,
            to_py(py, &request.optional_params)?,
            secret_fields,
            &request.url,
        ))?;
    Ok(())
}

pub(super) fn pre_call(
    py: Python<'_>,
    logger: &PythonLogger,
    request: &OcrDuringCallRequest,
    payload: &PythonPayload,
) -> PyResult<()> {
    py.import("litellm.rust_bridge.ocr")?
        .getattr("pre_call")?
        .call1((
            logger.object(py),
            request.api_key.as_deref(),
            &payload.body,
            &payload.headers,
            &request.url,
        ))?;
    Ok(())
}

pub(super) fn post_call(
    py: Python<'_>,
    logger: &PythonLogger,
    original_response: &Value,
    payload: &PythonPayload,
) -> PyResult<()> {
    py.import("litellm.rust_bridge.ocr")?
        .getattr("post_call")?
        .call1((
            logger.object(py),
            to_py(py, original_response)?,
            &payload.body,
            &payload.headers,
        ))?;
    Ok(())
}

pub(super) fn response(py: Python<'_>, response: &LiteLLMOcrResponse) -> PyResult<Py<PyAny>> {
    py.import("litellm.rust_bridge.ocr")?
        .getattr("build_response")?
        .call1((to_py(py, response)?,))
        .map(Bound::unbind)
}
