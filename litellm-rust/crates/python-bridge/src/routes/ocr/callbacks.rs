use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use litellm_core::ocr::LiteLLMOcrResponse;
use litellm_core::ocr::hooks::OcrDuringCallRequest;
use litellm_python_interop::to_py_preserving_errors as to_py;

use litellm_core::call_lifecycle::CallbackFamily;

use super::host::PythonPayload;
use crate::lifecycle::{PythonCallState, PythonLogger};

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
    state: &PythonCallState,
    request: &OcrDuringCallRequest,
    payload: &PythonPayload,
) -> PyResult<()> {
    let logger = state.logger()?;
    if state.supplied {
        let additional_args = PyDict::new(py);
        additional_args.set_item("complete_input_dict", &payload.body)?;
        additional_args.set_item("headers", &payload.headers)?;
        additional_args.set_item("api_base", &request.url)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("input", "OCR document processing")?;
        kwargs.set_item("api_key", request.api_key.as_deref())?;
        kwargs.set_item("additional_args", additional_args)?;
        logger
            .object(py)
            .call_method("pre_call", (), Some(&kwargs))?;
        return Ok(());
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("api_key", request.api_key.as_deref())?;
    kwargs.set_item("body", &payload.body)?;
    kwargs.set_item("headers", &payload.headers)?;
    kwargs.set_item("url", &request.url)?;
    py.import("litellm.rust_bridge.leaves")?
        .getattr("record_pre_call")?
        .call((logger.object(py),), Some(&kwargs))?;
    state.dispatch_request(py, CallbackFamily::RequestPreCall)
}

pub(super) fn post_call(
    py: Python<'_>,
    state: &PythonCallState,
    original_response: &Value,
    payload: &PythonPayload,
) -> PyResult<()> {
    let logger = state.logger()?;
    if state.supplied {
        let additional_args = PyDict::new(py);
        additional_args.set_item("complete_input_dict", &payload.body)?;
        additional_args.set_item("headers", &payload.headers)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("original_response", to_py(py, original_response)?)?;
        kwargs.set_item("additional_args", additional_args)?;
        logger
            .object(py)
            .call_method("post_call", (), Some(&kwargs))?;
        return Ok(());
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("original_response", to_py(py, original_response)?)?;
    kwargs.set_item("body", &payload.body)?;
    kwargs.set_item("headers", &payload.headers)?;
    py.import("litellm.rust_bridge.leaves")?
        .getattr("record_post_call")?
        .call((logger.object(py),), Some(&kwargs))?;
    state.dispatch_request(py, CallbackFamily::RequestPostCall)
}

pub(super) fn response(py: Python<'_>, response: &LiteLLMOcrResponse) -> PyResult<Py<PyAny>> {
    py.import("litellm.rust_bridge.ocr")?
        .getattr("build_response")?
        .call1((to_py(py, response)?,))
        .map(Bound::unbind)
}
