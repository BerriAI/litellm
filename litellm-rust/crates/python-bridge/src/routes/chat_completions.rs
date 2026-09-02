use std::time::Duration;

use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::chat_completions::{
    chat_completions as run_chat_completions, chat_completions_decline_reason,
};
use litellm_core::error::Error;
use litellm_python_interop::from_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::chat_completions_error_to_pyerr;
use crate::marshal::{optional_object, optional_object_to_map, optional_timeout};

use super::{block_on, into_py_future};

struct ChatCompletionsInputs {
    model: String,
    messages: Value,
    optional_params: Map<String, Value>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Map<String, Value>>,
    timeout: Option<Duration>,
}

#[allow(clippy::too_many_arguments)]
fn marshal_inputs(
    py: Python<'_>,
    model: String,
    messages: Py<PyAny>,
    optional_params: Option<Py<PyAny>>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<ChatCompletionsInputs> {
    let messages: Value = from_py(messages.bind(py))?;
    if !messages.is_array() {
        return Err(PyValueError::new_err("messages must be a list"));
    }
    Ok(ChatCompletionsInputs {
        model,
        messages,
        optional_params: optional_object_to_map(py, "optional_params", optional_params)?,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers: optional_object(py, "extra_headers", extra_headers)?,
        timeout: optional_timeout(timeout_seconds),
    })
}

async fn call(inputs: ChatCompletionsInputs) -> Result<ChatCompletionsResponse, Error> {
    run_chat_completions(ChatCompletionsRequest {
        model: &inputs.model,
        messages: inputs.messages,
        optional_params: inputs.optional_params,
        api_key: inputs.api_key.as_deref(),
        api_base: inputs.api_base.as_deref(),
        custom_llm_provider: inputs.custom_llm_provider.as_deref(),
        extra_headers: inputs.extra_headers,
        timeout: inputs.timeout,
    })
    .await
}

/// The decline reason for this request, or `None` when the Rust path accepts
/// it. Resolves no credentials and performs no I/O, so a host can ask before
/// committing to either path.
#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, custom_llm_provider=None))]
fn chat_completions_decline(
    py: Python<'_>,
    model: String,
    messages: Py<PyAny>,
    optional_params: Option<Py<PyAny>>,
    custom_llm_provider: Option<String>,
) -> PyResult<Option<String>> {
    let messages = from_py(messages.bind(py))?;
    let optional_params = optional_object_to_map(py, "optional_params", optional_params)?;
    Ok(chat_completions_decline_reason(
        &model,
        custom_llm_provider.as_deref(),
        messages,
        &optional_params,
    )
    .map(str::to_string))
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None, trace=false))]
#[allow(clippy::too_many_arguments)]
fn chat_completions(
    py: Python<'_>,
    model: String,
    messages: Py<PyAny>,
    optional_params: Option<Py<PyAny>>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    trace: bool,
) -> PyResult<Py<PyAny>> {
    let inputs = marshal_inputs(
        py,
        model,
        messages,
        optional_params,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout_seconds,
    )?;
    block_on(py, call(inputs), trace, chat_completions_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None, trace=false))]
#[allow(clippy::too_many_arguments)]
fn achat_completions(
    py: Python<'_>,
    model: String,
    messages: Py<PyAny>,
    optional_params: Option<Py<PyAny>>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    trace: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let inputs = marshal_inputs(
        py,
        model,
        messages,
        optional_params,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout_seconds,
    )?;
    into_py_future(py, call(inputs), trace, chat_completions_error_to_pyerr)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(chat_completions_decline, module)?)?;
    module.add_function(wrap_pyfunction!(chat_completions, module)?)?;
    module.add_function(wrap_pyfunction!(achat_completions, module)?)
}
