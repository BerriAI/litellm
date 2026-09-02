use std::time::Duration;

use litellm_core::chat_completions::types::ChatCompletionsRequest;
use litellm_core::chat_completions::{
    chat_completions as run_chat_completions, chat_completions_decline_reason,
};
use litellm_python_interop::from_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::fallback_route_error_to_pyerr;
use crate::marshal::{optional_object_to_map, optional_timeout};

use super::runtime::{run_async, run_sync};

type MarshaledChatCompletionsInputs = (
    Value,
    Map<String, Value>,
    Option<Map<String, Value>>,
    Option<Duration>,
);

fn marshal_chat_completions_inputs(
    py: Python<'_>,
    messages: Py<PyAny>,
    optional_params: Option<Py<PyAny>>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<MarshaledChatCompletionsInputs> {
    let messages: Value = from_py(messages.bind(py))?;
    if !messages.is_array() {
        return Err(PyValueError::new_err("messages must be a list"));
    }
    let optional_params = optional_object_to_map(py, "optional_params", optional_params)?;
    let extra_headers = match extra_headers {
        Some(headers) => Some(optional_object_to_map(py, "extra_headers", Some(headers))?),
        None => None,
    };
    Ok((
        messages,
        optional_params,
        extra_headers,
        optional_timeout(timeout_seconds),
    ))
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
#[pyo3(signature = (model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
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
) -> PyResult<Py<PyAny>> {
    let (messages, optional_params, extra_headers, timeout) = marshal_chat_completions_inputs(
        py,
        messages,
        optional_params,
        extra_headers,
        timeout_seconds,
    )?;

    run_sync(
        py,
        async move {
            run_chat_completions(ChatCompletionsRequest {
                model: &model,
                messages,
                optional_params,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: custom_llm_provider.as_deref(),
                extra_headers,
                timeout,
            })
            .await
        },
        fallback_route_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
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
) -> PyResult<Bound<'_, PyAny>> {
    let (messages, optional_params, extra_headers, timeout) = marshal_chat_completions_inputs(
        py,
        messages,
        optional_params,
        extra_headers,
        timeout_seconds,
    )?;

    run_async(
        py,
        async move {
            run_chat_completions(ChatCompletionsRequest {
                model: &model,
                messages,
                optional_params,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: custom_llm_provider.as_deref(),
                extra_headers,
                timeout,
            })
            .await
        },
        fallback_route_error_to_pyerr,
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(chat_completions_decline, module)?)?;
    module.add_function(wrap_pyfunction!(chat_completions, module)?)?;
    module.add_function(wrap_pyfunction!(achat_completions, module)?)
}
