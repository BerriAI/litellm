use litellm_core::chat_completions::Error;
use std::future::Future;

use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::chat_completions::{
    chat_completions as run_chat_completions, chat_completions_decline_reason,
};
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::chat_completions_error_to_pyerr;
use crate::execution::{run_async, run_sync};
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty, required_array};

struct ChatCompletionsInputs {
    model: String,
    messages: Value,
    optional_params: Option<Value>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Value>,
    timeout_seconds: Option<f64>,
}

fn prepare_chat_completions(
    inputs: ChatCompletionsInputs,
) -> PyResult<impl Future<Output = Result<ChatCompletionsResponse, Error>> + Send + 'static> {
    let messages = required_array("messages", inputs.messages)?;
    let optional_params = object_or_empty("optional_params", inputs.optional_params)?;
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model: inputs.model,
        api_key: inputs.api_key,
        api_base: inputs.api_base,
        custom_llm_provider: inputs.custom_llm_provider,
        extra_headers: inputs.extra_headers,
        timeout_seconds: inputs.timeout_seconds,
    })?;

    Ok(async move {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        run_chat_completions(ChatCompletionsRequest {
            model: &model,
            messages: Value::Array(messages),
            optional_params: optional_params.into(),
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        })
        .await
    })
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, custom_llm_provider=None))]
fn chat_completions_decline(
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] messages: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    custom_llm_provider: Option<String>,
) -> PyResult<Option<String>> {
    let optional_params = object_or_empty("optional_params", optional_params)?.into();
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
    #[pyo3(from_py_with = litellm_python_interop::from_py)] messages: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    run_sync(
        py,
        prepare_chat_completions(ChatCompletionsInputs {
            model,
            messages,
            optional_params,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout_seconds,
        })?,
        chat_completions_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn achat_completions(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] messages: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    run_async(
        py,
        prepare_chat_completions(ChatCompletionsInputs {
            model,
            messages,
            optional_params,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout_seconds,
        })?,
        chat_completions_error_to_pyerr,
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    super::super::add_function(module, wrap_pyfunction!(chat_completions_decline, module)?)?;
    super::super::add_function(module, wrap_pyfunction!(chat_completions, module)?)?;
    super::super::add_function(module, wrap_pyfunction!(achat_completions, module)?)
}
