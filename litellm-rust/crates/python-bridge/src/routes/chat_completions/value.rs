use litellm_core::Error;
use std::future::Future;

use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::chat_completions::{AdmissionContext, chat_completions as run_chat_completions};
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::execution_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty, required_array};

fn prepare_chat_completions(
    inputs: ChatCompletionsInputs,
) -> PyResult<impl Future<Output = Result<ChatCompletionsResponse, Error>> + Send + 'static> {
    let messages = required_array("messages", inputs.messages)?;
    let optional_params = object_or_empty("optional_params", inputs.optional_params)?;
    let context: AdmissionContext = inputs
        .host_facts
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        .unwrap_or_default();
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model: inputs.model,
        api_key: inputs.api_key,
        api_base: inputs.api_base,
        custom_llm_provider: inputs.custom_llm_provider,
        extra_headers: inputs.extra_headers,
        timeout_seconds: inputs.timeout_seconds,
    })?;

    crate::errors::admit(litellm_core::chat_completions::admit(
        &options.model,
        options.custom_llm_provider.as_deref(),
        Value::Array(messages.clone()),
        &optional_params,
        options.extra_headers.as_ref(),
        context,
    ))?;
    if let Some(on_request) = inputs.on_request {
        Python::attach(|py| {
            on_request
                .call0(py)
                .map(|_| ())
                .map_err(|error| crate::errors::host_callback_error(py, error))
        })?;
    }

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
            optional_params,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        })
        .await
    })
}

bridge_route! {
    sync = chat_completions,
    asynchronous = achat_completions,
    inputs = ChatCompletionsInputs,
    required = {
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        messages: serde_json::Value,
    },
    optional = {
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        optional_params: Option<serde_json::Value>,
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<serde_json::Value>,
        timeout_seconds: Option<f64>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        host_facts: Option<serde_json::Value>,
        on_request: Option<Py<PyAny>>,
    },
    prepare = prepare_chat_completions,
    errors = execution_error_to_pyerr,
}
