use std::future::Future;

use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::chat_completions::{
    chat_completions as run_chat_completions, chat_completions_decline_reason,
};
use litellm_core::error::CoreResult;
use litellm_python_interop::from_py;
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::fallback_route_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty, required_value};

fn prepare_chat_completions(
    py: Python<'_>,
    inputs: ChatCompletionsInputs,
) -> PyResult<impl Future<Output = CoreResult<ChatCompletionsResponse>> + Send + 'static> {
    let messages = required_value(py, "messages", inputs.messages, Value::is_array, "list")?;
    let optional_params = object_or_empty(py, "optional_params", inputs.optional_params)?;
    let options = RouteOptions::from_python(
        py,
        RouteOptionsInputs {
            model: inputs.model,
            api_key: inputs.api_key,
            api_base: inputs.api_base,
            custom_llm_provider: inputs.custom_llm_provider,
            extra_headers: inputs.extra_headers,
            timeout_seconds: inputs.timeout_seconds,
        },
    )?;

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
            messages,
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
    let optional_params = object_or_empty(py, "optional_params", optional_params)?;
    Ok(chat_completions_decline_reason(
        &model,
        custom_llm_provider.as_deref(),
        messages,
        &optional_params,
    )
    .map(str::to_string))
}

bridge_route! {
    sync = chat_completions,
    asynchronous = achat_completions,
    inputs = ChatCompletionsInputs,
    required = {
        model: String,
        messages: Py<PyAny>,
    },
    optional = {
        optional_params: Option<Py<PyAny>>,
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        extra_headers: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    },
    prepare = prepare_chat_completions,
    errors = fallback_route_error_to_pyerr,
    extra = [chat_completions_decline],
}
