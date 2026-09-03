use litellm_core::Error;
use std::future::Future;

use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::chat_completions::{
    chat_completions as run_chat_completions, chat_completions_decline_reason,
};
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::chat_completions_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty, required_value};

fn prepare_chat_completions(
    inputs: ChatCompletionsInputs,
) -> PyResult<impl Future<Output = Result<ChatCompletionsResponse, Error>> + Send + 'static> {
    let messages = required_value("messages", inputs.messages, Value::is_array, "list")?;
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
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] messages: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    custom_llm_provider: Option<String>,
) -> PyResult<Option<String>> {
    let optional_params = object_or_empty("optional_params", optional_params)?;
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
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        messages: Value,
    },
    optional = {
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        optional_params: Option<Value>,
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<Value>,
        timeout_seconds: Option<f64>,
    },
    prepare = prepare_chat_completions,
    errors = chat_completions_error_to_pyerr,
    extra = [chat_completions_decline],
}
