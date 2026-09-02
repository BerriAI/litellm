use std::future::Future;

use litellm_core::error::CoreResult;
use litellm_core::messages::messages as run_messages;
use litellm_core::messages::types::{AnthropicMessagesResponse, MessagesRequest};
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::fallback_route_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, required_value};

fn prepare_messages(
    inputs: MessagesInputs,
) -> PyResult<impl Future<Output = CoreResult<AnthropicMessagesResponse>> + Send + 'static> {
    let body = required_value("body", inputs.body, Value::is_object, "dict")?;
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
        run_messages(MessagesRequest {
            model: &model,
            body,
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
    sync = messages,
    asynchronous = amessages,
    inputs = MessagesInputs,
    required = {
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        body: Value,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<Value>,
        timeout_seconds: Option<f64>,
    },
    prepare = prepare_messages,
    errors = fallback_route_error_to_pyerr,
}
