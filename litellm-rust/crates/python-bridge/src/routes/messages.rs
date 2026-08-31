use litellm_core::error::CoreResult;
use litellm_core::messages::messages as run_messages;
use litellm_core::messages::types::{AnthropicMessagesResponse, MessagesRequest};
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{RouteOptions, required_value};
use crate::routes::BridgeRoute;

struct MessagesCall {
    options: RouteOptions,
    body: Value,
}

impl BridgeRoute<MessagesInputs> for MessagesCall {
    type Output = AnthropicMessagesResponse;

    fn from_python(py: Python<'_>, inputs: MessagesInputs) -> PyResult<Self> {
        Ok(Self {
            options: RouteOptions::from_python(
                py,
                inputs.model,
                inputs.api_key,
                inputs.api_base,
                inputs.custom_llm_provider,
                inputs.extra_headers,
                inputs.timeout_seconds,
            )?,
            body: required_value(py, "body", inputs.body, Value::is_object, "dict")?,
        })
    }

    async fn run(self) -> CoreResult<AnthropicMessagesResponse> {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = self.options;
        run_messages(MessagesRequest {
            model: &model,
            body: self.body,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        })
        .await
    }
}

bridge_route! {
    sync = messages,
    asynchronous = amessages,
    inputs = MessagesInputs,
    required = {
        model: String,
        body: Py<PyAny>,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        extra_headers: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    },
    call = MessagesCall,
    errors = core_error_to_pyerr,
}
