use crate::errors::core_error_to_pyerr;
use crate::marshal::{NativeRequestContext, NativeRequestOptions, required_value};
use litellm_core::Error;
use litellm_core::messages::messages as run_route;
use litellm_core::messages::types::{AnthropicMessagesResponse, MessagesRequest};
use litellm_core::request_context::LiteLlmRequestContext;
use pyo3::prelude::*;
use serde_json::Value;
use std::future::Future;

#[derive(FromPyObject)]
struct MessagesInputs {
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    body: Value,
}

fn prepare_messages(
    input: MessagesInputs,
    options: NativeRequestOptions,
    context: NativeRequestContext,
) -> PyResult<impl Future<Output = Result<AnthropicMessagesResponse, Error>> + Send + 'static> {
    if let Some(reason) = messages_decline(
        &input.model,
        input.options.provider("anthropic"),
        input
            .body
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        false,
        false,
        input.body.get("response_format").and_then(Value::as_str),
    ) {
        return Err(crate::errors::RustBridgeDeclined::new_err(reason));
    }
    let context: LiteLlmRequestContext = context.into();
    let body = required_value("body", input.body, Value::is_object, "dict")?;
    Ok(async move {
        run_route(
            MessagesRequest {
                model: &input.model,
                body,
            },
            &options.into(),
            &context,
        )
        .await
    })
}

#[pyfunction]
#[pyo3(signature = (_model, custom_llm_provider, *, stream=false, has_agentic_hook=false, has_custom_client=false, request_format=None))]
fn messages_decline(
    _model: &str,
    custom_llm_provider: &str,
    stream: bool,
    has_agentic_hook: bool,
    has_custom_client: bool,
    request_format: Option<&str>,
) -> Option<String> {
    super::definition::request_decline(
        litellm_core::messages::messages_provider_supported(custom_llm_provider),
        stream,
        has_agentic_hook,
        has_custom_client,
        request_format,
    )
}

bridge_route! {
    sync = messages,
    asynchronous = amessages,
    request = MessagesInputs,
    prepare = prepare_messages,
    errors = core_error_to_pyerr,
    extra = [messages_decline],
}
