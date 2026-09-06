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

bridge_route! {
    sync = messages,
    asynchronous = amessages,
    request = MessagesInputs,
    prepare = prepare_messages,
    errors = core_error_to_pyerr,
}
