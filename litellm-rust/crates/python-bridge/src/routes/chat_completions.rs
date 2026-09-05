use crate::errors::chat_completions_error_to_pyerr;
use crate::marshal::{NativeRequestContext, NativeRequestOptions, required_value};
use litellm_core::Error;
use litellm_core::chat_completions::chat_completions as run_route;
use litellm_core::chat_completions::chat_completions_decline_reason;
use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::request_context::LiteLlmRequestContext;
use litellm_core::request_options::RequestOptions;
use pyo3::prelude::*;
use serde_json::{Map, Value};
use std::future::Future;

#[derive(FromPyObject)]
struct ChatCompletionsInputs {
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    messages: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    optional_params: Map<String, Value>,
}

fn prepare_chat_completions(
    input: ChatCompletionsInputs,
    options: NativeRequestOptions,
    context: NativeRequestContext,
    _callback_adapter: Option<Py<PyAny>>,
    _python_context: crate::execution::PythonCallContext<'_>,
) -> PyResult<impl Future<Output = Result<ChatCompletionsResponse, Error>> + Send + 'static> {
    let context: LiteLlmRequestContext = context.into();
    let messages = required_value("messages", input.messages, Value::is_array, "list")?;
    let options: RequestOptions = options.into();
    if let Some(reason) = chat_completions_decline_reason(
        &input.model,
        options.custom_llm_provider.as_deref(),
        messages.clone(),
        &input.optional_params,
        &options,
        &context,
    ) {
        return Err(crate::errors::RustBridgeDeclined::new_err(reason));
    }
    Ok(async move {
        run_route(
            ChatCompletionsRequest {
                model: &input.model,
                messages,
                optional_params: input.optional_params,
            },
            &options,
            &context,
        )
        .await
    })
}

bridge_route! {
    sync = chat_completions,
    asynchronous = achat_completions,
    request = ChatCompletionsInputs,
    prepare = prepare_chat_completions,
    errors = chat_completions_error_to_pyerr,
}
