use crate::errors::chat_completions_error_to_pyerr;
use crate::marshal::{NativeRequestContext, NativeRequestOptions, required_value};
use litellm_core::Error;
use litellm_core::chat_completions::chat_completions as run_route;
use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::native_outcome::NativeOutcome;
use litellm_core::request_context::LiteLlmRequestContext;
use litellm_core::request_options::RequestOptions;
use pyo3::prelude::*;
use serde_json::{Map, Value};
use std::future::Future;

enum ChatCompletionsRouteError {
    Declined(String),
    Terminal(Error),
}

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
) -> PyResult<
    impl Future<Output = Result<ChatCompletionsResponse, ChatCompletionsRouteError>> + Send + 'static,
> {
    let context: LiteLlmRequestContext = context.into();
    let messages = required_value("messages", input.messages, Value::is_array, "list")?;
    let options: RequestOptions = options.into();
    Ok(async move {
        match run_route(
            ChatCompletionsRequest {
                model: &input.model,
                messages,
                optional_params: input.optional_params,
            },
            &options,
            &context,
        )
        .await
        .map_err(ChatCompletionsRouteError::Terminal)?
        {
            NativeOutcome::Completed(response) => Ok(response),
            NativeOutcome::Declined(decline) => Err(ChatCompletionsRouteError::Declined(
                decline.reason().to_string(),
            )),
        }
    })
}

fn chat_completions_route_error_to_pyerr(error: ChatCompletionsRouteError) -> PyErr {
    match error {
        ChatCompletionsRouteError::Declined(reason) => {
            crate::errors::RustBridgeDeclined::new_err(reason)
        }
        ChatCompletionsRouteError::Terminal(error) => chat_completions_error_to_pyerr(error),
    }
}

bridge_route! {
    sync = chat_completions,
    asynchronous = achat_completions,
    request = ChatCompletionsInputs,
    prepare = prepare_chat_completions,
    errors = chat_completions_route_error_to_pyerr,
}
