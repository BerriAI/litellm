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
    Ok(async move {
        run_route(
            ChatCompletionsRequest {
                model: &input.model,
                messages,
                optional_params: input.optional_params,
            },
            &options.into(),
            &context,
        )
        .await
    })
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, custom_llm_provider=None, *, options, context))]
#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 preserves chat preflight inputs alongside separated options and context"
)]
fn chat_completions_decline(
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] messages: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    custom_llm_provider: Option<String>,
    options: NativeRequestOptions,
    context: NativeRequestContext,
) -> PyResult<Option<String>> {
    let context: LiteLlmRequestContext = context.into();
    let options: RequestOptions = options.into();
    let optional_params = match optional_params {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "optional_params must be a dict",
            ));
        }
    };
    Ok(chat_completions_decline_reason(
        &model,
        custom_llm_provider.as_deref(),
        messages,
        &optional_params,
        &options,
        &context,
    )
    .map(str::to_string))
}

bridge_route! {
    sync = chat_completions,
    asynchronous = achat_completions,
    request = ChatCompletionsInputs,
    prepare = prepare_chat_completions,
    errors = chat_completions_error_to_pyerr,
    extra = [chat_completions_decline],
}
