use litellm_core::chat_completions::Error;
use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::chat_completions::{
    chat_completions as run_chat_completions, chat_completions_decline_reason,
};
use litellm_host_python::{from_py_argument, run_async, run_sync};
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::chat_completions_error_to_pyerr;
use crate::marshal::{
    RouteOptions, extra_headers_argument, messages_argument, optional_params_argument,
    optional_timeout,
};

async fn execute(
    messages: Vec<Value>,
    optional_params: Map<String, Value>,
    options: RouteOptions,
) -> Result<ChatCompletionsResponse, Error> {
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
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, custom_llm_provider=None))]
pub(crate) fn chat_completions_decline(
    model: String,
    #[pyo3(from_py_with = from_py_argument)] messages: Value,
    #[pyo3(from_py_with = optional_params_argument)] optional_params: Option<Map<String, Value>>,
    custom_llm_provider: Option<String>,
) -> Option<String> {
    chat_completions_decline_reason(
        &model,
        custom_llm_provider.as_deref(),
        messages,
        &optional_params.unwrap_or_default(),
    )
    .map(str::to_string)
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn chat_completions(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = messages_argument)] messages: Vec<Value>,
    #[pyo3(from_py_with = optional_params_argument)] optional_params: Option<Map<String, Value>>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = extra_headers_argument)] extra_headers: Option<Map<String, Value>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let options = RouteOptions {
        model,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout: optional_timeout(timeout_seconds),
    };
    run_sync(
        py,
        execute(messages, optional_params.unwrap_or_default(), options),
        chat_completions_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn achat_completions<'py>(
    py: Python<'py>,
    model: String,
    #[pyo3(from_py_with = messages_argument)] messages: Vec<Value>,
    #[pyo3(from_py_with = optional_params_argument)] optional_params: Option<Map<String, Value>>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = extra_headers_argument)] extra_headers: Option<Map<String, Value>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let options = RouteOptions {
        model,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout: optional_timeout(timeout_seconds),
    };
    run_async(
        py,
        execute(messages, optional_params.unwrap_or_default(), options),
        chat_completions_error_to_pyerr,
    )
}

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    #[test]
    fn chat_completions_decline_keeps_existing_reasons() {
        Python::initialize();
        Python::attach(|py| {
            let decline = crate::native_module(py)
                .getattr("chat_completions_decline")
                .expect("decline helper should be registered");
            let empty = PyList::empty(py);
            let unreadable = py
                .eval(c"'nope'", None, None)
                .expect("string messages should convert");

            let unknown: Option<String> = decline
                .call1(("unknown-model", &empty))
                .and_then(|value| value.extract())
                .expect("unknown providers should decline");
            assert_eq!(
                unknown.as_deref(),
                Some("provider is not on the rust chat completions path")
            );

            let empty_reason: Option<String> = decline
                .call1(("anthropic/claude-sonnet-4-5", &empty))
                .and_then(|value| value.extract())
                .expect("empty lists should decline");
            assert_eq!(empty_reason.as_deref(), Some("empty message list"));

            let unreadable_reason: Option<String> = decline
                .call1(("anthropic/claude-sonnet-4-5", unreadable))
                .and_then(|value| value.extract())
                .expect("non-list messages should decline");
            assert_eq!(
                unreadable_reason.as_deref(),
                Some("unreadable message list")
            );
        });
    }
}
