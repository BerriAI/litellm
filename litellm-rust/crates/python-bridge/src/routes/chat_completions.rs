use pyo3::types::{PyDict, PyTuple};

use crate::errors::RustBridgeDeclined;
use crate::logger::{run_async, run_sync};
use litellm_core::chat_completions::{
    Error, chat_completions as run_chat_completions, chat_completions_decline_reason,
    types::ChatCompletionsRequest,
};
use litellm_host_python::from_py_argument;
use litellm_http::HttpClientConfig;
use litellm_types::utils::ChatCompletionsResponse;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::{
    errors::chat_completions_error_to_pyerr,
    marshal::{
        RouteOptions, extra_headers_argument, messages_argument, optional_params_argument,
        optional_timeout,
    },
};

async fn execute(
    config: HttpClientConfig,
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
    run_chat_completions(
        crate::http::pool(),
        &config,
        ChatCompletionsRequest {
            model: &model,
            messages: Value::Array(messages),
            optional_params,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        },
    )
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
    let config = crate::http::call_config(py, &PyDict::new(py), false)?;
    run_sync(
        py,
        execute(
            config,
            messages,
            optional_params.unwrap_or_default(),
            options,
        ),
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
    let config = crate::http::call_config(py, &PyDict::new(py), true)?;
    run_async(
        py,
        execute(
            config,
            messages,
            optional_params.unwrap_or_default(),
            options,
        ),
        chat_completions_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (request, args, kwargs))]
pub(crate) fn completion(
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    drop((request, args, kwargs));
    Err(RustBridgeDeclined::new_err(
        "native chat completions route is not implemented",
    ))
}

#[pyfunction]
#[pyo3(signature = (request, args, kwargs))]
pub(crate) fn acompletion(
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    drop((request, args, kwargs));
    Err(RustBridgeDeclined::new_err(
        "native chat completions route is not implemented",
    ))
}

#[cfg(test)]
mod tests {
    use pyo3::{
        prelude::*,
        types::{PyDict, PyList, PyTuple},
    };

    use crate::errors::RustBridgeDeclined;

    #[test]
    fn both_entrypoints_decline_before_provider_execution() {
        Python::initialize();
        Python::attach(|py| {
            let request = PyDict::new(py);
            let args = PyTuple::empty(py);
            let kwargs = PyDict::new(py);

            for entrypoint in [super::completion, super::acompletion] {
                let error = entrypoint(request.clone().into_any(), args.clone(), kwargs.clone())
                    .expect_err(
                        "native chat completions must decline until a route machine exists",
                    );
                assert!(error.is_instance_of::<RustBridgeDeclined>(py));
            }
        });
    }

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
