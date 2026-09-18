use litellm_core::messages::Error;
use litellm_core::messages::messages as run_messages;
use litellm_core::messages::types::{AnthropicMessagesResponse, MessagesRequest};
use litellm_host_python::{run_async, run_sync};
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::messages_error_to_pyerr;
use crate::marshal::{RouteOptions, body_argument, extra_headers_argument, optional_timeout};

async fn execute(
    body: Map<String, Value>,
    options: RouteOptions,
) -> Result<AnthropicMessagesResponse, Error> {
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
        body: Value::Object(body),
        api_key: api_key.as_deref(),
        api_base: api_base.as_deref(),
        custom_llm_provider: custom_llm_provider.as_deref(),
        extra_headers,
        timeout,
    })
    .await
}

#[pyfunction]
#[pyo3(signature = (model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn messages(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = body_argument)] body: Map<String, Value>,
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
    run_sync(py, execute(body, options), messages_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn amessages<'py>(
    py: Python<'py>,
    model: String,
    #[pyo3(from_py_with = body_argument)] body: Map<String, Value>,
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
    run_async(py, execute(body, options), messages_error_to_pyerr)
}
