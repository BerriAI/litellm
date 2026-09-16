use litellm_core::messages::Error;
use litellm_core::messages::messages as run_messages;
use litellm_core::messages::types::{AnthropicMessagesResponse, MessagesRequest};
use pyo3::prelude::*;
use serde_json::Value;
use std::future::Future;

use crate::errors::core_error_to_pyerr;
use crate::execution::{run_async, run_sync};
use crate::marshal::{RouteOptions, RouteOptionsInputs, required_object};

struct MessagesInputs {
    model: String,
    body: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Value>,
    timeout_seconds: Option<f64>,
}

fn prepare_messages(
    inputs: MessagesInputs,
) -> PyResult<impl Future<Output = Result<AnthropicMessagesResponse, Error>> + Send + 'static> {
    let body = required_object("body", inputs.body)?;
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
            body: Value::Object(body),
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        })
        .await
    })
}

#[pyfunction]
#[pyo3(signature = (model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn messages(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] body: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    run_sync(
        py,
        prepare_messages(MessagesInputs {
            model,
            body,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout_seconds,
        })?,
        core_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn amessages(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] body: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    run_async(
        py,
        prepare_messages(MessagesInputs {
            model,
            body,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout_seconds,
        })?,
        core_error_to_pyerr,
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    super::super::add_function(module, wrap_pyfunction!(messages, module)?)?;
    super::super::add_function(module, wrap_pyfunction!(amessages, module)?)
}
