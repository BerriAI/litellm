use std::time::Duration;

use litellm_core::error::Error;
use litellm_core::messages::messages as run_messages;
use litellm_core::messages::types::{AnthropicMessagesResponse, MessagesRequest};
use litellm_python_interop::from_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::core_error_to_pyerr;
use crate::marshal::{optional_object, optional_timeout};

use super::{block_on, into_py_future};

struct MessagesInputs {
    model: String,
    body: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Map<String, Value>>,
    timeout: Option<Duration>,
}

#[allow(clippy::too_many_arguments)]
fn marshal_inputs(
    py: Python<'_>,
    model: String,
    body: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<MessagesInputs> {
    let body: Value = from_py(body.bind(py))?;
    if !body.is_object() {
        return Err(PyValueError::new_err("body must be a dict"));
    }
    Ok(MessagesInputs {
        model,
        body,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers: optional_object(py, "extra_headers", extra_headers)?,
        timeout: optional_timeout(timeout_seconds),
    })
}

async fn call(inputs: MessagesInputs) -> Result<AnthropicMessagesResponse, Error> {
    run_messages(MessagesRequest {
        model: &inputs.model,
        body: inputs.body,
        api_key: inputs.api_key.as_deref(),
        api_base: inputs.api_base.as_deref(),
        custom_llm_provider: inputs.custom_llm_provider.as_deref(),
        extra_headers: inputs.extra_headers,
        timeout: inputs.timeout,
    })
    .await
}

#[pyfunction]
#[pyo3(signature = (model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None, trace=false))]
#[allow(clippy::too_many_arguments)]
fn messages(
    py: Python<'_>,
    model: String,
    body: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    trace: bool,
) -> PyResult<Py<PyAny>> {
    let inputs = marshal_inputs(
        py,
        model,
        body,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout_seconds,
    )?;
    block_on(py, call(inputs), trace, core_error_to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None, trace=false))]
#[allow(clippy::too_many_arguments)]
fn amessages(
    py: Python<'_>,
    model: String,
    body: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    trace: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let inputs = marshal_inputs(
        py,
        model,
        body,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout_seconds,
    )?;
    into_py_future(py, call(inputs), trace, core_error_to_pyerr)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(messages, module)?)?;
    module.add_function(wrap_pyfunction!(amessages, module)?)
}
