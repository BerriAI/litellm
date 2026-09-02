use std::time::Duration;

use litellm_core::messages::messages as run_messages;
use litellm_core::messages::types::{AnthropicMessagesResponse, MessagesRequest};
use litellm_python_interop::{from_py, release_gil, to_py};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::errors::core_error_to_pyerr;
use crate::function_trace::{TraceResponse, trace_call};
use crate::marshal::{optional_object_to_map, optional_timeout};

fn messages_response_to_py(
    py: Python<'_>,
    response: TraceResponse<AnthropicMessagesResponse>,
) -> PyResult<Py<PyAny>> {
    to_py(py, &response)
}

type MarshaledMessagesInputs = (Value, Option<Map<String, Value>>, Option<Duration>);

fn marshal_messages_inputs(
    py: Python<'_>,
    body: Py<PyAny>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<MarshaledMessagesInputs> {
    let body: Value = from_py(body.bind(py))?;
    if !body.is_object() {
        return Err(PyValueError::new_err("body must be a dict"));
    }
    let extra_headers = match extra_headers {
        Some(headers) => Some(optional_object_to_map(py, "extra_headers", Some(headers))?),
        None => None,
    };
    Ok((body, extra_headers, optional_timeout(timeout_seconds)))
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
    let (body, extra_headers, timeout) =
        marshal_messages_inputs(py, body, extra_headers, timeout_seconds)?;

    let result = release_gil(py, || {
        pyo3_async_runtimes::tokio::get_runtime().block_on(trace_call(
            run_messages(MessagesRequest {
                model: &model,
                body,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: custom_llm_provider.as_deref(),
                extra_headers,
                timeout,
            }),
            trace,
        ))
    });

    match result {
        Ok(response) => messages_response_to_py(py, response),
        Err(err) => Err(core_error_to_pyerr(err)),
    }
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
    let (body, extra_headers, timeout) =
        marshal_messages_inputs(py, body, extra_headers, timeout_seconds)?;

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let response = trace_call(
            run_messages(MessagesRequest {
                model: &model,
                body,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: custom_llm_provider.as_deref(),
                extra_headers,
                timeout,
            }),
            trace,
        )
        .await
        .map_err(core_error_to_pyerr)?;

        Python::attach(|py| messages_response_to_py(py, response))
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(messages, module)?)?;
    module.add_function(wrap_pyfunction!(amessages, module)?)
}
