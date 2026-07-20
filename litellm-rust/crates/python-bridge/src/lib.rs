use std::collections::HashMap;
use std::time::Duration;

use litellm_ai_gateway::io::messages::{MessagesRequest, messages as run_messages};
use litellm_ai_gateway::io::ocr::{OcrRequest, ocr as run_ocr};
use litellm_ai_gateway::io::responses_ws::ResponsesWebSocketConnection as RustResponsesWebSocketConnection;
use litellm_core::error::CoreError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::{Map, Value};

mod gil;

type MarshaledOcrInputs = (
    Value,
    Option<Map<String, Value>>,
    Map<String, Value>,
    Option<Duration>,
);

fn py_to_json(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let json = py.import("json")?;
    let encoded: String = json.call_method1("dumps", (value,))?.extract()?;
    serde_json::from_str(&encoded).map_err(|err| PyValueError::new_err(err.to_string()))
}

fn json_to_py(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    let json = py.import("json")?;
    let encoded =
        serde_json::to_string(&value).map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(json.call_method1("loads", (encoded,))?.unbind())
}

fn core_error_to_pyerr(err: CoreError) -> PyErr {
    match err {
        CoreError::Auth(message) => PyValueError::new_err(message),
        CoreError::InvalidProvider(_)
        | CoreError::InvalidRequest(_)
        | CoreError::InvalidType { .. }
        | CoreError::MissingField(_) => PyValueError::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn optional_object_to_map(
    py: Python<'_>,
    name: &'static str,
    value: Option<Py<PyAny>>,
) -> PyResult<Map<String, Value>> {
    match value {
        Some(value) => match py_to_json(py, value.bind(py))? {
            Value::Object(map) => Ok(map),
            _ => Err(PyValueError::new_err(format!("{name} must be a dict"))),
        },
        None => Ok(Map::new()),
    }
}

fn optional_timeout(timeout_seconds: Option<f64>) -> Option<Duration> {
    timeout_seconds.and_then(|secs| {
        if secs.is_finite() && secs > 0.0 {
            Some(Duration::from_secs_f64(secs))
        } else {
            None
        }
    })
}

fn marshal_headers(
    py: Python<'_>,
    headers: Option<Py<PyAny>>,
) -> PyResult<HashMap<String, String>> {
    let value = match headers {
        Some(headers) => py_to_json(py, headers.bind(py))?,
        None => Value::Object(Map::new()),
    };
    let Value::Object(headers) = value else {
        return Err(PyValueError::new_err("headers must be a dict"));
    };
    headers
        .into_iter()
        .map(|(name, value)| {
            value
                .as_str()
                .map(|value| (name, value.to_string()))
                .ok_or_else(|| PyValueError::new_err("header values must be strings"))
        })
        .collect()
}

#[pyclass]
struct ResponsesWebSocketConnection {
    inner: RustResponsesWebSocketConnection,
}

#[pymethods]
impl ResponsesWebSocketConnection {
    #[classmethod]
    #[pyo3(signature = (url, headers=None, timeout_seconds=None))]
    fn connect<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        url: String,
        headers: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers = marshal_headers(py, headers)?;
        let timeout = optional_timeout(timeout_seconds);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let inner = RustResponsesWebSocketConnection::connect_url(&url, &headers, timeout)
                .await
                .map_err(core_error_to_pyerr)?;
            Python::attach(|py| Py::new(py, ResponsesWebSocketConnection { inner }))
        })
    }

    fn send_text<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.send_text(text).await.map_err(core_error_to_pyerr)
        })
    }

    fn recv_text<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.recv_text().await.map_err(core_error_to_pyerr)
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.close().await.map_err(core_error_to_pyerr)
        })
    }
}

fn marshal_inputs(
    py: Python<'_>,
    document: Py<PyAny>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<MarshaledOcrInputs> {
    let document = py_to_json(py, document.bind(py))?;
    let extra_headers = match extra_headers {
        Some(headers) => Some(optional_object_to_map(py, "extra_headers", Some(headers))?),
        None => None,
    };
    let optional_params = optional_object_to_map(py, "optional_params", optional_params)?;
    let timeout = optional_timeout(timeout_seconds);

    Ok((document, extra_headers, optional_params, timeout))
}

#[pyfunction]
#[pyo3(signature = (model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn ocr(
    py: Python<'_>,
    model: String,
    document: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let (document, extra_headers, optional_params, timeout) = marshal_inputs(
        py,
        document,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;

    let result = gil::release_gil(py, || {
        pyo3_async_runtimes::tokio::get_runtime().block_on(run_ocr(OcrRequest {
            model: &model,
            document,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            optional_params,
            timeout,
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: Default::default(),
            litellm_call_id: None,
        }))
    });

    match result {
        Ok(value) => json_to_py(py, value),
        Err(err) => Err(core_error_to_pyerr(err)),
    }
}

#[pyfunction]
#[pyo3(signature = (model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn aocr(
    py: Python<'_>,
    model: String,
    document: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'_, PyAny>> {
    let (document, extra_headers, optional_params, timeout) = marshal_inputs(
        py,
        document,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let value = run_ocr(OcrRequest {
            model: &model,
            document,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            optional_params,
            timeout,
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: Default::default(),
            litellm_call_id: None,
        })
        .await
        .map_err(core_error_to_pyerr)?;

        Python::attach(|py| json_to_py(py, value))
    })
}

type MarshaledMessagesInputs = (Value, Option<Map<String, Value>>, Option<Duration>);

fn marshal_messages_inputs(
    py: Python<'_>,
    body: Py<PyAny>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<MarshaledMessagesInputs> {
    let body = py_to_json(py, body.bind(py))?;
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
#[pyo3(signature = (model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
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
) -> PyResult<Py<PyAny>> {
    let (body, extra_headers, timeout) =
        marshal_messages_inputs(py, body, extra_headers, timeout_seconds)?;

    let result = gil::release_gil(py, || {
        pyo3_async_runtimes::tokio::get_runtime().block_on(run_messages(MessagesRequest {
            model: &model,
            body,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        }))
    });

    match result {
        Ok(value) => json_to_py(py, value),
        Err(err) => Err(core_error_to_pyerr(err)),
    }
}

#[pyfunction]
#[pyo3(signature = (model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
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
) -> PyResult<Bound<'_, PyAny>> {
    let (body, extra_headers, timeout) =
        marshal_messages_inputs(py, body, extra_headers, timeout_seconds)?;

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let value = run_messages(MessagesRequest {
            model: &model,
            body,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        })
        .await
        .map_err(core_error_to_pyerr)?;

        Python::attach(|py| json_to_py(py, value))
    })
}

#[pyfunction]
fn gil_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let stats = PyDict::new(py);
    stats.set_item("releases", gil::release_count())?;
    Ok(stats.into_any().unbind())
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(aocr, module)?)?;
    module.add_function(wrap_pyfunction!(messages, module)?)?;
    module.add_function(wrap_pyfunction!(amessages, module)?)?;
    module.add_class::<ResponsesWebSocketConnection>()?;
    module.add_function(wrap_pyfunction!(gil_stats, module)?)?;
    Ok(())
}
