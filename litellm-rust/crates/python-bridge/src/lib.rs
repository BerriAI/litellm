use std::sync::Arc;
use std::time::Duration;

use litellm_core::error::CoreError;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
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

/// Map a core error to the closest Python exception. Caller-input problems
/// (auth, bad types, missing fields) -> `ValueError`; everything else
/// (network, upstream status, parse failures) -> `RuntimeError`.
fn core_error_to_pyerr(err: CoreError) -> PyErr {
    match err {
        CoreError::Auth(message) => PyValueError::new_err(message),
        CoreError::InvalidProvider(_)
        | CoreError::InvalidType { .. }
        | CoreError::InvalidRequest(_)
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

/// Perform a Mistral OCR call end to end and return the response as a dict.
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
    let custom_llm_provider = custom_llm_provider.unwrap_or_else(|| "mistral".to_string());
    let (document, extra_headers, optional_params, timeout) = marshal_inputs(
        py,
        document,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;

    // Release the GIL while the sync API waits on async Rust work.
    let result = gil::release_gil(py, || {
        pyo3_async_runtimes::tokio::get_runtime().block_on(litellm_providers::ocr::ocr(
            litellm_providers::ocr::OcrRequest {
                model: &model,
                document,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: &custom_llm_provider,
                extra_headers,
                optional_params,
                timeout,
            },
        ))
    });

    match result {
        Ok(value) => json_to_py(py, value),
        Err(err) => Err(core_error_to_pyerr(err)),
    }
}

/// Perform an OCR call end to end and return an asyncio awaitable.
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
    let custom_llm_provider = custom_llm_provider.unwrap_or_else(|| "mistral".to_string());
    let (document, extra_headers, optional_params, timeout) = marshal_inputs(
        py,
        document,
        extra_headers,
        optional_params,
        timeout_seconds,
    )?;

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let value = litellm_providers::ocr::ocr(litellm_providers::ocr::OcrRequest {
            model: &model,
            document,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: &custom_llm_provider,
            extra_headers,
            optional_params,
            timeout,
        })
        .await
        .map_err(core_error_to_pyerr)?;

        Python::with_gil(|py| json_to_py(py, value))
    })
}

/// Bridge GIL accounting, e.g. `{"releases": 12}`. Lets the Python side observe
/// how often the sync bridge has dropped the GIL while awaiting Rust work.
#[pyfunction]
fn gil_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let stats = PyDict::new(py);
    stats.set_item("releases", gil::release_count())?;
    Ok(stats.into_any().unbind())
}

fn core_error_to_runtime(err: CoreError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

fn realtime_payload_to_py(
    py: Python<'_>,
    payload: litellm_providers::realtime::RealtimePayload,
) -> PyResult<Py<PyAny>> {
    match payload {
        litellm_providers::realtime::RealtimePayload::Text(text) => {
            Ok(text.into_pyobject(py)?.into_any().unbind())
        }
        litellm_providers::realtime::RealtimePayload::Binary(bytes) => {
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        }
    }
}

#[pyclass]
struct RustRealtimeUpstream {
    inner: Arc<litellm_providers::realtime::UpstreamHandle>,
}

#[pymethods]
impl RustRealtimeUpstream {
    fn send<'py>(
        &self,
        py: Python<'py>,
        message: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let payload = if let Ok(text) = message.extract::<String>() {
            litellm_providers::realtime::RealtimePayload::Text(text)
        } else if let Ok(bytes) = message.extract::<Vec<u8>>() {
            litellm_providers::realtime::RealtimePayload::Binary(bytes)
        } else {
            return Err(PyValueError::new_err(
                "realtime message must be str or bytes",
            ));
        };
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .send_payload(payload)
                .await
                .map_err(core_error_to_runtime)
        })
    }

    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match inner.recv_payload().await {
                Ok(Some(payload)) => Python::with_gil(|py| realtime_payload_to_py(py, payload)),
                Ok(None) => Err(PyStopAsyncIteration::new_err("upstream closed")),
                Err(err) => Err(core_error_to_runtime(err)),
            }
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.close().await.map_err(core_error_to_runtime)
        })
    }

    fn __aiter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.recv(py)
    }
}

#[pyfunction]
#[pyo3(signature = (url, headers, max_size=None))]
fn realtime_connect<'py>(
    py: Python<'py>,
    url: String,
    headers: Bound<'py, PyDict>,
    max_size: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let headers_vec: Vec<(String, String)> = headers
        .iter()
        .map(|(key, value)| Ok((key.extract::<String>()?, value.extract::<String>()?)))
        .collect::<PyResult<_>>()?;

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let upstream = litellm_providers::realtime::dial_url(&url, &headers_vec, max_size)
            .await
            .map_err(core_error_to_runtime)?;
        Ok(RustRealtimeUpstream {
            inner: Arc::new(litellm_providers::realtime::UpstreamHandle::new(upstream)),
        })
    })
}

#[pymodule]
fn litellm_python_bridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(aocr, module)?)?;
    module.add_function(wrap_pyfunction!(gil_stats, module)?)?;
    module.add_function(wrap_pyfunction!(realtime_connect, module)?)?;
    module.add_class::<RustRealtimeUpstream>()?;
    Ok(())
}
