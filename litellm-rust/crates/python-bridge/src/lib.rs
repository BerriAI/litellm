use std::sync::Arc;
use std::time::Duration;

use litellm_ai_gateway::io::ocr::run_ocr;
use litellm_ai_gateway::io::realtime::{dial_url, UpstreamHandle};
use litellm_core::error::CoreError;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::{Map, Value};

mod gil;

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
        CoreError::InvalidType { .. } | CoreError::MissingField(_) => {
            PyValueError::new_err(err.to_string())
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Perform a Mistral OCR call end to end and return the response as a dict.
#[pyfunction]
#[pyo3(signature = (model, document, api_key=None, api_base=None, optional_params=None, timeout_seconds=None))]
fn ocr(
    py: Python<'_>,
    model: String,
    document: Py<PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    optional_params: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let document = py_to_json(py, document.bind(py))?;

    let optional_params = match optional_params {
        Some(params) => match py_to_json(py, params.bind(py))? {
            Value::Object(map) => map,
            _ => return Err(PyValueError::new_err("optional_params must be a dict")),
        },
        None => Map::new(),
    };

    let timeout = timeout_seconds.and_then(|secs| {
        if secs.is_finite() && secs > 0.0 {
            Some(Duration::from_secs_f64(secs))
        } else {
            None
        }
    });

    // Release the GIL during the blocking HTTP call (counted for observability).
    let result = gil::release_gil(py, || {
        run_ocr(
            &model,
            document,
            api_key.as_deref(),
            api_base.as_deref(),
            optional_params,
            timeout,
        )
    });

    match result {
        Ok(value) => json_to_py(py, value),
        Err(err) => Err(core_error_to_pyerr(err)),
    }
}

/// Bridge GIL accounting, e.g. `{"releases": 12}`. Lets the Python side observe
/// how often the bridge has dropped the GIL for blocking work.
#[pyfunction]
fn gil_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let stats = PyDict::new(py);
    stats.set_item("releases", gil::release_count())?;
    Ok(stats.into_any().unbind())
}

fn core_error_to_runtime(err: CoreError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Async upstream WebSocket handle returned by [`realtime_connect`].
///
/// Quacks just enough like `websockets.asyncio.client.ClientConnection` for
/// LiteLLM's `RealTimeStreaming` to drive it: an async `send(text)`, an async
/// iterator yielding text frames (via `recv()` / `__anext__`), and an async
/// `close()`. Sends and receives can be awaited concurrently because the
/// underlying split halves live behind independent tokio mutexes.
#[pyclass]
struct RustRealtimeUpstream {
    inner: Arc<UpstreamHandle>,
}

#[pymethods]
impl RustRealtimeUpstream {
    /// Send a single text frame upstream. Raises `RuntimeError` on a closed
    /// or broken socket so callers can map it to their preferred Python
    /// exception type.
    fn send<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.send_text(text).await.map_err(core_error_to_runtime)
        })
    }

    /// Pull the next text frame. Raises `StopAsyncIteration` on a clean close
    /// so it composes with `async for`, and `RuntimeError` on a transport
    /// failure.
    fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            match inner.recv_text().await {
                Ok(Some(text)) => Ok(text),
                Ok(None) => Err(PyStopAsyncIteration::new_err("upstream closed")),
                Err(err) => Err(core_error_to_runtime(err)),
            }
        })
    }

    /// Close the upstream socket. Idempotent: closing an already-closed handle
    /// is a no-op so `__aexit__` and an explicit `close()` can both run.
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

/// Dial an upstream realtime WebSocket with the caller-supplied URL and
/// headers; returns a [`RustRealtimeUpstream`] handle.
///
/// Python keeps owning URL construction (query params, transcription `intent`,
/// `OpenAI-Beta` passthrough, custom SSL) and just hands the resolved values
/// to Rust. The connection itself, including TLS, runs on tokio inside Rust.
#[pyfunction]
fn realtime_connect<'py>(
    py: Python<'py>,
    url: String,
    headers: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let headers_vec: Vec<(String, String)> = headers
        .iter()
        .map(|(key, value)| Ok((key.extract::<String>()?, value.extract::<String>()?)))
        .collect::<PyResult<_>>()?;

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let upstream = dial_url(&url, &headers_vec)
            .await
            .map_err(core_error_to_runtime)?;
        Ok(RustRealtimeUpstream {
            inner: Arc::new(UpstreamHandle::new(upstream)),
        })
    })
}

#[pymodule]
fn litellm_python_bridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(gil_stats, module)?)?;
    module.add_function(wrap_pyfunction!(realtime_connect, module)?)?;
    module.add_class::<RustRealtimeUpstream>()?;
    Ok(())
}
