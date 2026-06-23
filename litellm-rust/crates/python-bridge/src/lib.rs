//! PyO3 bridge between Python `litellm.ocr` / `litellm.aocr` and the typed,
//! async Rust core.
//!
//! Two entry points, both named to mirror Python:
//!   * `aocr(...)` returns a Python awaitable driven by a Tokio runtime — the
//!     proxy awaits it directly (no thread-per-request executor).
//!   * `ocr(...)` `block_on`s the same future with the GIL released — for sync
//!     SDK callers.
//!
//! The bridge is the only place that touches Python objects: it parses the
//! arguments into a typed [`OcrRequest`] up front (GIL held), then the HTTP work
//! runs GIL-free, and the typed [`OcrResponse`] is converted back to a dict.

use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::ocr::types::{OcrHeaders, OcrParams, OcrProvider, OcrRequest, OcrResponse};
use litellm_providers::ocr as core_ocr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde_json::Value;

mod gil;

/// Round-trip a Python value to JSON via the `json` module, then into a typed
/// `T`. Using `json` (rather than a direct PyO3→serde walk) keeps Python's own
/// serialization rules authoritative and avoids an extra dependency.
fn py_to_typed<T: serde::de::DeserializeOwned>(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    what: &'static str,
) -> PyResult<T> {
    let json = py.import("json")?;
    let encoded: String = json.call_method1("dumps", (value,))?.extract()?;
    let json_value: Value = serde_json::from_str(&encoded)
        .map_err(|err| PyValueError::new_err(format!("invalid {what}: {err}")))?;
    serde_json::from_value(json_value)
        .map_err(|err| PyValueError::new_err(format!("invalid {what}: {err}")))
}

/// Serialize a typed `OcrResponse` back into a Python dict via the `json` module.
fn response_to_py(py: Python<'_>, response: OcrResponse) -> PyResult<Py<PyAny>> {
    let encoded =
        serde_json::to_string(&response).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let json = py.import("json")?;
    Ok(json.call_method1("loads", (encoded,))?.unbind())
}

/// Map a core error to the closest Python exception. Caller-input problems
/// (auth, bad types, missing fields, unsupported provider) -> `ValueError`;
/// everything else (network, upstream status, parse failures) -> `RuntimeError`.
///
/// The `RuntimeError` message preserves the HTTP status (e.g. "status 429"),
/// which the Python `exception_type` layer keys off to raise the right
/// `litellm` exception (RateLimitError, AuthenticationError, …).
fn core_error_to_pyerr(err: CoreError) -> PyErr {
    match err {
        CoreError::Auth(message) => PyValueError::new_err(message),
        CoreError::UnsupportedProvider(_)
        | CoreError::InvalidType { .. }
        | CoreError::MissingField(_) => PyValueError::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn parse_timeout(timeout_seconds: Option<f64>) -> Option<Duration> {
    timeout_seconds.and_then(|secs| {
        if secs.is_finite() && secs > 0.0 {
            Some(Duration::from_secs_f64(secs))
        } else {
            None
        }
    })
}

/// Parse the loose Python arguments into a typed [`OcrRequest`] (GIL held).
#[allow(clippy::too_many_arguments)]
fn build_request(
    py: Python<'_>,
    provider: String,
    model: String,
    document: &Bound<'_, PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<&Bound<'_, PyAny>>,
    timeout_seconds: Option<f64>,
    params: Option<&Bound<'_, PyAny>>,
) -> PyResult<OcrRequest> {
    let provider: OcrProvider = serde_json::from_value(Value::String(provider))
        .map_err(|err| PyValueError::new_err(format!("unknown OCR provider: {err}")))?;

    let document = py_to_typed(py, document, "document")?;

    let extra_headers: OcrHeaders = match extra_headers {
        Some(headers) => py_to_typed(py, headers, "extra_headers")?,
        None => OcrHeaders::new(),
    };

    let params: OcrParams = match params {
        Some(params) => py_to_typed(py, params, "params")?,
        None => OcrParams::default(),
    };

    Ok(OcrRequest {
        provider,
        model,
        document,
        api_key,
        api_base,
        extra_headers,
        timeout: parse_timeout(timeout_seconds),
        params,
    })
}

/// Async OCR — returns a Python awaitable. The HTTP call runs on the Tokio
/// runtime with the GIL released; `litellm.aocr()` simply awaits the result.
#[pyfunction]
#[pyo3(signature = (provider, model, document, api_key=None, api_base=None, extra_headers=None, timeout_seconds=None, params=None))]
#[allow(clippy::too_many_arguments)]
fn aocr<'py>(
    py: Python<'py>,
    provider: String,
    model: String,
    document: Bound<'py, PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<Bound<'py, PyAny>>,
    timeout_seconds: Option<f64>,
    params: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let request = build_request(
        py,
        provider,
        model,
        &document,
        api_key,
        api_base,
        extra_headers.as_ref(),
        timeout_seconds,
        params.as_ref(),
    )?;

    gil::note_offload();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        match core_ocr(request).await {
            Ok(response) => Python::with_gil(|py| response_to_py(py, response)),
            Err(err) => Err(core_error_to_pyerr(err)),
        }
    })
}

/// Sync OCR — `block_on`s the same async core on the shared Tokio runtime with
/// the GIL released, so other Python threads keep running during the HTTP wait.
#[pyfunction]
#[pyo3(signature = (provider, model, document, api_key=None, api_base=None, extra_headers=None, timeout_seconds=None, params=None))]
#[allow(clippy::too_many_arguments)]
fn ocr(
    py: Python<'_>,
    provider: String,
    model: String,
    document: Bound<'_, PyAny>,
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<Bound<'_, PyAny>>,
    timeout_seconds: Option<f64>,
    params: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let request = build_request(
        py,
        provider,
        model,
        &document,
        api_key,
        api_base,
        extra_headers.as_ref(),
        timeout_seconds,
        params.as_ref(),
    )?;

    let runtime = pyo3_async_runtimes::tokio::get_runtime();
    // Release the GIL for the whole blocking await (counted for observability).
    let result = gil::release_gil(py, || runtime.block_on(core_ocr(request)));

    match result {
        Ok(response) => response_to_py(py, response),
        Err(err) => Err(core_error_to_pyerr(err)),
    }
}

/// Bridge GIL accounting, e.g. `{"releases": 12}`. Counts both the sync
/// `block_on` releases and the async `aocr` offloads — i.e. every call whose
/// HTTP work ran off the GIL.
#[pyfunction]
fn gil_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let stats = PyDict::new(py);
    stats.set_item("releases", gil::release_count())?;
    Ok(stats.into_any().unbind())
}

#[pymodule]
fn litellm_python_bridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(aocr, module)?)?;
    module.add_function(wrap_pyfunction!(gil_stats, module)?)?;
    Ok(())
}
