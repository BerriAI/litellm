use std::time::Duration;

use litellm_ai_gateway::io::ocr::{ocr as run_ocr, OcrRequest};
use litellm_core::error::CoreError;
use pyo3::exceptions::PyValueError;
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

fn core_error_to_pyerr(py: Python<'_>, err: CoreError) -> PyErr {
    let status_code = err.public_status_code();
    let message = err.to_string();
    build_rust_ocr_error(py, &message, status_code).unwrap_or_else(|import_err| import_err)
}

/// Raise the typed `litellm.ocr.rust_bridge.RustOcrError` carrying the public
/// status so the Python host can map it to the matching public exception.
fn build_rust_ocr_error(
    py: Python<'_>,
    message: &str,
    status_code: Option<u16>,
) -> PyResult<PyErr> {
    let exc_type = py
        .import("litellm.ocr.rust_bridge")?
        .getattr("RustOcrError")?;
    let instance = exc_type.call1((message, status_code))?;
    Ok(PyErr::from_value(instance))
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

    let result = gil::release_gil(py, || {
        pyo3_async_runtimes::tokio::get_runtime().block_on(run_ocr(OcrRequest {
            model: &model,
            document,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: &custom_llm_provider,
            extra_headers,
            optional_params,
            timeout,
        }))
    });

    match result {
        Ok(value) => json_to_py(py, value),
        Err(err) => Err(core_error_to_pyerr(py, err)),
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
    let custom_llm_provider = custom_llm_provider.unwrap_or_else(|| "mistral".to_string());
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
            custom_llm_provider: &custom_llm_provider,
            extra_headers,
            optional_params,
            timeout,
        })
        .await
        .map_err(|err| Python::with_gil(|py| core_error_to_pyerr(py, err)))?;

        Python::with_gil(|py| json_to_py(py, value))
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
    module.add_function(wrap_pyfunction!(gil_stats, module)?)?;
    Ok(())
}
