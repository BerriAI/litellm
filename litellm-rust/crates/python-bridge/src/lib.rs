use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_providers::ocr::run_ocr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
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

#[pymodule]
fn litellm_python_bridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(gil_stats, module)?)?;
    Ok(())
}
