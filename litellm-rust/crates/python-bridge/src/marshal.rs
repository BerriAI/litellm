use std::collections::HashMap;
use std::time::Duration;

use litellm_python_interop::from_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

pub(crate) fn optional_object_to_map(
    py: Python<'_>,
    name: &'static str,
    value: Option<Py<PyAny>>,
) -> PyResult<Map<String, Value>> {
    match value {
        Some(value) => match from_py(value.bind(py))? {
            Value::Object(map) => Ok(map),
            _ => Err(PyValueError::new_err(format!("{name} must be a dict"))),
        },
        None => Ok(Map::new()),
    }
}

pub(crate) fn optional_object(
    py: Python<'_>,
    name: &'static str,
    value: Option<Py<PyAny>>,
) -> PyResult<Option<Map<String, Value>>> {
    value
        .map(|value| optional_object_to_map(py, name, Some(value)))
        .transpose()
}

pub(crate) fn optional_timeout(timeout_seconds: Option<f64>) -> Option<Duration> {
    timeout_seconds.and_then(|secs| {
        if secs.is_finite() && secs > 0.0 {
            Some(Duration::from_secs_f64(secs))
        } else {
            None
        }
    })
}

pub(crate) fn marshal_headers(headers: Option<Value>) -> PyResult<HashMap<String, String>> {
    let value = match headers {
        Some(headers) => headers,
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
