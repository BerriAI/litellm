use std::collections::HashMap;
use std::time::Duration;

use litellm_python_interop::from_py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

pub(crate) struct RouteOptions {
    pub(crate) model: String,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) custom_llm_provider: Option<String>,
    pub(crate) extra_headers: Option<Map<String, Value>>,
    pub(crate) timeout: Option<Duration>,
}

impl RouteOptions {
    pub(crate) fn from_python(
        py: Python<'_>,
        model: String,
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        extra_headers: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers: optional_object(py, "extra_headers", extra_headers)?,
            timeout: optional_timeout(timeout_seconds),
        })
    }
}

pub(crate) fn required_value(
    py: Python<'_>,
    name: &'static str,
    value: Py<PyAny>,
    expected: fn(&Value) -> bool,
    expected_name: &'static str,
) -> PyResult<Value> {
    let value = from_py(value.bind(py))?;
    if expected(&value) {
        return Ok(value);
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a {expected_name}"
    )))
}

pub(crate) fn object_or_empty(
    py: Python<'_>,
    name: &'static str,
    value: Option<Py<PyAny>>,
) -> PyResult<Map<String, Value>> {
    match value {
        Some(value) => object(py, name, value),
        None => Ok(Map::new()),
    }
}

fn optional_object(
    py: Python<'_>,
    name: &'static str,
    value: Option<Py<PyAny>>,
) -> PyResult<Option<Map<String, Value>>> {
    value.map(|value| object(py, name, value)).transpose()
}

fn object(py: Python<'_>, name: &'static str, value: Py<PyAny>) -> PyResult<Map<String, Value>> {
    match from_py(value.bind(py))? {
        Value::Object(map) => Ok(map),
        _ => Err(PyValueError::new_err(format!("{name} must be a dict"))),
    }
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

pub(crate) fn marshal_headers(
    py: Python<'_>,
    headers: Option<Py<PyAny>>,
) -> PyResult<HashMap<String, String>> {
    let value = match headers {
        Some(headers) => from_py(headers.bind(py))?,
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
