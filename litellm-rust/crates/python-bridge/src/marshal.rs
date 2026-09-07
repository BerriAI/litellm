use std::collections::HashMap;
use std::time::Duration;

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

pub(crate) struct RouteOptionsInputs {
    pub(crate) model: String,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) custom_llm_provider: Option<String>,
    pub(crate) extra_headers: Option<Value>,
    pub(crate) timeout_seconds: Option<f64>,
}

impl RouteOptions {
    pub(crate) fn from_python(inputs: RouteOptionsInputs) -> PyResult<Self> {
        Ok(Self {
            model: inputs.model,
            api_key: inputs.api_key,
            api_base: inputs.api_base,
            custom_llm_provider: inputs.custom_llm_provider,
            extra_headers: optional_object("extra_headers", inputs.extra_headers)?,
            timeout: optional_timeout(inputs.timeout_seconds),
        })
    }
}

pub(crate) fn required_value(
    name: &'static str,
    value: Value,
    expected: fn(&Value) -> bool,
    expected_name: &'static str,
) -> PyResult<Value> {
    if expected(&value) {
        return Ok(value);
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a {expected_name}"
    )))
}

pub(crate) fn object_or_empty(
    name: &'static str,
    value: Option<Value>,
) -> PyResult<Map<String, Value>> {
    match value {
        Some(value) => object(name, value),
        None => Ok(Map::new()),
    }
}

fn optional_object(
    name: &'static str,
    value: Option<Value>,
) -> PyResult<Option<Map<String, Value>>> {
    value.map(|value| object(name, value)).transpose()
}

fn object(name: &'static str, value: Value) -> PyResult<Map<String, Value>> {
    match value {
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
