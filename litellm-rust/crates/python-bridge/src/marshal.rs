use std::time::Duration;

use pyo3::exceptions::{PyTypeError, PyValueError};
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
            timeout: optional_timeout(inputs.timeout_seconds)?,
        })
    }
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
        _ => Err(PyTypeError::new_err(format!("{name} must be a dict"))),
    }
}

pub(crate) fn optional_timeout(timeout_seconds: Option<f64>) -> PyResult<Option<Duration>> {
    timeout_seconds
        .map(|seconds| {
            if seconds <= 0.0 {
                return Err(PyValueError::new_err(
                    "timeout_seconds must be greater than zero",
                ));
            }
            Duration::try_from_secs_f64(seconds)
                .map_err(|_| PyValueError::new_err("timeout_seconds must be a finite duration"))
        })
        .transpose()
}

#[cfg(test)]
#[path = "../tests/unit/marshal.rs"]
mod tests;
