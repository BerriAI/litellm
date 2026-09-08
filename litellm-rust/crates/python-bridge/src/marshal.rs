use std::collections::HashMap;
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

pub(crate) fn required_value(
    name: &'static str,
    value: Value,
    expected: fn(&Value) -> bool,
    expected_name: &'static str,
) -> PyResult<Value> {
    if expected(&value) {
        return Ok(value);
    }
    Err(PyTypeError::new_err(format!(
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

pub(crate) fn marshal_headers(headers: Option<Value>) -> PyResult<HashMap<String, String>> {
    let value = match headers {
        Some(headers) => headers,
        None => Value::Object(Map::new()),
    };
    let Value::Object(headers) = value else {
        return Err(PyTypeError::new_err("headers must be a dict"));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_rejects_values_that_cannot_form_a_positive_duration() {
        Python::initialize();
        Python::attach(|py| {
            for timeout in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
                let error = optional_timeout(Some(timeout)).expect_err("timeout must be rejected");
                assert!(error.is_instance_of::<PyValueError>(py));
            }
        });
    }

    #[test]
    fn timeout_accepts_none_and_positive_finite_values() {
        assert_eq!(optional_timeout(None).unwrap(), None);
        assert_eq!(
            optional_timeout(Some(1.5)).unwrap(),
            Some(Duration::from_millis(1500))
        );
    }
}
