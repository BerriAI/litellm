use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use litellm_core::auth::InputSource;
use litellm_python_interop::from_py_preserving_errors as from_py;

pub(crate) fn optional_timeout(timeout_seconds: Option<f64>) -> Option<Duration> {
    timeout_seconds.and_then(|secs| {
        if secs.is_finite() && secs > 0.0 {
            Some(Duration::from_secs_f64(secs))
        } else {
            None
        }
    })
}

pub(crate) fn python_timeout_seconds(py: Python<'_>, timeout: Py<PyAny>) -> PyResult<Option<f64>> {
    py.import("litellm.rust_bridge.timeouts")?
        .getattr("timeout_to_seconds")?
        .call1((timeout,))?
        .extract()
}

pub(crate) fn project_optional_fields(
    kwargs: &Bound<'_, PyDict>,
    names: &[&str],
) -> PyResult<Map<String, Value>> {
    names
        .iter()
        .filter_map(|name| match kwargs.get_item(name) {
            Ok(Some(value)) => Some(from_py(&value).map(|value| ((*name).to_string(), value))),
            Ok(None) => None,
            Err(error) => Some(Err(error)),
        })
        .collect()
}

struct RequestFieldSources<'py> {
    body: Option<Bound<'py, PyAny>>,
    credentials: Option<Bound<'py, PyAny>>,
}

impl<'py> RequestFieldSources<'py> {
    fn extract(proxy_request: &Bound<'py, PyAny>) -> PyResult<Self> {
        let proxy_request = proxy_request.cast::<PyDict>()?;
        let body = proxy_request
            .get_item("body_fields")?
            .or(proxy_request.get_item("body")?);
        let credentials = proxy_request.get_item("credential_fields")?;
        Ok(Self { body, credentials })
    }

    fn contains(&self, name: &str) -> bool {
        self.body
            .as_ref()
            .is_some_and(|fields| fields.contains(name).unwrap_or(false))
            || self
                .credentials
                .as_ref()
                .is_some_and(|fields| fields.contains(name).unwrap_or(false))
    }
}

pub(crate) fn request_input_sources<'a>(
    kwargs: &Bound<'_, PyDict>,
    names: impl Iterator<Item = &'a str>,
) -> PyResult<BTreeMap<String, InputSource>> {
    let Some(proxy_request) = kwargs.get_item("proxy_server_request")? else {
        return Ok(BTreeMap::new());
    };
    let sources = RequestFieldSources::extract(&proxy_request)?;
    Ok(names
        .filter(|name| sources.contains(name))
        .map(|name| (name.to_string(), InputSource::Request))
        .collect())
}

pub(crate) fn marshal_headers(headers: Option<Value>) -> PyResult<HashMap<String, String>> {
    let value = headers.unwrap_or_else(|| Value::Object(Map::new()));
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
