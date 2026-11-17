use http::Method;
use pyo3::{exceptions::PyValueError, prelude::*};

/// Upper-cased, as `RouteChecks.get_request_method` hands it over.
pub(crate) fn parse_method(text: &str) -> PyResult<Method> {
    Method::from_bytes(text.to_ascii_uppercase().as_bytes())
        .map_err(|error| PyValueError::new_err(format!("invalid request method {text:?}: {error}")))
}
