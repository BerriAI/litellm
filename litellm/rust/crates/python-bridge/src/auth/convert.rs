use http::Method;
use jiff::Timestamp;
use pyo3::{exceptions::PyValueError, prelude::*};

/// A Python epoch float, kept to the microsecond a `datetime` holds. One outside what a `Timestamp`
/// holds saturates.
pub(crate) fn timestamp(epoch: f64) -> Timestamp {
    let micros = (epoch * 1_000_000.0).round() as i64;
    Timestamp::from_microsecond(micros).unwrap_or(if micros < 0 {
        Timestamp::MIN
    } else {
        Timestamp::MAX
    })
}

pub(crate) fn py_timestamp(value: &Bound<'_, PyAny>) -> PyResult<Timestamp> {
    Ok(timestamp(value.extract()?))
}

pub(crate) fn py_optional_timestamp(value: &Bound<'_, PyAny>) -> PyResult<Option<Timestamp>> {
    Ok(value.extract::<Option<f64>>()?.map(timestamp))
}

/// Upper-cased, as `RouteChecks.get_request_method` hands it over.
pub(crate) fn parse_method(text: &str) -> PyResult<Method> {
    Method::from_bytes(text.to_ascii_uppercase().as_bytes())
        .map_err(|error| PyValueError::new_err(format!("invalid request method {text:?}: {error}")))
}

pub(crate) fn py_method(value: &Bound<'_, PyAny>) -> PyResult<Option<Method>> {
    value
        .extract::<Option<String>>()?
        .as_deref()
        .map(parse_method)
        .transpose()
}
