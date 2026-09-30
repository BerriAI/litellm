use litellm_host_python::{ready_future, to_py};
use pyo3::prelude::*;

pub(super) fn ready_none(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    ready_value(py, &())
}

pub(super) fn ready_value<'py, T: serde::Serialize>(
    py: Python<'py>,
    value: &T,
) -> PyResult<Bound<'py, PyAny>> {
    ready_future(py, to_py(py, value)?.bind(py))
}
