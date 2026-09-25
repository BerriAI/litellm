use std::future::Future;

use pyo3::prelude::*;
use serde::Serialize;

pub(crate) fn run_sync<T, E, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(E) -> PyErr,
) -> PyResult<Py<PyAny>>
where
    T: Serialize + Send + 'static,
    E: Send + 'static,
    F: Future<Output = Result<T, E>> + Send + 'static,
{
    litellm_host_python::run_sync(py, super::capture(py).instrument(future), map_error)
}

pub(crate) fn run_async<T, E, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(E) -> PyErr,
) -> PyResult<Bound<'_, PyAny>>
where
    T: Serialize + Send + 'static,
    E: Send + 'static,
    F: Future<Output = Result<T, E>> + Send + 'static,
{
    litellm_host_python::run_async(py, super::capture(py).instrument(future), map_error)
}

pub(crate) fn run_sync_value<T, F>(py: Python<'_>, future: F) -> PyResult<T>
where
    T: Send + 'static,
    F: Future<Output = PyResult<T>> + Send + 'static,
{
    litellm_host_python::run_sync_value(py, super::capture(py).instrument(future))
}

pub(crate) fn run_async_value<T, F>(py: Python<'_>, future: F) -> PyResult<Bound<'_, PyAny>>
where
    T: for<'py> IntoPyObject<'py> + Send + 'static,
    F: Future<Output = PyResult<T>> + Send + 'static,
{
    litellm_host_python::run_async_value(py, super::capture(py).instrument(future))
}
