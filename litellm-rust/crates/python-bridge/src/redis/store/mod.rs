pub(crate) mod auth_cache;
mod policy;
pub(crate) mod rate_limits;
pub(crate) mod request;
pub(crate) mod router_cooldowns;
pub(crate) mod router_usage;
pub(crate) mod spend_counters;
mod value;

use std::{
    pin::Pin,
    sync::{Mutex, PoisonError},
    time::Duration,
};

use litellm_host_python::PythonContext;
use litellm_redis_store::{ErrorKind, Number};
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyBool, PyFloat, PyInt},
};

use crate::execution::{run_async_value, run_sync_value};
use crate::redis::events::RedisCall;

/// The labels `RedisCache` reports for this operation, in the caller's context so its
/// `service_target` reaches the service event.
fn call(py: Python<'_>, call_type: &'static str) -> PyResult<RedisCall> {
    Ok(RedisCall {
        context: PythonContext::capture(py)?,
        call_type,
        caller: None,
    })
}

/// Runs one Redis operation as a Python awaitable, reported to `ServiceLogging` as `call_type`.
fn redis_async<'py, T, E: 'static>(
    py: Python<'py>,
    call_type: &'static str,
    error: fn(E) -> PyErr,
    operation: impl Future<Output = Result<T, E>> + Send + 'static,
) -> PyResult<Bound<'py, PyAny>>
where
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    let scope = call(py, call_type)?;
    run_async_value(
        py,
        async move { scope.scope(operation).await.map_err(error) },
    )
}

/// Runs one Redis operation with the GIL released, for the router's sync paths.
fn redis_sync<T, E: 'static>(
    py: Python<'_>,
    call_type: &'static str,
    error: fn(E) -> PyErr,
    operation: impl Future<Output = Result<T, E>> + Send + 'static,
) -> PyResult<T>
where
    T: Send + 'static,
{
    let scope = call(py, call_type)?;
    run_sync_value(
        py,
        async move { scope.scope(operation).await.map_err(error) },
    )
}

fn ttl(seconds: Option<f64>) -> Option<Duration> {
    seconds.map(|seconds| Duration::from_secs_f64(seconds.max(0.0)))
}

type Resolve = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

/// A Redis call declared on a request's batch. `resolve` flushes that batch if it has not gone
/// out and answers the call, once.
#[pyclass(frozen, name = "NativePending")]
pub(crate) struct NativePending {
    resolve: Mutex<Option<Resolve>>,
    call_type: &'static str,
}

impl NativePending {
    fn new<T, E: 'static>(
        call_type: &'static str,
        error: fn(E) -> PyErr,
        reply: impl Future<Output = Result<T, E>> + Send + 'static,
    ) -> Self
    where
        T: for<'a> IntoPyObject<'a> + Send + 'static,
    {
        let resolve = async move {
            let reply = reply.await.map_err(error)?;
            Python::attach(|py| reply.into_py_any(py))
        };
        Self {
            resolve: Mutex::new(Some(Box::pin(resolve))),
            call_type,
        }
    }
}

#[pymethods]
impl NativePending {
    fn resolve<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let resolve = self
            .resolve
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("this Redis call was already resolved"))?;
        redis_async(py, self.call_type, std::convert::identity, resolve)
    }
}

pyo3::create_exception!(
    _native,
    NativeRedisStoreError,
    pyo3::exceptions::PyException,
    "A native Redis store operation failed. Args are (kind, message), kind spelled as litellm-redis-store's ErrorKind."
);

fn kind_error(kind: ErrorKind, error: &(dyn std::error::Error + 'static)) -> PyErr {
    let kind: &'static str = kind.into();
    NativeRedisStoreError::new_err((kind, litellm_error::Report::new(error).to_string()))
}

fn store_error(error: litellm_cache::Error) -> PyErr {
    PyRuntimeError::new_err(litellm_error::Report::new(&error).to_string())
}

/// A Python `int` or `float` across the bridge, keeping which one it is.
pub(crate) struct PyNumber(Number);

impl<'a, 'py> FromPyObject<'a, 'py> for PyNumber {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>() {
            return Ok(Self(Number::Int(value.extract()?)));
        }
        if value.is_instance_of::<PyFloat>() {
            return Ok(Self(Number::Float(value.extract()?)));
        }
        Err(PyTypeError::new_err("expected an int or a float"))
    }
}

impl<'py> IntoPyObject<'py> for PyNumber {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Ok(match self.0 {
            Number::Int(value) => value.into_pyobject(py)?.into_any(),
            Number::Float(value) => value.into_pyobject(py)?.into_any(),
        })
    }
}
