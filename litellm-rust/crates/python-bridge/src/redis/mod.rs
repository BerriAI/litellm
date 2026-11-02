mod client;
mod events;
pub(crate) use client::native_redis_breaker_states;
mod policy;
pub(crate) mod router_cooldowns;
pub(crate) mod router_usage;

use litellm_host_python::PythonContext;
use litellm_redis_store::Number;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyBool, PyFloat, PyInt},
};

use self::events::RedisCall;

/// The labels `RedisCache` reports for this operation, in the caller's context so its
/// `service_target` reaches the service event.
fn call(py: Python<'_>, call_type: &'static str) -> PyResult<RedisCall> {
    Ok(RedisCall {
        context: PythonContext::capture(py)?,
        call_type,
        caller: None,
    })
}

fn memory_error(error: litellm_cache::Error) -> PyErr {
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
