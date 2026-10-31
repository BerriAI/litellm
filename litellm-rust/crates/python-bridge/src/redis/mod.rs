mod parallel_slots;
mod pod_lock;
mod router_cooldowns;
mod spend_counters;

use std::{future::Future, time::Duration};

use litellm_error::Report;
use litellm_redis_client::{Endpoint, Error as RedisError, Namespace, RedisClient, RedisConfig};
use litellm_redis_store::Error;
use litellm_secret_value::SecretValue;
use pyo3::{exceptions::PyValueError, prelude::*};

pub(crate) use parallel_slots::NativeParallelSlots;
pub(crate) use pod_lock::NativePodLock;
pub(crate) use router_cooldowns::NativeRouterCooldowns;
pub(crate) use spend_counters::NativeSpendCounters;

use crate::execution::run_async_value;

#[pyclass(frozen, name = "NativeRedisClient")]
pub(crate) struct NativeRedisClient(RedisClient);

#[pymethods]
impl NativeRedisClient {
    #[new]
    #[pyo3(signature = (url, *, namespace=None, timeout_seconds))]
    fn new(url: String, namespace: Option<String>, timeout_seconds: f64) -> PyResult<Self> {
        let timeout = Duration::try_from_secs_f64(timeout_seconds)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        RedisClient::new(RedisConfig {
            endpoint: Endpoint::Url(SecretValue::new(url)),
            namespace: namespace.map(Namespace::new).unwrap_or_default(),
            timeout,
            breaker: None,
        })
        .map(Self)
        .map_err(|error| PyValueError::new_err(Report::new(&error).to_string()))
    }
}

#[pyclass(frozen, get_all, name = "NativeRedisFailure")]
pub(crate) struct NativeRedisFailure {
    kind: &'static str,
    message: String,
    timeout: bool,
}

pub(crate) fn failure(error: &Error) -> NativeRedisFailure {
    let (kind, timeout) = match error {
        Error::Redis(RedisError::Transport(source)) => ("transport", source.is_timeout()),
        Error::Redis(RedisError::Server(_)) => ("server", false),
        Error::Redis(RedisError::Decode(_)) | Error::Format(_) => ("decode", false),
        Error::Redis(RedisError::BreakerOpen) => ("breaker_open", false),
    };
    NativeRedisFailure {
        kind,
        message: Report::new(error).to_string(),
        timeout,
    }
}

/// Runs one store operation as an awaitable whose result is the value or a `NativeRedisFailure`.
pub(crate) fn store_call<'py, T, F>(
    py: Python<'py>,
    operation: F,
    to_py: fn(Python<'_>, T) -> PyResult<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Send + 'static,
    F: Future<Output = Result<T, Error>> + Send + 'static,
{
    run_async_value(py, async move {
        let result = operation.await;
        Python::attach(|py| match result {
            Ok(value) => to_py(py, value),
            Err(error) => Ok(Py::new(py, failure(&error))?.into_any()),
        })
    })
}
