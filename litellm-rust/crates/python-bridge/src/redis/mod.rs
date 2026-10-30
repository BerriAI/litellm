use std::time::Duration;

use litellm_redis_client::{ConnectError, Endpoint, Error, Namespace, RedisClient, RedisConfig};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};

use crate::execution::run_async_value;

#[pyclass(frozen, name = "NativeRedisClient")]
pub(crate) struct NativeRedisClient(RedisClient);

#[pyclass(frozen, get_all, name = "NativeRedisFailure")]
pub(crate) struct NativeRedisFailure {
    kind: &'static str,
    message: String,
    timeout: bool,
}

#[pymethods]
impl NativeRedisClient {
    #[new]
    #[pyo3(signature = (url, *, namespace=None, timeout_seconds))]
    fn new(url: String, namespace: Option<String>, timeout_seconds: f64) -> PyResult<Self> {
        let timeout = Duration::try_from_secs_f64(timeout_seconds)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        RedisClient::new(RedisConfig {
            endpoint: Endpoint::Url(url),
            namespace: namespace.map(Namespace::new).unwrap_or_default(),
            timeout,
            breaker: None,
        })
        .map(Self)
        .map_err(|error: ConnectError| PyValueError::new_err(error.to_string()))
    }

    fn get<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        run_async_value(py, async move {
            let reply = client.get::<Vec<u8>>(&key).await;
            Python::attach(|py| match reply {
                Ok(Some(value)) => Ok(PyBytes::new(py, &value).into_any().unbind()),
                Ok(None) => Ok(py.None()),
                Err(error) => failure(py, &error),
            })
        })
    }

    #[pyo3(signature = (key, value, *, ttl_ms, nx))]
    fn set<'py>(
        &self,
        py: Python<'py>,
        key: String,
        value: Vec<u8>,
        ttl_ms: u64,
        nx: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let ttl = Duration::from_millis(ttl_ms);
        run_async_value(py, async move {
            let reply = match nx {
                true => client.set_nx(&key, value, ttl).await,
                false => client.set(&key, value, ttl).await.map(|()| true),
            };
            Python::attach(|py| match reply {
                Ok(written) => Ok(written.into_pyobject(py)?.to_owned().into_any().unbind()),
                Err(error) => failure(py, &error),
            })
        })
    }

    fn delete<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        run_async_value(py, async move {
            let reply = client.del(&[&key]).await;
            Python::attach(|py| match reply {
                Ok(deleted) => Ok(deleted.into_pyobject(py)?.into_any().unbind()),
                Err(error) => failure(py, &error),
            })
        })
    }
}

fn failure(py: Python<'_>, error: &Error) -> PyResult<Py<PyAny>> {
    let (kind, timeout) = match error {
        Error::Transport(source) => ("transport", source.is_timeout()),
        Error::Server(_) => ("server", false),
        Error::Decode(_) => ("decode", false),
        Error::BreakerOpen => ("breaker_open", false),
    };
    let message = match error {
        Error::Transport(source) | Error::Server(source) | Error::Decode(source) => {
            format!("{error}: {source}")
        }
        Error::BreakerOpen => error.to_string(),
    };
    Ok(Py::new(
        py,
        NativeRedisFailure {
            kind,
            message,
            timeout,
        },
    )?
    .into_any())
}
