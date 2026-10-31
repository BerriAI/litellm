use std::time::Duration;

use litellm_redis_client::RedisClient;
use litellm_redis_store::{
    Error,
    router_cooldowns::{self, Cooldown},
};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

use super::{NativeRedisClient, NativeRedisFailure, failure};
use crate::execution::{run_async_value, run_sync_value};

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct CooldownEntry {
    exception_received: String,
    status_code: String,
    timestamp: f64,
    cooldown_time: f64,
}

#[pyclass(frozen, name = "NativeRouterCooldowns")]
pub(crate) struct NativeRouterCooldowns(RedisClient);

#[pymethods]
impl NativeRouterCooldowns {
    #[new]
    fn new(client: &NativeRedisClient) -> Self {
        Self(client.0.clone())
    }

    #[pyo3(signature = (model_id, entry, *, ttl_seconds))]
    fn put(
        &self,
        py: Python<'_>,
        model_id: String,
        entry: CooldownEntry,
        ttl_seconds: u64,
    ) -> PyResult<Option<NativeRedisFailure>> {
        let client = self.0.clone();
        let cooldown = Cooldown {
            exception_received: entry.exception_received,
            status_code: entry.status_code,
            timestamp: entry.timestamp,
            cooldown_time: entry.cooldown_time,
        };
        let ttl = Duration::from_secs(ttl_seconds);
        let written = run_sync_value(py, async move {
            Ok(router_cooldowns::put(&client, &model_id, &cooldown, ttl).await)
        })?;
        Ok(written.err().map(|error| failure(&error)))
    }

    fn get_many(&self, py: Python<'_>, model_ids: Vec<String>) -> PyResult<Py<PyAny>> {
        let client = self.0.clone();
        let read = run_sync_value(py, async move { Ok(get_many(&client, &model_ids).await) })?;
        read_to_py(py, read)
    }

    fn aget_many<'py>(&self, py: Python<'py>, model_ids: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        run_async_value(py, async move {
            let read = get_many(&client, &model_ids).await;
            Python::attach(|py| read_to_py(py, read))
        })
    }
}

type Read = Result<Vec<Option<Result<Cooldown, Error>>>, Error>;

async fn get_many(client: &RedisClient, model_ids: &[String]) -> Read {
    let model_ids: Vec<&str> = model_ids.iter().map(String::as_str).collect();
    router_cooldowns::get_many(client, &model_ids).await
}

fn read_to_py(py: Python<'_>, read: Read) -> PyResult<Py<PyAny>> {
    let entries = match read {
        Ok(entries) => entries,
        Err(error) => return Ok(Py::new(py, failure(&error))?.into_any()),
    };
    let items = entries
        .into_iter()
        .map(|entry| match entry {
            None => Ok(py.None()),
            Some(Ok(cooldown)) => cooldown_to_py(py, &cooldown),
            Some(Err(error)) => Ok(Py::new(py, failure(&error))?.into_any()),
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any().unbind())
}

fn cooldown_to_py(py: Python<'_>, cooldown: &Cooldown) -> PyResult<Py<PyAny>> {
    let entry = PyDict::new(py);
    entry.set_item("exception_received", &cooldown.exception_received)?;
    entry.set_item("status_code", &cooldown.status_code)?;
    entry.set_item("timestamp", cooldown.timestamp)?;
    entry.set_item("cooldown_time", cooldown.cooldown_time)?;
    Ok(entry.into_any().unbind())
}
