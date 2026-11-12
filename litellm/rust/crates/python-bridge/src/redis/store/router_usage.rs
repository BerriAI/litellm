use std::{collections::HashMap, sync::Arc, time::Duration};

use litellm_redis_store::router_usage::{Counter, Kind, RouterUsage};
use pyo3::prelude::*;

use super::{
    PyNumber, Spelled, kind_error, policy::router_tier_policies, redis_async, redis_sync, remote,
};
use crate::redis::client::NativeRedisClient;

/// `litellm.redis.stores.router_usage.UsageCounter`.
#[derive(FromPyObject)]
struct PyCounter {
    deployment_id: String,
    model: Option<String>,
    minute: String,
    kind: Spelled<Kind>,
}

impl From<PyCounter> for Counter {
    fn from(counter: PyCounter) -> Self {
        Self {
            deployment_id: counter.deployment_id,
            model: counter.model,
            minute: counter.minute,
            kind: counter.kind.0,
        }
    }
}

fn counters(counters: Vec<PyCounter>) -> Vec<Counter> {
    counters.into_iter().map(Counter::from).collect()
}

/// One router's global TPM/RPM counters. The `_sync` methods block with the GIL released, for the
/// Rust router's sync calls, which have no event loop to await on.
#[pyclass(frozen, name = "NativeRouterUsage")]
pub(crate) struct NativeRouterUsage(Arc<RouterUsage>);

#[pymethods]
impl NativeRouterUsage {
    #[new]
    fn new(py: Python<'_>, client: Option<PyRef<'_, NativeRedisClient>>) -> PyResult<Self> {
        let remote = client.map(|client| remote(&client));
        let policy = router_tier_policies(py)?.usage;
        Ok(Self(Arc::new(RouterUsage::new(remote, policy))))
    }

    /// Adds the Redis tier and keeps memory; `false` when one was already attached.
    fn attach(&self, client: PyRef<'_, NativeRedisClient>) -> bool {
        self.0.attach(remote(&client))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        counter: PyCounter,
        value: PyNumber,
        ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counter = Counter::from(counter);
        redis_async(py, "async_increment", kind_error, async move {
            Ok(store
                .increment(&counter, value.0, Duration::from_secs(ttl_seconds))
                .await
                .map(PyNumber))
        })
    }

    /// `_increment_deployment`'s `async_increment_pipeline`, always on its own round trip.
    fn increment_deployment<'py>(
        &self,
        py: Python<'py>,
        deployment_id: String,
        model: Option<String>,
        minute: String,
        tokens: PyNumber,
        requests: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_increment_pipeline", kind_error, async move {
            store
                .increment_deployment(
                    &deployment_id,
                    model.as_deref(),
                    &minute,
                    tokens.0,
                    requests,
                )
                .await
        })
    }

    fn get_many<'py>(
        &self,
        py: Python<'py>,
        counters: Vec<PyCounter>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counters = self::counters(counters);
        redis_async(py, "async_batch_get_cache", kind_error, async move {
            let read = store.get_many(&counters).await?;
            Ok(read
                .into_iter()
                .map(|count| count.map(PyNumber))
                .collect::<Vec<_>>())
        })
    }

    fn increment_sync(
        &self,
        py: Python<'_>,
        counter: PyCounter,
        value: PyNumber,
        ttl_seconds: u64,
    ) -> PyResult<Option<PyNumber>> {
        let store = self.0.clone();
        let counter = Counter::from(counter);
        redis_sync(py, "increment_cache", kind_error, async move {
            Ok(store
                .increment(&counter, value.0, Duration::from_secs(ttl_seconds))
                .await
                .map(PyNumber))
        })
    }

    fn increment_deployment_sync(
        &self,
        py: Python<'_>,
        deployment_id: String,
        model: Option<String>,
        minute: String,
        tokens: PyNumber,
        requests: i64,
    ) -> PyResult<()> {
        let store = self.0.clone();
        redis_sync(py, "increment_pipeline", kind_error, async move {
            store
                .increment_deployment(
                    &deployment_id,
                    model.as_deref(),
                    &minute,
                    tokens.0,
                    requests,
                )
                .await
        })
    }

    fn get_many_sync(
        &self,
        py: Python<'_>,
        counters: Vec<PyCounter>,
    ) -> PyResult<Vec<Option<PyNumber>>> {
        let store = self.0.clone();
        let counters = self::counters(counters);
        redis_sync(py, "batch_get_cache", kind_error, async move {
            let read = store.get_many(&counters).await?;
            Ok(read.into_iter().map(|count| count.map(PyNumber)).collect())
        })
    }

    fn flush_local_counters(&self) -> PyResult<()> {
        self.0.flush_local_counters().map_err(kind_error)
    }

    fn snapshot(&self) -> PyResult<HashMap<String, PyNumber>> {
        Ok(self
            .0
            .snapshot()
            .map_err(kind_error)?
            .into_iter()
            .map(|(key, count)| (key, PyNumber(count)))
            .collect())
    }
}
