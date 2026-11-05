use std::{sync::Arc, time::Duration};

use litellm_redis_store::router_usage::{Counter, RouterUsage};
use pyo3::{prelude::*, types::PyDict};

use super::{PyData, PyNumber, policy::router_tier_policies, redis_async, remote, runtime_error, unwrap_all};
use crate::redis::client::NativeRedisClient;

/// One router's global TPM/RPM counters. Every caller on `main` is async.
#[pyclass(frozen, name = "NativeRouterUsage")]
pub(crate) struct NativeRouterUsage(Arc<RouterUsage>);

#[pymethods]
impl NativeRouterUsage {
    #[new]
    #[pyo3(signature = (client, redis_ttl_seconds = 60))]
    fn new(py: Python<'_>, client: Option<PyRef<'_, NativeRedisClient>>, redis_ttl_seconds: u64) -> PyResult<Self> {
        let remote = client.map(|client| remote(&client, redis_ttl_seconds));
        let policy = router_tier_policies(py)?.usage;
        Ok(Self(Arc::new(RouterUsage::new(remote, policy))))
    }

    /// Adds the Redis tier and keeps memory; `false` when one was already attached.
    #[pyo3(signature = (client, redis_ttl_seconds = 60))]
    fn attach(&self, client: PyRef<'_, NativeRedisClient>, redis_ttl_seconds: u64) -> bool {
        self.0.attach(remote(&client, redis_ttl_seconds))
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        counter: PyData<Counter>,
        value: PyNumber,
        ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_increment", runtime_error, async move {
            store
                .increment(&counter.0, value.0, Duration::from_secs(ttl_seconds))
                .await
                .map(PyNumber)
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
        redis_async(py, "async_increment_pipeline", runtime_error, async move {
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
        counters: Vec<PyData<Counter>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let counters = unwrap_all(counters);
        redis_async(py, "async_batch_get_cache", runtime_error, async move {
            let read = store.get_many(&counters).await?;
            Ok(read
                .into_iter()
                .map(|count| count.map(PyNumber))
                .collect::<Vec<_>>())
        })
    }

    fn flush_local_counters(&self) -> PyResult<()> {
        self.0.flush_local_counters().map_err(runtime_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = PyDict::new(py);
        for (key, count) in self.0.snapshot().map_err(runtime_error)? {
            snapshot.set_item(key, PyNumber(count))?;
        }
        Ok(snapshot)
    }
}
