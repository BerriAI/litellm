use std::sync::Arc;

use litellm_redis_store::{
    Error,
    deployment_health::{DeploymentHealth, HealthStates},
};
use pyo3::prelude::*;

use super::{
    PyData, duration, kind_error, policy::router_tier_policies, redis_async, redis_sync, remote,
};
use crate::redis::client::NativeRedisClient;

/// The background health checks' states behind `DeploymentHealthCache`. The `_sync` methods block
/// with the GIL released, for the health check writer and the router's sync calls.
#[pyclass(frozen, name = "NativeDeploymentHealth")]
pub(crate) struct NativeDeploymentHealth(Arc<DeploymentHealth>);

#[pymethods]
impl NativeDeploymentHealth {
    #[new]
    fn new(py: Python<'_>, client: Option<PyRef<'_, NativeRedisClient>>) -> PyResult<Self> {
        Ok(Self(Arc::new(DeploymentHealth::new(
            client.map(|client| remote(&client)),
            router_tier_policies(py)?.default_in_memory_ttl,
        ))))
    }

    /// Adds the Redis tier and keeps memory; `false` when one was already attached.
    fn attach(&self, client: PyRef<'_, NativeRedisClient>) -> bool {
        self.0.attach(remote(&client))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn get<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_get_cache", kind_error, async move {
            store.get().await.map(|states| states.map(PyData))
        })
    }

    fn get_sync(&self, py: Python<'_>) -> PyResult<Option<PyData<HealthStates>>> {
        let store = self.0.clone();
        redis_sync(py, "get_cache", kind_error, async move {
            store.get().await.map(|states| states.map(PyData))
        })
    }

    fn get_shared_sync(&self, py: Python<'_>) -> PyResult<Option<PyData<HealthStates>>> {
        let store = self.0.clone();
        redis_sync(py, "get_cache", kind_error, async move {
            Ok::<_, Error>(store.get_shared().await.map(PyData))
        })
    }

    fn set_sync(
        &self,
        py: Python<'_>,
        states: PyData<HealthStates>,
        ttl_seconds: f64,
    ) -> PyResult<()> {
        let store = self.0.clone();
        let ttl = duration(ttl_seconds)?;
        redis_sync(py, "set_cache", kind_error, async move {
            store.set(states.0, ttl).await
        })
    }
}
