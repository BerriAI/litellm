use std::sync::Arc;

use litellm_redis_store::{
    Send,
    router_cooldowns::{Cooldown, RouterCooldowns},
};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use super::{
    NativePending, PyData, PyNumber, pending, policy::router_tier_policies, redis_sync, remote,
    request::NativeSend, runtime_error, ttl,
};
use crate::redis::client::NativeRedisClient;

/// One router's cooldown store. The router's sync completion path calls `put`, `active`,
/// `minimum` and `increment_allowed_fails`, so those block with the GIL released like redis-py.
#[pyclass(frozen, name = "NativeRouterCooldowns")]
pub(crate) struct NativeRouterCooldowns(Arc<RouterCooldowns>);

/// `[(model_id, cooldown), ...]` the way `RedisCache` hands the router its batch read.
struct PyActive(Vec<(String, Cooldown)>);

impl<'py> IntoPyObject<'py> for PyActive {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        active_to_py(py, self.0)?
            .into_pyobject(py)
            .map(Bound::into_any)
    }
}

fn active_to_py(
    py: Python<'_>,
    active: Vec<(String, Cooldown)>,
) -> PyResult<Vec<Bound<'_, PyTuple>>> {
    active
        .into_iter()
        .map(|(model_id, cooldown)| {
            PyTuple::new(
                py,
                [
                    model_id.into_pyobject(py)?.into_any(),
                    PyData(cooldown).into_pyobject(py)?,
                ],
            )
        })
        .collect()
}

#[pymethods]
impl NativeRouterCooldowns {
    #[new]
    #[pyo3(signature = (client, redis_ttl_seconds = 60))]
    fn new(py: Python<'_>, client: Option<PyRef<'_, NativeRedisClient>>, redis_ttl_seconds: u64) -> PyResult<Self> {
        let remote = client.map(|client| remote(&client, redis_ttl_seconds));
        let policy = router_tier_policies(py)?.cooldowns;
        Ok(Self(Arc::new(RouterCooldowns::new(remote, policy))))
    }

    /// Adds the Redis tier and keeps memory; `false` when one was already attached.
    #[pyo3(signature = (client, redis_ttl_seconds = 60))]
    fn attach(&self, client: PyRef<'_, NativeRedisClient>, redis_ttl_seconds: u64) -> bool {
        self.0.attach(remote(&client, redis_ttl_seconds))
    }

    fn put(
        &self,
        py: Python<'_>,
        model_id: String,
        cooldown: PyData<Cooldown>,
        ttl_seconds: f64,
    ) -> PyResult<()> {
        let store = self.0.clone();
        let ttl = ttl(Some(ttl_seconds)).unwrap_or_default();
        redis_sync(py, "set_cache", runtime_error, async move {
            store.put(&model_id, &cooldown.0, ttl).await
        })
    }

    fn increment_allowed_fails(
        &self,
        py: Python<'_>,
        model_id: String,
        suffix: Option<String>,
        ttl_seconds: f64,
    ) -> PyResult<i64> {
        let store = self.0.clone();
        let ttl = ttl(Some(ttl_seconds)).unwrap_or_default();
        redis_sync(py, "increment_cache", runtime_error, async move {
            store
                .increment_allowed_fails(&model_id, suffix.as_deref(), ttl)
                .await
        })
    }

    /// The deployments still cooling down, read on the batch `send` picks.
    fn active(&self, model_ids: Vec<String>, send: &NativeSend) -> PyResult<NativePending> {
        let active = self.0.active(send.send(), &model_ids).map_err(runtime_error)?;
        Ok(pending("async_batch_get_cache", runtime_error, active, PyActive))
    }

    /// `active` on its own with the GIL released, for the router's sync completion path.
    fn active_sync<'py>(
        &self,
        py: Python<'py>,
        model_ids: Vec<String>,
    ) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        let active = self.0.active(Send::Now, &model_ids).map_err(runtime_error)?;
        let active = redis_sync(py, "batch_get_cache", runtime_error, active)?;
        active_to_py(py, active)
    }

    fn minimum(&self, py: Python<'_>, model_ids: Vec<String>) -> PyResult<Option<PyNumber>> {
        let store = self.0.clone();
        redis_sync(py, "batch_get_cache", runtime_error, async move {
            store
                .minimum(&model_ids)
                .await
                .map(|minimum| minimum.map(PyNumber))
        })
    }

    fn delete_local(&self, model_id: &str) -> PyResult<()> {
        self.0.delete_local(model_id).map_err(runtime_error)
    }

    fn flush_local_counters(&self) -> PyResult<()> {
        self.0.flush_local_counters().map_err(runtime_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = PyDict::new(py);
        for (key, cooldown) in self.0.snapshot().map_err(runtime_error)? {
            snapshot.set_item(key, PyData(cooldown))?;
        }
        Ok(snapshot)
    }
}
