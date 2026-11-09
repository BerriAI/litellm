use std::{collections::HashMap, sync::Arc};

use litellm_redis_store::{
    Send,
    router_cooldowns::{Cooldown, RouterCooldowns},
};
use pyo3::{prelude::*, types::PyTuple};

use super::{
    NativePending, PyData, PyNumber, duration, kind_error, pending, policy::router_tier_policies,
    redis_async, redis_sync, remote, request::NativeSend,
};
use crate::redis::client::NativeRedisClient;

/// One router's cooldown store. The router's sync completion path calls `put`, `active_sync`,
/// `minimum` and `increment_allowed_fails`, so those block with the GIL released like redis-py; the
/// `async_` twins return awaitables for async callers.
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
    fn new(py: Python<'_>, client: Option<PyRef<'_, NativeRedisClient>>) -> PyResult<Self> {
        let remote = client.map(|client| remote(&client));
        let policy = router_tier_policies(py)?.cooldowns;
        Ok(Self(Arc::new(RouterCooldowns::new(remote, policy))))
    }

    /// Adds the Redis tier and keeps memory; `false` when one was already attached.
    fn attach(&self, client: PyRef<'_, NativeRedisClient>) -> bool {
        self.0.attach(remote(&client))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn put(
        &self,
        py: Python<'_>,
        model_id: String,
        cooldown: PyData<Cooldown>,
        ttl_seconds: f64,
    ) -> PyResult<()> {
        let store = self.0.clone();
        let ttl = duration(ttl_seconds)?;
        redis_sync(py, "set_cache", kind_error, async move {
            store.put_str(&model_id, &cooldown.0, ttl).await
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
        let ttl = duration(ttl_seconds)?;
        redis_sync(py, "increment_cache", kind_error, async move {
            store
                .increment_allowed_fails_int(&model_id, suffix.as_deref(), ttl)
                .await
        })
    }

    fn async_put<'py>(
        &self,
        py: Python<'py>,
        model_id: String,
        cooldown: PyData<Cooldown>,
        ttl_seconds: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let ttl = duration(ttl_seconds)?;
        redis_async(py, "async_set_cache", kind_error, async move {
            store.put(&model_id, &cooldown.0, ttl).await
        })
    }

    fn async_increment_allowed_fails<'py>(
        &self,
        py: Python<'py>,
        model_id: String,
        suffix: Option<String>,
        ttl_seconds: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let ttl = duration(ttl_seconds)?;
        redis_async(py, "async_increment", kind_error, async move {
            store
                .increment_allowed_fails(&model_id, suffix.as_deref(), ttl)
                .await
        })
    }

    /// The deployments still cooling down, read on the batch `send` picks.
    fn active(&self, model_ids: Vec<String>, send: &NativeSend) -> PyResult<NativePending> {
        let active = self.0.active(send.send(), &model_ids).map_err(kind_error)?;
        Ok(pending(
            "async_batch_get_cache",
            kind_error,
            active,
            PyActive,
        ))
    }

    /// `active` on its own with the GIL released, for the router's sync completion path.
    fn active_sync<'py>(
        &self,
        py: Python<'py>,
        model_ids: Vec<String>,
    ) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        let active = self.0.active(Send::Now, &model_ids).map_err(kind_error)?;
        let active = redis_sync(py, "batch_get_cache", kind_error, active)?;
        active_to_py(py, active)
    }

    fn minimum(&self, py: Python<'_>, model_ids: Vec<String>) -> PyResult<Option<PyNumber>> {
        let store = self.0.clone();
        redis_sync(py, "batch_get_cache", kind_error, async move {
            store
                .minimum(&model_ids)
                .await
                .map(|minimum| minimum.map(PyNumber))
        })
    }

    fn delete_local(&self, model_id: &str) -> PyResult<()> {
        self.0.delete_local(model_id).map_err(kind_error)
    }

    fn flush_local_counters(&self) -> PyResult<()> {
        self.0.flush_local_counters().map_err(kind_error)
    }

    fn snapshot(&self) -> PyResult<HashMap<String, PyData<Cooldown>>> {
        Ok(self
            .0
            .snapshot()
            .map_err(kind_error)?
            .into_iter()
            .map(|(key, cooldown)| (key, PyData(cooldown)))
            .collect())
    }
}
