use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use litellm_redis_store::router_cooldowns::{Cooldown, DeclaredCooldowns, RouterCooldowns};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyDict, PyTuple},
};

use super::{
    PyNumber, call, policy::router_tier_policies, request::NativeRequestRedis, store_error,
};
use crate::execution::{run_async_value, run_sync_value};
use crate::redis::client::NativeRedisClient;

/// One router's cooldown store. The router's sync completion path calls `put`, `active`,
/// `minimum` and `increment_allowed_fails`, so those block with the GIL released like redis-py.
#[pyclass(frozen, name = "NativeRouterCooldowns")]
pub(crate) struct NativeRouterCooldowns(Arc<RouterCooldowns>);

pub(crate) struct PyCooldown(Cooldown);

impl<'a, 'py> FromPyObject<'a, 'py> for PyCooldown {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let fields = value.cast::<PyDict>()?;
        let item = |name: &str| fields.as_any().get_item(name);
        Ok(Self(Cooldown {
            exception_received: item("exception_received")?.extract()?,
            status_code: item("status_code")?.extract()?,
            timestamp: item("timestamp")?.extract::<PyNumber>()?.0,
            cooldown_time: item("cooldown_time")?.extract::<PyNumber>()?.0,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyCooldown {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let fields = PyDict::new(py);
        fields.set_item("exception_received", self.0.exception_received)?;
        fields.set_item("status_code", self.0.status_code)?;
        fields.set_item("timestamp", PyNumber(self.0.timestamp))?;
        fields.set_item("cooldown_time", PyNumber(self.0.cooldown_time))?;
        Ok(fields)
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
                    PyCooldown(cooldown).into_pyobject(py)?.into_any(),
                ],
            )
        })
        .collect()
}

#[pymethods]
impl NativeRouterCooldowns {
    #[new]
    fn new(py: Python<'_>, client: Option<PyRef<'_, NativeRedisClient>>) -> PyResult<Self> {
        let client = client.map(|client| client.0.clone());
        let policy = router_tier_policies(py)?.cooldowns;
        Ok(Self(Arc::new(RouterCooldowns::new(client, policy))))
    }

    fn put(
        &self,
        py: Python<'_>,
        model_id: String,
        cooldown: PyCooldown,
        ttl_seconds: f64,
    ) -> PyResult<()> {
        let store = self.0.clone();
        let scope = call(py, "set_cache")?;
        run_sync_value(py, async move {
            scope
                .scope(store.put(
                    &model_id,
                    &cooldown.0,
                    Duration::from_secs_f64(ttl_seconds.max(0.0)),
                ))
                .await
                .map_err(store_error)
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
        let scope = call(py, "increment_cache")?;
        run_sync_value(py, async move {
            scope
                .scope(store.increment_allowed_fails(
                    &model_id,
                    suffix.as_deref(),
                    Duration::from_secs_f64(ttl_seconds.max(0.0)),
                ))
                .await
                .map_err(store_error)
        })
    }

    fn active<'py>(
        &self,
        py: Python<'py>,
        model_ids: Vec<String>,
    ) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        let store = self.0.clone();
        let scope = call(py, "batch_get_cache")?;
        let active = run_sync_value(py, async move {
            scope
                .scope(store.active(&model_ids))
                .await
                .map_err(store_error)
        })?;
        active_to_py(py, active)
    }

    fn aactive<'py>(&self, py: Python<'py>, model_ids: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let scope = call(py, "async_batch_get_cache")?;
        run_async_value(py, async move {
            let active = scope
                .scope(store.active(&model_ids))
                .await
                .map_err(store_error)?;
            Python::attach(|py| {
                Ok(active_to_py(py, active)?
                    .into_iter()
                    .map(Bound::unbind)
                    .collect::<Vec<_>>())
            })
        })
    }

    fn declare_active(
        &self,
        request: &NativeRequestRedis,
        model_ids: Vec<String>,
    ) -> PyResult<NativeDeclaredCooldowns> {
        let declared = self
            .0
            .declare_active(&request.0, &model_ids)
            .map_err(store_error)?;
        Ok(NativeDeclaredCooldowns {
            store: self.0.clone(),
            declared: Mutex::new(Some(declared)),
        })
    }

    fn minimum(&self, py: Python<'_>, model_ids: Vec<String>) -> PyResult<Option<PyNumber>> {
        let store = self.0.clone();
        let scope = call(py, "batch_get_cache")?;
        let minimum = run_sync_value(py, async move {
            scope
                .scope(store.minimum(&model_ids))
                .await
                .map_err(store_error)
        })?;
        Ok(minimum.map(PyNumber))
    }

    fn evict_local(&self, model_id: &str) -> PyResult<()> {
        self.0.evict_local(model_id).map_err(store_error)
    }

    fn flush_local_counters(&self) -> PyResult<()> {
        self.0.flush_local_counters().map_err(store_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = PyDict::new(py);
        for (key, cooldown) in self.0.snapshot().map_err(store_error)? {
            snapshot.set_item(key, PyCooldown(cooldown))?;
        }
        Ok(snapshot)
    }
}

/// An `active` read declared on a request's pre-call batch. `resolve` answers it once.
#[pyclass(frozen, name = "NativeDeclaredCooldowns")]
pub(crate) struct NativeDeclaredCooldowns {
    store: Arc<RouterCooldowns>,
    declared: Mutex<Option<DeclaredCooldowns>>,
}

#[pymethods]
impl NativeDeclaredCooldowns {
    fn resolve<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let declared = self
            .declared
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("this cooldown read was already resolved"))?;
        let store = self.store.clone();
        let scope = call(py, "async_batch_get_cache")?;
        run_async_value(py, async move {
            let active = scope
                .scope(store.resolve_active(declared))
                .await
                .map_err(store_error)?;
            Python::attach(|py| {
                Ok(active_to_py(py, active)?
                    .into_iter()
                    .map(Bound::unbind)
                    .collect::<Vec<_>>())
            })
        })
    }
}
