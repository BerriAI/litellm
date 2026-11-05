use std::{sync::Arc, time::Duration};

use litellm_redis_store::router_usage::{Counter, RouterUsage};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use super::{PyNumber, policy::router_tier_policies, redis_async, store_error};
use crate::redis::client::NativeRedisClient;

/// One router's global TPM/RPM counters. Every caller on `main` is async.
#[pyclass(frozen, name = "NativeRouterUsage")]
pub(crate) struct NativeRouterUsage(Arc<RouterUsage>);

/// `(deployment_id, model, minute, kind)`, the kind spelled as its key segment.
pub(crate) struct PyCounter(Counter);

impl<'a, 'py> FromPyObject<'a, 'py> for PyCounter {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (deployment_id, model, minute, kind): (String, Option<String>, String, String) =
            value.extract()?;
        let kind = kind
            .parse()
            .map_err(|_| PyValueError::new_err(format!("unknown usage counter kind {kind:?}")))?;
        Ok(Self(Counter {
            deployment_id,
            model,
            minute,
            kind,
        }))
    }
}

#[pymethods]
impl NativeRouterUsage {
    #[new]
    fn new(py: Python<'_>, client: Option<PyRef<'_, NativeRedisClient>>) -> PyResult<Self> {
        let client = client.map(|client| client.0.clone());
        let policy = router_tier_policies(py)?.usage;
        Ok(Self(Arc::new(RouterUsage::new(client, policy))))
    }

    fn increment<'py>(
        &self,
        py: Python<'py>,
        counter: PyCounter,
        value: PyNumber,
        ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_increment", store_error, async move {
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
        redis_async(py, "async_increment_pipeline", store_error, async move {
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
        let counters = counters
            .into_iter()
            .map(|counter| counter.0)
            .collect::<Vec<_>>();
        redis_async(py, "async_batch_get_cache", store_error, async move {
            let read = store.get_many(&counters).await?;
            Ok(read
                .into_iter()
                .map(|count| count.map(PyNumber))
                .collect::<Vec<_>>())
        })
    }

    fn flush_local_counters(&self) -> PyResult<()> {
        self.0.flush_local_counters().map_err(store_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = PyDict::new(py);
        for (key, count) in self.0.snapshot().map_err(store_error)? {
            snapshot.set_item(key, PyNumber(count))?;
        }
        Ok(snapshot)
    }
}
