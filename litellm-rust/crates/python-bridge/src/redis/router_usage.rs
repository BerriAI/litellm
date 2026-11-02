use std::{sync::Arc, time::Duration};

use litellm_redis_store::router_usage::{Counter, Kind, RouterUsage};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use super::{
    PyNumber, call,
    client::{PythonRedisSettings, shared_client},
    memory_error,
    policy::router_tier_policies,
};
use crate::execution::run_async_value;

/// One router's global TPM/RPM counters. Every caller on `main` is async.
#[pyclass(frozen, name = "NativeRouterUsage")]
pub(crate) struct NativeRouterUsage(Arc<RouterUsage>);

/// `(deployment_id, model, minute, "tpm" | "rpm")`.
pub(crate) struct PyCounter(Counter);

impl<'a, 'py> FromPyObject<'a, 'py> for PyCounter {
    type Error = PyErr;

    fn extract(value: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (deployment_id, model, minute, kind): (String, Option<String>, String, String) =
            value.extract()?;
        let kind = match kind.as_str() {
            "tpm" => Kind::Tpm,
            "rpm" => Kind::Rpm,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown usage counter kind {kind:?}"
                )));
            }
        };
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
    #[pyo3(signature = (redis_kwargs, *, namespace = None))]
    fn new(
        py: Python<'_>,
        redis_kwargs: Option<PythonRedisSettings>,
        namespace: Option<String>,
    ) -> PyResult<Self> {
        let client = redis_kwargs
            .map(|kwargs| shared_client(py, kwargs, namespace))
            .transpose()?
            .flatten();
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
        let scope = call(py, "async_increment")?;
        run_async_value(py, async move {
            scope
                .scope(store.increment(&counter.0, value.0, Duration::from_secs(ttl_seconds)))
                .await
                .map(PyNumber)
                .map_err(memory_error)
        })
    }

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
        let scope = call(py, "async_increment_pipeline")?;
        run_async_value(py, async move {
            scope
                .scope(store.increment_deployment(
                    &deployment_id,
                    model.as_deref(),
                    &minute,
                    tokens.0,
                    requests,
                ))
                .await
                .map_err(memory_error)
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
        let scope = call(py, "async_batch_get_cache")?;
        run_async_value(py, async move {
            let read = scope
                .scope(store.get_many(&counters))
                .await
                .map_err(memory_error)?;
            Ok(read
                .into_iter()
                .map(|count| count.map(PyNumber))
                .collect::<Vec<_>>())
        })
    }

    fn flush_local_counters(&self) -> PyResult<()> {
        self.0.flush_local_counters().map_err(memory_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = PyDict::new(py);
        for (key, count) in self.0.snapshot().map_err(memory_error)? {
            snapshot.set_item(key, PyNumber(count))?;
        }
        Ok(snapshot)
    }
}
