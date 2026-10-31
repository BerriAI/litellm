use std::time::Duration;

use litellm_redis_client::RedisClient;
use litellm_redis_store::pod_lock::{self, Acquire};
use pyo3::{IntoPyObjectExt, prelude::*};

use super::{NativeRedisClient, store_call};

#[pyclass(frozen, name = "NativePodLock")]
pub(crate) struct NativePodLock(RedisClient);

#[pymethods]
impl NativePodLock {
    #[new]
    fn new(client: &NativeRedisClient) -> Self {
        Self(client.0.clone())
    }

    #[pyo3(signature = (cronjob_id, owner, *, ttl_seconds))]
    fn acquire<'py>(
        &self,
        py: Python<'py>,
        cronjob_id: String,
        owner: String,
        ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let ttl = Duration::from_secs(ttl_seconds);
        store_call(
            py,
            async move { pod_lock::acquire(&client, &cronjob_id, &owner, ttl).await },
            |py, acquired| match acquired {
                Acquire::Acquired => ("acquired", None::<String>).into_py_any(py),
                Acquire::HeldBy(owner) => ("held_by", Some(owner)).into_py_any(py),
                Acquire::Vacant => ("vacant", None::<String>).into_py_any(py),
            },
        )
    }

    fn holder<'py>(&self, py: Python<'py>, cronjob_id: String) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        store_call(
            py,
            async move { pod_lock::holder(&client, &cronjob_id).await },
            |py, owner| owner.into_py_any(py),
        )
    }

    fn release<'py>(
        &self,
        py: Python<'py>,
        cronjob_id: String,
        owner: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        store_call(
            py,
            async move { pod_lock::release(&client, &cronjob_id, &owner).await },
            |py, released| released.into_py_any(py),
        )
    }

    fn remove<'py>(&self, py: Python<'py>, cronjob_id: String) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        store_call(
            py,
            async move { pod_lock::remove(&client, &cronjob_id).await },
            |py, removed| removed.into_py_any(py),
        )
    }
}
