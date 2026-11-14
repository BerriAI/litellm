use std::sync::Arc;

use litellm_redis_store::sensitive_route_pins::SensitiveRoutePins;
use pyo3::prelude::*;

use super::{duration, kind_error, redis_async, remote};
use crate::redis::client::NativeRedisClient;

#[pyclass(frozen, name = "NativeSensitiveRoutePins")]
pub(crate) struct NativeSensitiveRoutePins(Arc<SensitiveRoutePins>);

#[pymethods]
impl NativeSensitiveRoutePins {
    #[new]
    fn new(client: Option<PyRef<'_, NativeRedisClient>>) -> Self {
        Self(Arc::new(SensitiveRoutePins::new(
            client.map(|client| remote(&client)),
        )))
    }

    fn get<'py>(
        &self,
        py: Python<'py>,
        tenant: String,
        session_id: String,
        default_ttl_seconds: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let default_ttl = duration(default_ttl_seconds)?;
        redis_async(py, "async_get_cache", kind_error, async move {
            store.get(&tenant, &session_id, default_ttl).await
        })
    }

    fn set<'py>(
        &self,
        py: Python<'py>,
        tenant: String,
        session_id: String,
        model: String,
        ttl_seconds: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let ttl = duration(ttl_seconds)?;
        redis_async(py, "async_set_cache", kind_error, async move {
            store.set(&tenant, &session_id, &model, ttl).await
        })
    }
}
