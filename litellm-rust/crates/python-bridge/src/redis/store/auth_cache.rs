use std::sync::Arc;

use litellm_python_compat::Value;
use litellm_redis_store::auth_cache::{AuthCache, Settings};
use pyo3::prelude::*;

use super::{
    NativePending, PyNumber, kind_error, pending, redis_async, redis_sync, remote, request::NativeSend,
    ttl, value::PyValue,
};
use crate::redis::client::NativeRedisClient;

fn values(values: Vec<Option<Value>>) -> Vec<Option<PyValue>> {
    values.into_iter().map(|value| value.map(PyValue)).collect()
}

/// `UserApiKeyCache` with `enable_redis_auth_cache` on. The sync methods back `DualCache`'s sync
/// API and block with the GIL released, like redis-py's sync client.
#[pyclass(frozen, name = "NativeAuthCache")]
pub(crate) struct NativeAuthCache(Arc<AuthCache>);

#[pymethods]
impl NativeAuthCache {
    #[new]
    #[pyo3(signature = (client, *, memory_entries = 200, memory_ttl_seconds = None, redis_ttl_seconds = 60))]
    fn new(
        client: Option<PyRef<'_, NativeRedisClient>>,
        memory_entries: usize,
        memory_ttl_seconds: Option<f64>,
        redis_ttl_seconds: u64,
    ) -> Self {
        Self(Arc::new(AuthCache::new(
            client.map(|client| remote(&client, redis_ttl_seconds)),
            Settings {
                memory_entries,
                memory_ttl: ttl(memory_ttl_seconds),
            },
        )))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn get<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_get_cache", kind_error, async move {
            store.get(&key).await.map(|value| value.map(PyValue))
        })
    }

    fn get_sync(&self, py: Python<'_>, key: String) -> PyResult<Option<PyValue>> {
        let store = self.0.clone();
        redis_sync(py, "get_cache", kind_error, async move {
            store.get(&key).await.map(|value| value.map(PyValue))
        })
    }

    #[pyo3(signature = (key, value, ttl_seconds = None))]
    fn set<'py>(
        &self,
        py: Python<'py>,
        key: String,
        value: PyValue,
        ttl_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_set_cache", kind_error, async move {
            store.set(&key, value.0, ttl(ttl_seconds)).await
        })
    }

    #[pyo3(signature = (key, value, ttl_seconds = None))]
    fn set_sync(
        &self,
        py: Python<'_>,
        key: String,
        value: PyValue,
        ttl_seconds: Option<f64>,
    ) -> PyResult<()> {
        let store = self.0.clone();
        redis_sync(py, "set_cache", kind_error, async move {
            store.set_str(&key, value.0, ttl(ttl_seconds)).await
        })
    }

    /// Memory now and the `SET` on the batch `send` picks.
    #[pyo3(signature = (key, value, ttl_seconds, send))]
    fn declare_set(
        &self,
        key: String,
        value: PyValue,
        ttl_seconds: Option<f64>,
        send: &NativeSend,
    ) -> PyResult<NativePending> {
        let written = self
            .0
            .declare_set(send.send(), &key, value.0, ttl(ttl_seconds))
            .map_err(kind_error)?;
        Ok(pending("async_set_cache", kind_error, written, std::convert::identity))
    }

    fn declare_delete(&self, key: String, send: &NativeSend) -> PyResult<NativePending> {
        let deleted = self.0.declare_delete(send.send(), &key).map_err(kind_error)?;
        Ok(pending("async_set_cache", kind_error, deleted, std::convert::identity))
    }

    #[pyo3(signature = (entries, ttl_seconds = None))]
    fn set_many<'py>(
        &self,
        py: Python<'py>,
        entries: Vec<(String, PyValue)>,
        ttl_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let entries = entries
            .into_iter()
            .map(|(key, value)| (key, value.0))
            .collect();
        redis_async(py, "async_set_cache_pipeline", kind_error, async move {
            store.set_many(entries, ttl(ttl_seconds)).await
        })
    }

    /// On a request batch it returns before anything is sent, and `Now` is a pipeline of its own.
    fn write_back<'py>(
        &self,
        py: Python<'py>,
        entries: Vec<(String, PyValue, Option<f64>)>,
        send: &NativeSend,
    ) -> PyResult<Bound<'py, PyAny>> {
        let entries = entries
            .into_iter()
            .map(|(key, value, seconds)| (key, value.0, ttl(seconds)))
            .collect();
        let written = self.0.write_back(send.send(), entries).map_err(kind_error)?;
        redis_async(py, "async_set_cache_pipeline_with_ttls", kind_error, written)
    }

    fn delete<'py>(&self, py: Python<'py>, keys: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_delete_cache", kind_error, async move {
            store.delete(&keys).await
        })
    }

    fn delete_sync(&self, py: Python<'_>, keys: Vec<String>) -> PyResult<()> {
        let store = self.0.clone();
        redis_sync(py, "delete_cache", kind_error, async move {
            store.delete(&keys).await
        })
    }

    /// One `MGET` of keys the caller found missing in memory, on the batch `send` picks; `None`
    /// without Redis.
    fn get_many(&self, keys: Vec<String>, send: &NativeSend) -> Option<NativePending> {
        let read = self.0.get_many(send.send(), &keys)?;
        Some(pending("async_batch_get_cache", kind_error, read, values))
    }

    fn increment_if_present<'py>(
        &self,
        py: Python<'py>,
        key: String,
        amount: PyNumber,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(
            py,
            "async_increment_if_present",
            kind_error,
            async move { store.increment_if_present(&key, amount.0).await },
        )
    }

    fn get_local(&self, key: &str) -> PyResult<Option<PyValue>> {
        Ok(self.0.get_local(key).map_err(kind_error)?.map(PyValue))
    }

    #[pyo3(signature = (key, value, ttl_seconds = None))]
    fn set_local(&self, key: &str, value: PyValue, ttl_seconds: Option<f64>) -> PyResult<()> {
        self.0
            .set_local(key, value.0, ttl(ttl_seconds))
            .map_err(kind_error)
    }

    fn delete_local(&self, key: &str) -> PyResult<()> {
        self.0.delete_local(key).map_err(kind_error)
    }

    fn flush_local(&self) -> PyResult<()> {
        self.0.flush_local().map_err(kind_error)
    }

    /// `(management objects, user-key objects)` held in memory.
    fn len_local(&self) -> PyResult<(usize, usize)> {
        self.0.len_local().map_err(kind_error)
    }
}
