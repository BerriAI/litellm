use std::{sync::Arc, time::Duration};

use litellm_python_compat::Value;
use litellm_redis_store::auth_cache::{AuthCache, DeclaredWrite, Error, Settings};
use pyo3::prelude::*;

use super::{
    NativePending, PyNumber, kind_error, redis_async, redis_sync, request::NativeRequestRedis,
    store_error, ttl, value::PyValue,
};
use crate::redis::client::NativeRedisClient;

fn auth_cache_error(error: Error) -> PyErr {
    kind_error(error.kind(), &error)
}

/// A `set_pre_call` or `delete_pre_call`, settled once and raising when Redis refused it.
fn declared_write(declared: DeclaredWrite) -> NativePending {
    NativePending::new("async_set_cache", auth_cache_error, async move {
        declared.resolve().await
    })
}

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
            client.map(|client| client.0.clone()),
            Settings {
                memory_entries,
                memory_ttl: ttl(memory_ttl_seconds),
                redis_ttl: Duration::from_secs(redis_ttl_seconds),
            },
        )))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn get<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_get_cache", store_error, async move {
            store.get(&key).await.map(|value| value.map(PyValue))
        })
    }

    fn get_sync(&self, py: Python<'_>, key: String) -> PyResult<Option<PyValue>> {
        let store = self.0.clone();
        redis_sync(py, "get_cache", store_error, async move {
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
        redis_async(py, "async_set_cache", store_error, async move {
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
        redis_sync(py, "set_cache", store_error, async move {
            store.set_str(&key, value.0, ttl(ttl_seconds)).await
        })
    }

    /// Memory now and the `SET` on the request's pre-call batch; `None` without Redis.
    #[pyo3(signature = (request, key, value, ttl_seconds = None))]
    fn set_pre_call(
        &self,
        request: &NativeRequestRedis,
        key: String,
        value: PyValue,
        ttl_seconds: Option<f64>,
    ) -> PyResult<Option<NativePending>> {
        Ok(self
            .0
            .set_pre_call(&request.0, &key, value.0, ttl(ttl_seconds))
            .map_err(store_error)?
            .map(declared_write))
    }

    fn delete_pre_call(
        &self,
        request: &NativeRequestRedis,
        key: String,
    ) -> PyResult<Option<NativePending>> {
        Ok(self
            .0
            .delete_pre_call(&request.0, &key)
            .map_err(store_error)?
            .map(declared_write))
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
        redis_async(py, "async_set_cache_pipeline", store_error, async move {
            store.set_many(entries, ttl(ttl_seconds)).await
        })
    }

    /// On the request's pre-call batch when `request` is given, returning before anything is sent.
    fn write_back<'py>(
        &self,
        py: Python<'py>,
        request: Option<&NativeRequestRedis>,
        entries: Vec<(String, PyValue, Option<f64>)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let request = request.map(|request| request.0.clone());
        let entries = entries
            .into_iter()
            .map(|(key, value, seconds)| (key, value.0, ttl(seconds)))
            .collect();
        redis_async(
            py,
            "async_set_cache_pipeline_with_ttls",
            auth_cache_error,
            async move { store.write_back(request.as_ref(), entries).await },
        )
    }

    fn delete<'py>(&self, py: Python<'py>, keys: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_delete_cache", auth_cache_error, async move {
            store.delete(&keys).await
        })
    }

    fn delete_sync(&self, py: Python<'_>, keys: Vec<String>) -> PyResult<()> {
        let store = self.0.clone();
        redis_sync(py, "delete_cache", auth_cache_error, async move {
            store.delete(&keys).await
        })
    }

    /// Reads of keys the caller found missing in memory, on the request's pre-call batch; `None`
    /// without Redis.
    fn declare_get_many(
        &self,
        request: &NativeRequestRedis,
        keys: Vec<String>,
    ) -> Option<NativePending> {
        self.0.declare_get_many(&request.0, &keys).map(|declared| {
            NativePending::new("async_batch_get_cache", auth_cache_error, async move {
                declared.resolve().await.map(values)
            })
        })
    }

    /// One round trip for keys missing in memory; resolves to `None` without Redis.
    fn get_many<'py>(&self, py: Python<'py>, keys: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_batch_get_cache", auth_cache_error, async move {
            store
                .get_many(&keys)
                .await
                .transpose()
                .map(|read| read.map(values))
        })
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
            auth_cache_error,
            async move { store.increment_if_present(&key, amount.0).await },
        )
    }

    fn get_local(&self, key: &str) -> PyResult<Option<PyValue>> {
        Ok(self.0.get_local(key).map_err(store_error)?.map(PyValue))
    }

    #[pyo3(signature = (key, value, ttl_seconds = None))]
    fn set_local(&self, key: &str, value: PyValue, ttl_seconds: Option<f64>) -> PyResult<()> {
        self.0
            .set_local(key, value.0, ttl(ttl_seconds))
            .map_err(store_error)
    }

    fn delete_local(&self, key: &str) -> PyResult<()> {
        self.0.delete_local(key).map_err(store_error)
    }

    fn flush_local(&self) -> PyResult<()> {
        self.0.flush_local().map_err(store_error)
    }

    /// `(management objects, user-key objects)` held in memory.
    fn len_local(&self) -> PyResult<(usize, usize)> {
        self.0.len_local().map_err(store_error)
    }
}
