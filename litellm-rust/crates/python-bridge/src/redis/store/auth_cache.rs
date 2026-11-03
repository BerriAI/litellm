use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use litellm_python_compat::Value;
use litellm_redis_store::auth_cache::{AuthCache, DeclaredGets, DeclaredWrite, Error, Settings};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::{PyNumber, call, request::NativeRequestRedis, store_error, value::PyValue};
use crate::execution::{run_async_value, run_sync_value};
use crate::redis::client::NativeRedisClient;

pyo3::create_exception!(
    _native,
    NativeAuthCacheError,
    pyo3::exceptions::PyException,
    "An auth cache operation failed. Args are (kind, message), kind one of \"breaker_open\", \"timeout\", \"transport\", \"rejected\", \"decode\", \"encode\" or \"memory\"."
);

fn auth_cache_error(error: Error) -> PyErr {
    use litellm_redis_client::Error as RedisError;
    let kind = match &error {
        Error::Redis(RedisError::BreakerOpen) => "breaker_open",
        Error::Redis(RedisError::Transport(error)) if error.is_timeout() => "timeout",
        Error::Redis(RedisError::Transport(_)) => "transport",
        Error::Redis(RedisError::Server(_)) => "rejected",
        Error::Redis(RedisError::Decode(_)) | Error::Decode(_) => "decode",
        Error::Encode(_) => "encode",
        Error::Memory(_) => "memory",
    };
    NativeAuthCacheError::new_err((kind, litellm_error::Report::new(&error).to_string()))
}

fn ttl(seconds: Option<f64>) -> Option<Duration> {
    seconds.map(|seconds| Duration::from_secs_f64(seconds.max(0.0)))
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
        let scope = call(py, "async_get_cache")?;
        run_async_value(py, async move {
            let value = scope.scope(store.get(&key)).await.map_err(store_error)?;
            Ok(value.map(PyValue))
        })
    }

    fn get_sync(&self, py: Python<'_>, key: String) -> PyResult<Option<PyValue>> {
        let store = self.0.clone();
        let scope = call(py, "get_cache")?;
        run_sync_value(py, async move {
            let value = scope.scope(store.get(&key)).await.map_err(store_error)?;
            Ok(value.map(PyValue))
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
        let scope = call(py, "async_set_cache")?;
        run_async_value(py, async move {
            scope
                .scope(store.set(&key, value.0, ttl(ttl_seconds)))
                .await
                .map_err(store_error)
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
        let scope = call(py, "set_cache")?;
        run_sync_value(py, async move {
            scope
                .scope(store.set_str(&key, value.0, ttl(ttl_seconds)))
                .await
                .map_err(store_error)
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
    ) -> PyResult<Option<NativeDeclaredAuthWrite>> {
        Ok(self
            .0
            .set_pre_call(&request.0, &key, value.0, ttl(ttl_seconds))
            .map_err(store_error)?
            .map(NativeDeclaredAuthWrite::new))
    }

    fn delete_pre_call(
        &self,
        request: &NativeRequestRedis,
        key: String,
    ) -> PyResult<Option<NativeDeclaredAuthWrite>> {
        Ok(self
            .0
            .delete_pre_call(&request.0, &key)
            .map_err(store_error)?
            .map(NativeDeclaredAuthWrite::new))
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
        let scope = call(py, "async_set_cache_pipeline")?;
        run_async_value(py, async move {
            scope
                .scope(store.set_many(entries, ttl(ttl_seconds)))
                .await
                .map_err(store_error)
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
        let scope = call(py, "async_set_cache_pipeline_with_ttls")?;
        run_async_value(py, async move {
            scope
                .scope(store.write_back(request.as_ref(), entries))
                .await
                .map_err(auth_cache_error)
        })
    }

    fn delete<'py>(&self, py: Python<'py>, keys: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let scope = call(py, "async_delete_cache")?;
        run_async_value(py, async move {
            scope
                .scope(store.delete(&keys))
                .await
                .map_err(auth_cache_error)
        })
    }

    fn delete_sync(&self, py: Python<'_>, keys: Vec<String>) -> PyResult<()> {
        let store = self.0.clone();
        let scope = call(py, "delete_cache")?;
        run_sync_value(py, async move {
            scope
                .scope(store.delete(&keys))
                .await
                .map_err(auth_cache_error)
        })
    }

    /// Reads of keys the caller found missing in memory, on the request's pre-call batch; `None`
    /// without Redis.
    fn declare_get_many(
        &self,
        request: &NativeRequestRedis,
        keys: Vec<String>,
    ) -> Option<NativeDeclaredAuthReads> {
        self.0
            .declare_get_many(&request.0, &keys)
            .map(|declared| NativeDeclaredAuthReads(Mutex::new(Some(declared))))
    }

    /// One round trip for keys missing in memory; resolves to `None` without Redis.
    fn get_many<'py>(&self, py: Python<'py>, keys: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let scope = call(py, "async_batch_get_cache")?;
        run_async_value(py, async move {
            scope
                .scope(store.get_many(&keys))
                .await
                .transpose()
                .map(|read| read.map(values))
                .map_err(auth_cache_error)
        })
    }

    fn increment_if_present<'py>(
        &self,
        py: Python<'py>,
        key: String,
        amount: PyNumber,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let scope = call(py, "async_increment_if_present")?;
        run_async_value(py, async move {
            scope
                .scope(store.increment_if_present(&key, amount.0))
                .await
                .map_err(auth_cache_error)
        })
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

/// A `declare_get_many` on a request's pre-call batch. `resolve` answers it once.
#[pyclass(frozen, name = "NativeDeclaredAuthReads")]
pub(crate) struct NativeDeclaredAuthReads(Mutex<Option<DeclaredGets>>);

#[pymethods]
impl NativeDeclaredAuthReads {
    fn resolve<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let declared = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("this auth cache read was already resolved"))?;
        let scope = call(py, "async_batch_get_cache")?;
        run_async_value(py, async move {
            scope
                .scope(declared.resolve())
                .await
                .map(values)
                .map_err(auth_cache_error)
        })
    }
}

/// A `set_pre_call` or `delete_pre_call`. `resolve` settles it once, raising when Redis refused it.
#[pyclass(frozen, name = "NativeDeclaredAuthWrite")]
pub(crate) struct NativeDeclaredAuthWrite(Mutex<Option<DeclaredWrite>>);

impl NativeDeclaredAuthWrite {
    fn new(declared: DeclaredWrite) -> Self {
        Self(Mutex::new(Some(declared)))
    }
}

#[pymethods]
impl NativeDeclaredAuthWrite {
    fn resolve<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let declared = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("this auth cache write was already resolved"))?;
        let scope = call(py, "async_set_cache")?;
        run_async_value(py, async move {
            scope
                .scope(declared.resolve())
                .await
                .map_err(auth_cache_error)
        })
    }
}
