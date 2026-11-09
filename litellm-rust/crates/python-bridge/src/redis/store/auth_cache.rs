use std::sync::Arc;

use futures_util::{StreamExt, stream::BoxStream};
use litellm_redis_client::SubscriptionEvent;
use tokio::sync::Mutex;

use litellm_python_compat::Value;
use litellm_redis_store::auth_cache::{AuthCache, Settings};
use pyo3::prelude::*;

use super::{
    NativePending, PyNumber, kind_error, pending, redis_async, redis_sync, remote,
    request::{NativeRequestRedis, NativeSend},
    ttl,
    value::PyValue,
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
    #[pyo3(signature = (client, *, memory_entries, memory_ttl_seconds = None))]
    fn new(
        client: Option<PyRef<'_, NativeRedisClient>>,
        memory_entries: usize,
        memory_ttl_seconds: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self(Arc::new(AuthCache::new(
            client.map(|client| remote(&client)),
            Settings {
                memory_entries,
                memory_ttl: ttl(memory_ttl_seconds)?,
            },
        ))))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    fn attach(&self, client: &NativeRedisClient) -> bool {
        self.0.attach(remote(client))
    }

    fn current_generation(&self) -> u64 {
        self.0.current_generation()
    }

    #[pyo3(signature = (key, value, generation, ttl_seconds = None))]
    fn set_if_generation(
        &self,
        key: &str,
        value: PyValue,
        generation: u64,
        ttl_seconds: Option<f64>,
    ) -> PyResult<bool> {
        Ok(self
            .0
            .set_if_generation(key, value.0, generation, ttl(ttl_seconds)?))
    }

    fn subscribe(&self, client: &NativeRedisClient, channel: &str) -> NativeAuthSubscription {
        self.0.subscriber_connected(false);
        NativeAuthSubscription {
            cache: self.0.clone(),
            stream: Arc::new(Mutex::new(Some(client.client.subscribe_events(channel)))),
        }
    }

    /// `request` answers a key its pre-call reads already found absent without another `GET`.
    #[pyo3(signature = (key, request = None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: String,
        request: Option<&NativeRequestRedis>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let request = request.map(|request| request.0.clone());
        redis_async(py, "async_get_cache", kind_error, async move {
            store
                .get(request.as_ref(), &key)
                .await
                .map(|value| value.map(PyValue))
        })
    }

    fn get_sync(&self, py: Python<'_>, key: String) -> PyResult<Option<PyValue>> {
        let store = self.0.clone();
        redis_sync(py, "get_cache", kind_error, async move {
            store.get(None, &key).await.map(|value| value.map(PyValue))
        })
    }

    /// `update_cache_ttl` and `update_in_memory_max_size`: both partitions keep what they hold.
    #[pyo3(signature = (*, memory_entries, memory_ttl_seconds = None))]
    fn reconfigure(&self, memory_entries: usize, memory_ttl_seconds: Option<f64>) -> PyResult<()> {
        self.0.reconfigure(Settings {
            memory_entries,
            memory_ttl: ttl(memory_ttl_seconds)?,
        });
        Ok(())
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
        let ttl = ttl(ttl_seconds)?;
        redis_async(py, "async_set_cache", kind_error, async move {
            store.set(&key, value.0, ttl).await
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
        let ttl = ttl(ttl_seconds)?;
        redis_sync(py, "set_cache", kind_error, async move {
            store.set_str(&key, value.0, ttl).await
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
            .declare_set(send.send(), &key, value.0, ttl(ttl_seconds)?)
            .map_err(kind_error)?;
        Ok(pending(
            "async_set_cache",
            kind_error,
            written,
            std::convert::identity,
        ))
    }

    fn declare_delete(&self, key: String, send: &NativeSend) -> PyResult<NativePending> {
        let deleted = self
            .0
            .declare_delete(send.send(), &key)
            .map_err(kind_error)?;
        Ok(pending(
            "async_delete_cache",
            kind_error,
            deleted,
            std::convert::identity,
        ))
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
        let ttl = ttl(ttl_seconds)?;
        redis_async(py, "async_set_cache_pipeline", kind_error, async move {
            store.set_many(entries, ttl).await
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
            .map(|(key, value, seconds)| Ok((key, value.0, ttl(seconds)?)))
            .collect::<PyResult<_>>()?;
        let written = self
            .0
            .write_back(send.send(), entries)
            .map_err(kind_error)?;
        redis_async(
            py,
            "async_set_cache_pipeline_with_ttls",
            kind_error,
            written,
        )
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
        redis_async(py, "async_increment_if_present", kind_error, async move {
            store.increment_if_present(&key, amount.0).await
        })
    }

    fn get_local(&self, key: &str) -> PyResult<Option<PyValue>> {
        Ok(self.0.get_local(key).map_err(kind_error)?.map(PyValue))
    }

    #[pyo3(signature = (key, value, ttl_seconds = None))]
    fn set_local(&self, key: &str, value: PyValue, ttl_seconds: Option<f64>) -> PyResult<()> {
        self.0
            .set_local(key, value.0, ttl(ttl_seconds)?)
            .map_err(kind_error)
    }

    fn delete_local(&self, key: &str) -> PyResult<()> {
        self.0.delete_local(key).map_err(kind_error)
    }

    fn flush_local(&self) -> PyResult<()> {
        self.0.flush_local().map_err(kind_error)
    }

    fn len_local(&self) -> PyResult<PyLocalSizes> {
        let (objects, key_objects) = self.0.len_local().map_err(kind_error)?;
        Ok(PyLocalSizes {
            objects,
            key_objects,
        })
    }
}

#[pyclass(frozen)]
pub(crate) struct NativeAuthSubscription {
    cache: Arc<AuthCache>,
    stream: Arc<Mutex<Option<BoxStream<'static, SubscriptionEvent>>>>,
}

#[pymethods]
impl NativeAuthSubscription {
    fn is_connected(&self) -> bool {
        self.cache.local_ready()
    }

    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cache = self.cache.clone();
        let stream = self.stream.clone();
        litellm_host_python::run_async_value(py, async move {
            let mut guard = stream.lock().await;
            let Some(stream) = guard.as_mut() else {
                return Ok(None::<Py<pyo3::types::PyBytes>>);
            };
            match stream.next().await {
                Some(SubscriptionEvent::Connected) => cache.subscriber_connected(true),
                Some(SubscriptionEvent::Message(payload)) => {
                    if let Ok(message) = serde_json::from_slice::<serde_json::Value>(&payload)
                        && let Some(key) =
                            message.get("cache_key").and_then(serde_json::Value::as_str)
                        && message
                            .get("new_value")
                            .and_then(serde_json::Value::as_f64)
                            .is_none()
                    {
                        cache.delete_local(key).map_err(kind_error)?;
                    }
                    return Ok(Some(Python::attach(|py| {
                        pyo3::types::PyBytes::new(py, &payload).unbind()
                    })));
                }
                _ => cache.subscriber_connected(false),
            }
            Ok(None)
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.cache.subscriber_connected(false);
        let stream = self.stream.clone();
        litellm_host_python::run_async_value(py, async move {
            stream.lock().await.take();
            Ok(())
        })
    }
}

impl Drop for NativeAuthSubscription {
    fn drop(&mut self) {
        self.cache.subscriber_connected(false);
    }
}

/// Entries the management-object and the user-key partitions hold in memory.
#[pyclass(frozen, eq, get_all, name = "LocalSizes")]
#[derive(PartialEq)]
pub(crate) struct PyLocalSizes {
    objects: usize,
    key_objects: usize,
}

#[pymethods]
impl PyLocalSizes {
    #[new]
    fn new(objects: usize, key_objects: usize) -> Self {
        Self {
            objects,
            key_objects,
        }
    }
}
