use std::sync::Arc;

use futures_util::StreamExt;
use litellm_host_python::{PythonContext, attach_blocking};
use litellm_redis_store::{
    auth_cache::AuthCache,
    auth_cache_invalidation::{self, AuthCacheInvalidation, Invalidation},
    spend_counters::SpendCounters,
};
use parking_lot::Mutex;
use pyo3::prelude::*;
use tokio::task::AbortHandle;

use super::{PyNumber, auth_cache::NativeAuthCache, spend_counters::NativeSpendCounters};
use crate::{execution::run_async_value, redis::client::NativeRedisClient};

const HOST_MODULE: &str = "litellm.rust_bridge.host.auth_cache_invalidation";

/// The auth cache invalidation channel: publishes in the background, and while started applies
/// every message to the native auth cache and spend counters, then hands it to the host for the
/// caches still in Python (`python_caches`, held opaquely). The subscription ends at `stop` or when
/// this is dropped.
#[pyclass(frozen, name = "NativeAuthCacheInvalidation")]
pub(crate) struct NativeAuthCacheInvalidation {
    channel: AuthCacheInvalidation,
    auth_cache: Arc<AuthCache>,
    spend_counters: Arc<SpendCounters>,
    apply_python: Arc<Py<PyAny>>,
    python_caches: Arc<Py<PyAny>>,
    context: PythonContext,
    subscription: Mutex<Option<AbortHandle>>,
}

#[pymethods]
impl NativeAuthCacheInvalidation {
    /// `namespace` is `RedisCache.namespace`, the channel's prefix.
    #[new]
    fn new(
        py: Python<'_>,
        client: PyRef<'_, NativeRedisClient>,
        namespace: Option<&str>,
        auth_cache: PyRef<'_, NativeAuthCache>,
        spend_counters: PyRef<'_, NativeSpendCounters>,
        python_caches: Py<PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            channel: AuthCacheInvalidation::new(
                client.client.clone(),
                namespace,
                litellm_host_python::runtime_handle()?,
            ),
            auth_cache: auth_cache.0.clone(),
            spend_counters: spend_counters.0.clone(),
            apply_python: Arc::new(py.import(HOST_MODULE)?.getattr("apply")?.unbind()),
            python_caches: Arc::new(python_caches),
            context: PythonContext::capture(py)?,
            subscription: Mutex::new(None),
        })
    }

    #[getter]
    fn channel(&self) -> &str {
        self.channel.channel()
    }

    /// `False` when the publish was dropped because too many are already waiting on Redis.
    #[pyo3(signature = (cache_key, new_value = None, ttl = None))]
    fn publish(
        &self,
        cache_key: String,
        new_value: Option<PyNumber>,
        ttl: Option<PyNumber>,
    ) -> bool {
        self.channel.publish(&match new_value {
            Some(value) => Invalidation::Set {
                cache_key,
                value: value.0,
                ttl: ttl.map(|ttl| ttl.0),
            },
            None => Invalidation::Evict(cache_key),
        })
    }

    /// Resolves once the pipeline was answered or after two seconds, whichever is first.
    fn broadcast_evictions<'py>(
        &self,
        py: Python<'py>,
        cache_keys: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let channel = self.channel.clone();
        run_async_value(py, async move {
            channel.broadcast_evictions(&cache_keys).await;
            Ok(())
        })
    }

    /// Subscribes once; a second call keeps the running subscription.
    fn start(&self) {
        let mut subscription = self.subscription.lock();
        if subscription.is_some() {
            return;
        }
        let mut messages = self.channel.subscribe();
        let auth_cache = self.auth_cache.clone();
        let spend_counters = self.spend_counters.clone();
        let apply_python = self.apply_python.clone();
        let python_caches = self.python_caches.clone();
        let context = self.context.clone();
        let task = self.channel.runtime().spawn(async move {
            while let Some(invalidation) = messages.next().await {
                auth_cache_invalidation::apply(&invalidation, &auth_cache, &spend_counters);
                let apply_python = apply_python.clone();
                let python_caches = python_caches.clone();
                let applied = attach_blocking(context.clone(), move |py| {
                    let (cache_key, new_value, ttl) = match invalidation {
                        Invalidation::Evict(cache_key) => (cache_key, None, None),
                        Invalidation::Set {
                            cache_key,
                            value,
                            ttl,
                        } => (cache_key, Some(value.as_f64()), ttl.map(|ttl| ttl.as_f64())),
                    };
                    apply_python
                        .bind(py)
                        .call1((python_caches.bind(py), cache_key, new_value, ttl))
                        .map(drop)
                })
                .await
                .and_then(|applied| applied);
                if let Err(error) = applied {
                    tracing::warn!(%error, "applying an auth cache invalidation to Python caches failed");
                }
            }
        });
        *subscription = Some(task.abort_handle());
    }

    fn stop(&self) {
        self.unsubscribe();
    }
}

impl NativeAuthCacheInvalidation {
    fn unsubscribe(&self) {
        if let Some(task) = self.subscription.lock().take() {
            task.abort();
        }
    }
}

impl Drop for NativeAuthCacheInvalidation {
    fn drop(&mut self) {
        self.unsubscribe();
    }
}
