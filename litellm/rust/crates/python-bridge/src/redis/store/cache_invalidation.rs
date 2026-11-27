use std::{future::Future, sync::Arc};

use futures_util::{Stream, StreamExt};
use litellm_host_python::{PythonContext, attach_blocking};
use litellm_redis_store::{
    auth_cache::AuthCache,
    cache_invalidation::{self, CacheInvalidation, Invalidation},
    spend_counters::SpendCounters,
};
use parking_lot::Mutex;
use pyo3::prelude::*;
use tokio::task::AbortHandle;

use super::{PyNumber, auth_cache::NativeAuthCache, spend_counters::NativeSpendCounters};
use crate::{execution::run_async_value, redis::client::NativeRedisClient};

const HOST_MODULE: &str = "litellm.rust_bridge.host.cache_invalidation";
const MAX_INVALIDATION_BATCH: usize = 128;

/// The auth cache invalidation channel: publishes in the background, and while started applies
/// every message to the native auth cache and spend counters, then hands it to the host for the
/// caches still in Python (`python_caches`, held opaquely). The subscription ends at `stop` or when
/// this is dropped.
#[pyclass(frozen, name = "NativeCacheInvalidation")]
pub(crate) struct NativeCacheInvalidation {
    channel: CacheInvalidation,
    auth_cache: Arc<AuthCache>,
    spend_counters: Arc<SpendCounters>,
    apply_python: Arc<Py<PyAny>>,
    python_caches: Arc<Py<PyAny>>,
    context: PythonContext,
    subscription: Mutex<Option<AbortHandle>>,
}

#[pymethods]
impl NativeCacheInvalidation {
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
            channel: CacheInvalidation::new(
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

    fn start(&self) {
        let mut subscription = self.subscription.lock();
        if subscription.is_some() {
            return;
        }
        let messages = self.channel.subscribe();
        let auth_cache = self.auth_cache.clone();
        let spend_counters = self.spend_counters.clone();
        let apply_python = self.apply_python.clone();
        let python_caches = self.python_caches.clone();
        let context = self.context.clone();
        let task = litellm_call_scope::spawn_detached_on(self.channel.runtime(), async move {
            apply_ready_invalidations(messages, |batch| {
                let auth_cache = auth_cache.clone();
                let spend_counters = spend_counters.clone();
                let apply_python = apply_python.clone();
                let python_caches = python_caches.clone();
                let applied = attach_blocking(context.clone(), move |py| {
                    for invalidation in batch {
                        py.detach(|| {
                            cache_invalidation::apply(&invalidation, &auth_cache, &spend_counters);
                        });
                        let (cache_key, new_value, ttl) = match invalidation {
                            Invalidation::Evict(cache_key) => (cache_key, None, None),
                            Invalidation::Set {
                                cache_key,
                                value,
                                ttl,
                            } => (cache_key, Some(value.as_f64()), ttl.map(|ttl| ttl.as_f64())),
                        };
                        if let Err(error) = apply_python.bind(py).call1((
                            python_caches.bind(py),
                            cache_key,
                            new_value,
                            ttl,
                        )) {
                            tracing::warn!(%error, "applying an auth cache invalidation to Python caches failed");
                        }
                    }
                });
                async move {
                    if let Err(error) = applied.await {
                        tracing::warn!(%error, "applying an auth cache invalidation to Python caches failed");
                    }
                }
            })
            .await;
        });
        *subscription = Some(task.abort_handle());
    }

    fn stop(&self) {
        self.unsubscribe();
    }
}

async fn apply_ready_invalidations<F: Future<Output = ()>>(
    messages: impl Stream<Item = Invalidation> + Unpin,
    mut apply: impl FnMut(Vec<Invalidation>) -> F,
) {
    let mut batches = messages.ready_chunks(MAX_INVALIDATION_BATCH);
    while let Some(batch) = batches.next().await {
        apply(batch).await;
    }
}

impl NativeCacheInvalidation {
    fn unsubscribe(&self) {
        if let Some(task) = self.subscription.lock().take() {
            task.abort();
        }
    }
}

impl Drop for NativeCacheInvalidation {
    fn drop(&mut self) {
        self.unsubscribe();
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, future::ready, pin::pin};

    use futures_util::{FutureExt, StreamExt, stream};
    use rstest::rstest;
    use tokio::sync::oneshot;

    use super::{Invalidation, MAX_INVALIDATION_BATCH, apply_ready_invalidations};

    #[rstest]
    fn ready_invalidations_are_applied_in_order_without_waiting_for_more() {
        let burst: Vec<_> = (0..MAX_INVALIDATION_BATCH)
            .map(|index| Invalidation::Evict(format!("team_id:{index}")))
            .collect();
        let lone = Invalidation::Evict("team_id:urgent".to_owned());
        let later = Invalidation::Evict("team_id:later".to_owned());
        let (send, receive) = oneshot::channel();
        let messages = stream::iter(burst.iter().cloned().chain([lone.clone()]))
            .chain(stream::once(async { receive.await.unwrap() }))
            .boxed_local();
        let seen = RefCell::new(Vec::new());
        let mut apply = pin!(apply_ready_invalidations(messages, |batch| {
            seen.borrow_mut().push(batch);
            ready(())
        }));

        assert!(apply.as_mut().now_or_never().is_none());
        assert_eq!(*seen.borrow(), [burst.clone(), vec![lone]]);
        send.send(later.clone()).unwrap();
        assert_eq!(apply.as_mut().now_or_never(), Some(()));
        assert_eq!(seen.borrow()[2], [later]);
    }
}
