mod keys;
mod types;

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures_util::{StreamExt, stream::BoxStream};
use litellm_redis_client::RedisClient;
use tokio::{runtime::Handle, sync::Semaphore};

pub use self::keys::channel;
pub use self::types::Invalidation;
use crate::{
    auth_cache::AuthCache,
    spend_counters::{SpendCounter, SpendCounters},
};

const MAX_PENDING_PUBLISHES: usize = 1024;
const MAX_IN_FLIGHT_PUBLISHES: usize = 16;
const BULK_EVICTION_WAIT: Duration = Duration::from_secs(2);

/// The `litellm_proxy.auth_cache_invalidation` channel every worker shares. Publishing is best
/// effort and never holds up the caller: a publish runs in the background, at most 16 hold a
/// connection at once, and past 1024 waiting ones a new one is dropped with a warning, so a Redis
/// that accepts connections but never replies cannot wedge the pool or the request.
#[derive(Clone)]
pub struct AuthCacheInvalidation {
    client: RedisClient,
    channel: Arc<str>,
    runtime: Handle,
    in_flight: Arc<Semaphore>,
    pending: Arc<AtomicUsize>,
}

impl AuthCacheInvalidation {
    /// `namespace` is `RedisCache.namespace`, which main prefixed the channel with. The client's
    /// own namespace misses one assigned to the `RedisCache` after it was built.
    pub fn new(client: RedisClient, namespace: Option<&str>, runtime: Handle) -> Self {
        Self {
            channel: channel(namespace).into(),
            client,
            runtime,
            in_flight: Arc::new(Semaphore::new(MAX_IN_FLIGHT_PUBLISHES)),
            pending: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn runtime(&self) -> &Handle {
        &self.runtime
    }

    pub fn channel(&self) -> &str {
        &self.channel
    }

    /// `publish_auth_cache_invalidation`: one `PUBLISH` in the background. `false` when it was
    /// dropped because too many publishes are already waiting.
    pub fn publish(&self, invalidation: &Invalidation) -> bool {
        let payload = invalidation.encode().into_bytes();
        let client = self.client.clone();
        let channel = self.channel.clone();
        let cache_key = invalidation.cache_key().to_owned();
        self.spawn(1, async move {
            if let Err(error) = client.publish(&channel, &[payload]).await {
                tracing::warn!(%error, cache_key, "auth cache invalidation publish failed");
            }
        })
        .is_some()
    }

    /// `broadcast_evictions`: one pipeline of `PUBLISH`es, waited for at most two seconds.
    pub async fn broadcast_evictions(&self, cache_keys: &[String]) {
        if cache_keys.is_empty() {
            return;
        }
        let payloads: Vec<_> = cache_keys
            .iter()
            .map(|cache_key| Invalidation::Evict(cache_key.clone()).encode().into_bytes())
            .collect();
        let client = self.client.clone();
        let channel = self.channel.clone();
        let count = cache_keys.len();
        let Some(task) = self.spawn(count, async move {
            if let Err(error) = client.publish(&channel, &payloads).await {
                tracing::warn!(%error, count, "auth cache invalidation publish failed");
            }
        }) else {
            return;
        };
        drop(tokio::time::timeout(BULK_EVICTION_WAIT, task).await);
    }

    /// Every message other workers publish, and this worker's own, decoded; undecodable ones are
    /// skipped. It resubscribes after a lost connection.
    pub fn subscribe(&self) -> BoxStream<'static, Invalidation> {
        self.client
            .subscribe(&self.channel)
            .filter_map(|payload| std::future::ready(Invalidation::decode(&payload)))
            .boxed()
    }

    fn spawn(
        &self,
        keys: usize,
        publish: impl Future<Output = ()> + Send + 'static,
    ) -> Option<tokio::task::JoinHandle<()>> {
        let pending = self.pending.fetch_add(1, Ordering::AcqRel);
        if pending >= MAX_PENDING_PUBLISHES {
            self.pending.fetch_sub(1, Ordering::AcqRel);
            tracing::warn!(
                keys,
                pending,
                "auth cache invalidation publish dropped: publishes already waiting on Redis; \
                 other workers keep their cached copy until its TTL expires"
            );
            return None;
        }
        let in_flight = self.in_flight.clone();
        let waiting = self.pending.clone();
        Some(self.runtime.spawn(async move {
            if let Ok(_permit) = in_flight.acquire_owned().await {
                publish.await;
            }
            waiting.fetch_sub(1, Ordering::AcqRel);
        }))
    }
}

/// What a message does to this worker's native caches: an eviction drops the auth cache's copy, and
/// both kinds reach the spend counters, which own only the `spend:*` keys.
pub fn apply(invalidation: &Invalidation, auth_cache: &AuthCache, spend_counters: &SpendCounters) {
    let applied = match invalidation {
        Invalidation::Evict(cache_key) => auth_cache.delete_local(cache_key).and_then(|()| {
            SpendCounter::parse(cache_key)
                .map_or(Ok(()), |counter| spend_counters.delete_local(&counter))
        }),
        Invalidation::Set {
            cache_key,
            value,
            ttl,
        } => SpendCounter::parse(cache_key).map_or(Ok(()), |counter| {
            spend_counters.set_local(&counter, *value, Invalidation::memory_ttl(*ttl))
        }),
    };
    if let Err(error) = applied {
        tracing::warn!(%error, cache_key = invalidation.cache_key(), "applying an auth cache invalidation failed");
    }
}
