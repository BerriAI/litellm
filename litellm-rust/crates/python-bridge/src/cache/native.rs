use std::{sync::Arc, time::Duration};

use litellm_cache::{CacheCodec, CacheConnectionResult, Error};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::RedisCache;
use litellm_cache_response::{
    CacheEntry, PartialHits, ResponseCache, ResponseCacheCodec, ResponseCacheRequest,
};
use serde_json::Value;
use tokio::sync::Mutex;

#[derive(Clone)]
pub(super) enum NativeResponseCache {
    Memory(Arc<ResponseCache<InMemoryCache<CacheEntry>>>),
    Redis {
        cache: Arc<ResponseCache<RedisCache<ResponseCacheCodec>>>,
        buffer: Option<Arc<RedisWriteBuffer>>,
    },
}

pub(super) struct RedisWriteBuffer {
    flush_size: usize,
    entries: Mutex<Vec<(ResponseCacheRequest, Value, Duration)>>,
}

impl NativeResponseCache {
    pub fn memory(capacity: usize, ttl: Duration, max_entry_bytes: usize) -> Self {
        Self::Memory(Arc::new(ResponseCache::new(Arc::new(
            InMemoryCache::with_clock_and_size_measurement(
                Some(capacity),
                Some(ttl),
                Some(max_entry_bytes),
                Some(Arc::new(|entry| {
                    ResponseCacheCodec.encode(entry).map(|bytes| bytes.len())
                })),
                super::now,
            ),
        ))))
    }

    pub fn redis(
        url: &str,
        ttl: Option<Duration>,
        namespace: Option<String>,
    ) -> Result<Self, Error> {
        let backend = RedisCache::new(url, ttl, ResponseCacheCodec)?.with_namespace(namespace);
        Ok(Self::Redis {
            cache: Arc::new(ResponseCache::new(Arc::new(backend))),
            buffer: None,
        })
    }
}

impl NativeResponseCache {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Memory(_) => "memory",
            Self::Redis { .. } => "redis",
        }
    }

    pub fn default_ttl(&self) -> Duration {
        match self {
            Self::Memory(cache) => cache.default_ttl(),
            Self::Redis { cache, .. } => cache.default_ttl(),
        }
    }

    pub fn namespace(&self) -> Option<&str> {
        match self {
            Self::Memory(_) => None,
            Self::Redis { cache, .. } => cache.backend().namespace(),
        }
    }

    pub fn capacity(&self) -> Option<usize> {
        match self {
            Self::Memory(cache) => Some(cache.backend().max_size_in_memory()),
            Self::Redis { .. } => None,
        }
    }

    pub fn with_redis_flush_size(self, flush_size: Option<usize>) -> Self {
        match self {
            Self::Redis { cache, .. } => Self::Redis {
                cache,
                buffer: flush_size.map(|flush_size| {
                    Arc::new(RedisWriteBuffer {
                        flush_size: flush_size.max(1),
                        entries: Mutex::new(Vec::new()),
                    })
                }),
            },
            memory => memory,
        }
    }

    pub fn lookup(
        &self,
        request: &ResponseCacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.lookup(request, now),
            Self::Redis { cache, .. } => cache.lookup(request, now),
        }
    }

    pub fn store(
        &self,
        request: &ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.store(request, response, now),
            Self::Redis { cache, .. } => cache.store(request, response, now),
        }
    }

    pub fn lookup_batch(
        &self,
        requests: &[ResponseCacheRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Memory(cache) => cache.lookup_batch(requests, now),
            Self::Redis { cache, .. } => cache.lookup_batch(requests, now),
        }
    }

    pub async fn async_lookup(
        &self,
        request: &ResponseCacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.async_lookup(request, now).await,
            Self::Redis { cache, .. } => cache.async_lookup(request, now).await,
        }
    }

    pub async fn async_store(
        &self,
        request: &ResponseCacheRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.async_store(request, response, now).await,
            Self::Redis {
                cache,
                buffer: None,
            } => cache.async_store(request, response, now).await,
            Self::Redis {
                cache,
                buffer: Some(buffer),
            } => {
                let pending = {
                    let mut entries = buffer.entries.lock().await;
                    entries.push((request.clone(), response, now));
                    (entries.len() >= buffer.flush_size).then(|| std::mem::take(&mut *entries))
                };
                // A failed flush drops its batch, as Python does. Requeueing would grow the
                // buffer and re-send an ever larger pipeline on every write during an outage.
                match pending {
                    Some(pending) => cache.async_store_entries(pending).await,
                    None => Ok(()),
                }
            }
        }
    }

    pub async fn async_lookup_batch(
        &self,
        requests: &[ResponseCacheRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Memory(cache) => cache.async_lookup_batch(requests, now).await,
            Self::Redis { cache, .. } => cache.async_lookup_batch(requests, now).await,
        }
    }

    pub async fn async_store_batch(
        &self,
        entries: Vec<(ResponseCacheRequest, Value)>,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.async_store_batch(entries, now).await,
            Self::Redis { cache, .. } => cache.async_store_batch(entries, now).await,
        }
    }

    pub async fn async_flush(&self) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.async_flush().await,
            Self::Redis { cache, buffer } => {
                if let Some(buffer) = buffer {
                    buffer.entries.lock().await.clear();
                }
                cache.async_flush().await
            }
        }
    }

    pub async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        match self {
            Self::Memory(cache) => cache.test_connection().await,
            Self::Redis { cache, .. } => cache.test_connection().await,
        }
    }
}
