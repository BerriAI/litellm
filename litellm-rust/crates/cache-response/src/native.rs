use std::{sync::Arc, time::Duration};

use litellm_cache::{CacheEntry, Error};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::RedisCache;
use serde_json::Value;

use crate::{ResponseCache, ResponseCacheCodec, ResponseCacheRequest};

pub enum NativeResponseCache<C = redis::Connection>
where
    C: redis::ConnectionLike + Send + 'static,
{
    Memory(Arc<ResponseCache<InMemoryCache<CacheEntry>>>),
    Redis(Arc<ResponseCache<RedisCache<ResponseCacheCodec, C>>>),
}

impl<C: redis::ConnectionLike + Send + 'static> Clone for NativeResponseCache<C> {
    fn clone(&self) -> Self {
        match self {
            Self::Memory(cache) => Self::Memory(Arc::clone(cache)),
            Self::Redis(cache) => Self::Redis(Arc::clone(cache)),
        }
    }
}

impl NativeResponseCache {
    pub fn memory(capacity: usize, ttl: Duration, max_entry_bytes: usize) -> Self {
        Self::Memory(Arc::new(ResponseCache::new(Arc::new(
            InMemoryCache::response_cache(capacity, ttl, max_entry_bytes),
        ))))
    }

    pub fn redis(
        url: &str,
        ttl: Option<Duration>,
        namespace: Option<String>,
    ) -> Result<Self, Error> {
        let backend = RedisCache::new(url, ttl, ResponseCacheCodec)?.with_namespace(namespace);
        Ok(Self::Redis(Arc::new(ResponseCache::new(Arc::new(backend)))))
    }
}

impl<C: redis::ConnectionLike + Send + 'static> NativeResponseCache<C> {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Memory(_) => "memory",
            Self::Redis(_) => "redis",
        }
    }

    pub fn lookup(
        &self,
        request: &ResponseCacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.lookup(request, now),
            Self::Redis(cache) => cache.lookup(request, now),
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
            Self::Redis(cache) => cache.store(request, response, now),
        }
    }

    pub async fn async_lookup(
        &self,
        request: &ResponseCacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.async_lookup(request, now).await,
            Self::Redis(cache) => cache.async_lookup(request, now).await,
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
            Self::Redis(cache) => cache.async_store(request, response, now).await,
        }
    }
}
