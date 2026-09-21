use std::{sync::Arc, time::Duration};

use litellm_cache::{CacheCodec, CacheConnectionResult, Error};
use litellm_cache_azure_blob::AzureBlobCache;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::{RedisCache, RedisTopology};
use litellm_cache_response::{
    CacheEntry, PartialHits, ResponseCache, ResponseCacheCodec, ResponseCacheRequest, WriteBuffer,
};
use litellm_cache_s3::{S3Cache, S3CacheConfig};
use serde_json::Value;

#[derive(Clone)]
pub(super) enum NativeResponseCache {
    Memory(Arc<ResponseCache<InMemoryCache<CacheEntry>>>),
    Redis {
        cache: Arc<ResponseCache<RedisCache<ResponseCacheCodec>>>,
        buffer: Option<Arc<WriteBuffer>>,
    },
    S3(Arc<ResponseCache<S3Cache<ResponseCacheCodec>>>),
    AzureBlob(Arc<ResponseCache<AzureBlobCache<ResponseCacheCodec>>>),
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
                super::request::now,
            ),
        ))))
    }

    pub fn redis(
        url: &str,
        topology: &RedisTopology,
        ttl: Option<Duration>,
        namespace: Option<String>,
    ) -> Result<Self, Error> {
        let backend =
            RedisCache::connect(url, topology, ttl, ResponseCacheCodec)?.with_namespace(namespace);
        Ok(Self::Redis {
            cache: Arc::new(ResponseCache::new(Arc::new(backend))),
            buffer: None,
        })
    }
    pub async fn s3(config: S3CacheConfig) -> Self {
        let runtime = tokio::runtime::Handle::current();
        Self::S3(Arc::new(ResponseCache::new(Arc::new(S3Cache::new(
            config,
            ResponseCacheCodec,
            runtime,
        )))))
    }

    pub async fn azure_blob(account_url: &str, container: &str) -> Result<Self, Error> {
        let backend = AzureBlobCache::connect(
            account_url,
            container,
            ResponseCacheCodec,
            tokio::runtime::Handle::current(),
        )
        .await?;
        Ok(Self::AzureBlob(Arc::new(ResponseCache::new(Arc::new(
            backend,
        )))))
    }

    pub fn azure_blob_identity(&self) -> Option<(&str, &str)> {
        match self {
            Self::AzureBlob(cache) => Some((
                cache.backend().account_url(),
                cache.backend().container_name(),
            )),
            Self::Memory(_) | Self::Redis { .. } | Self::S3(_) => None,
        }
    }
}

impl NativeResponseCache {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Memory(_) => "memory",
            Self::Redis { .. } => "redis",
            Self::S3(_) => "s3",
            Self::AzureBlob(_) => "azure-blob",
        }
    }

    pub fn default_ttl(&self) -> Option<Duration> {
        match self {
            Self::Memory(cache) => cache.default_ttl(),
            Self::Redis { cache, .. } => cache.default_ttl(),
            Self::S3(cache) => cache.default_ttl(),
            Self::AzureBlob(cache) => cache.default_ttl(),
        }
    }

    pub fn bucket(&self) -> Option<&str> {
        match self {
            Self::S3(cache) => Some(cache.backend().bucket()),
            _ => None,
        }
    }

    pub fn key_prefix(&self) -> Option<&str> {
        match self {
            Self::S3(cache) => Some(cache.backend().key_prefix()),
            _ => None,
        }
    }

    pub fn region(&self) -> Option<&str> {
        match self {
            Self::S3(cache) => Some(cache.backend().region()),
            _ => None,
        }
    }

    pub fn endpoint(&self) -> Option<&str> {
        match self {
            Self::S3(cache) => cache.backend().endpoint(),
            _ => None,
        }
    }

    pub fn namespace(&self) -> Option<&str> {
        match self {
            Self::Memory(_) | Self::AzureBlob(_) => None,
            Self::Redis { cache, .. } => cache.backend().namespace(),
            Self::S3(_) => None,
        }
    }

    pub fn topology(&self) -> Option<&RedisTopology> {
        match self {
            Self::Memory(_) | Self::AzureBlob(_) => None,
            Self::Redis { cache, .. } => Some(cache.backend().topology()),
            Self::S3(_) => None,
        }
    }

    pub fn capacity(&self) -> Option<usize> {
        match self {
            Self::Memory(cache) => Some(cache.backend().max_size_in_memory()),
            Self::Redis { .. } | Self::S3(_) | Self::AzureBlob(_) => None,
        }
    }

    pub fn max_entry_bytes(&self) -> Option<usize> {
        match self {
            Self::Memory(cache) => cache.backend().max_entry_bytes(),
            Self::Redis { .. } | Self::S3(_) | Self::AzureBlob(_) => None,
        }
    }

    pub fn with_redis_flush_size(self, flush_size: Option<usize>) -> Self {
        match self {
            Self::Redis { cache, .. } => Self::Redis {
                cache,
                buffer: flush_size.map(|flush_size| Arc::new(WriteBuffer::new(flush_size))),
            },
            other => other,
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
            Self::S3(cache) => cache.lookup(request, now),
            Self::AzureBlob(cache) => cache.lookup(request, now),
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
            Self::S3(cache) => cache.store(request, response, now),
            Self::AzureBlob(cache) => cache.store(request, response, now),
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
            Self::S3(cache) => cache.lookup_batch(requests, now),
            Self::AzureBlob(cache) => cache.lookup_batch(requests, now),
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
            Self::S3(cache) => cache.async_lookup(request, now).await,
            Self::AzureBlob(cache) => cache.async_lookup(request, now).await,
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
            } => buffer.async_store(cache, request, response, now).await,
            Self::S3(cache) => cache.async_store(request, response, now).await,
            Self::AzureBlob(cache) => cache.async_store(request, response, now).await,
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
            Self::S3(cache) => cache.async_lookup_batch(requests, now).await,
            Self::AzureBlob(cache) => cache.async_lookup_batch(requests, now).await,
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
            Self::S3(cache) => cache.async_store_batch(entries, now).await,
            Self::AzureBlob(cache) => cache.async_store_batch(entries, now).await,
        }
    }

    pub async fn async_flush(&self) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.async_flush().await,
            Self::Redis { cache, buffer } => {
                if let Some(buffer) = buffer {
                    buffer.clear()?;
                }
                cache.async_flush().await
            }
            Self::S3(cache) => cache.async_flush().await,
            Self::AzureBlob(cache) => cache.async_flush().await,
        }
    }

    pub async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        match self {
            Self::Memory(cache) => cache.test_connection().await,
            Self::Redis { cache, .. } => cache.test_connection().await,
            Self::S3(cache) => cache.test_connection().await,
            Self::AzureBlob(cache) => cache.test_connection().await,
        }
    }
}
