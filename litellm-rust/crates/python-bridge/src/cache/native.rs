use std::{path::Path, sync::Arc, time::Duration};

use litellm_cache::{CacheCodec, CacheConnectionResult, Error, ExactCacheContext};
use litellm_cache_azure_blob::AzureBlobCache;
use litellm_cache_disk::DiskCache;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::{RedisCache, RedisTopology};
use litellm_cache_redis_semantic::{RedisSemanticCache, RedisSemanticConfig};
use litellm_cache_response::{
    CacheEntry, PartialHits, ResponseCache, ResponseCacheCodec, ResponseCacheRequest, WriteBuffer,
};
use pyo3::{Py, PyAny, PyTraverseError, PyVisit};
use serde_json::Value;

use super::{embedder::PythonEmbedder, request::CacheRequest};

#[derive(Clone)]
pub(super) enum NativeResponseCache {
    Memory(Arc<ResponseCache<InMemoryCache<CacheEntry>>>),
    Redis {
        cache: Arc<ResponseCache<RedisCache<ResponseCacheCodec>>>,
        buffer: Option<Arc<WriteBuffer>>,
    },
    Disk(Arc<ResponseCache<DiskCache<ResponseCacheCodec>>>),
    AzureBlob(Arc<ResponseCache<AzureBlobCache<ResponseCacheCodec>>>),
    RedisSemantic(Arc<ResponseCache<RedisSemanticCache<PythonEmbedder>>>),
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
    pub fn disk(directory: &str) -> Result<Self, Error> {
        let cache = DiskCache::open(directory, ResponseCacheCodec)?;
        Ok(Self::Disk(Arc::new(ResponseCache::new(Arc::new(cache)))))
    }

    pub fn redis_semantic(
        url: &str,
        embedder: PythonEmbedder,
        config: RedisSemanticConfig,
    ) -> Result<Self, Error> {
        let backend = RedisSemanticCache::new(url, embedder, config)?;
        Ok(Self::RedisSemantic(Arc::new(ResponseCache::new(Arc::new(
            backend,
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
            Self::Memory(_) | Self::Redis { .. } | Self::Disk(_) | Self::RedisSemantic(_) => None,
        }
    }
}

impl NativeResponseCache {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Memory(_) => "memory",
            Self::Redis { .. } => "redis",
            Self::Disk(_) => "disk",
            Self::RedisSemantic(_) => "redis_semantic",
            Self::AzureBlob(_) => "azure-blob",
        }
    }

    pub fn default_ttl(&self) -> Option<Duration> {
        match self {
            Self::Memory(cache) => cache.default_ttl(),
            Self::Redis { cache, .. } => cache.default_ttl(),
            Self::Disk(cache) => cache.default_ttl(),
            Self::RedisSemantic(cache) => cache.default_ttl(),
            Self::AzureBlob(cache) => cache.default_ttl(),
        }
    }

    pub fn namespace(&self) -> Option<&str> {
        match self {
            Self::Memory(_) | Self::Disk(_) | Self::AzureBlob(_) | Self::RedisSemantic(_) => None,
            Self::Redis { cache, .. } => cache.backend().namespace(),
        }
    }

    pub fn topology(&self) -> Option<&RedisTopology> {
        match self {
            Self::Memory(_) | Self::Disk(_) | Self::AzureBlob(_) | Self::RedisSemantic(_) => None,
            Self::Redis { cache, .. } => Some(cache.backend().topology()),
        }
    }

    pub fn capacity(&self) -> Option<usize> {
        match self {
            Self::Memory(cache) => Some(cache.backend().max_size_in_memory()),
            Self::Redis { .. } | Self::Disk(_) | Self::AzureBlob(_) | Self::RedisSemantic(_) => {
                None
            }
        }
    }

    pub fn max_entry_bytes(&self) -> Option<usize> {
        match self {
            Self::Memory(cache) => cache.backend().max_entry_bytes(),
            Self::Redis { .. } | Self::Disk(_) | Self::AzureBlob(_) | Self::RedisSemantic(_) => {
                None
            }
        }
    }

    pub fn index_name(&self) -> Option<&str> {
        match self {
            Self::RedisSemantic(cache) => Some(cache.backend().index_name()),
            Self::Memory(_) | Self::Redis { .. } | Self::AzureBlob(_) | Self::Disk(_) => None,
        }
    }

    pub fn similarity_threshold(&self) -> Option<f32> {
        match self {
            Self::RedisSemantic(cache) => Some(cache.backend().similarity_threshold()),
            Self::Memory(_) | Self::Redis { .. } | Self::AzureBlob(_) | Self::Disk(_) => None,
        }
    }

    pub fn semantic_embedder(&self) -> Option<&PythonEmbedder> {
        match self {
            Self::RedisSemantic(cache) => Some(cache.backend().embedder()),
            Self::Memory(_) | Self::Redis { .. } | Self::AzureBlob(_) | Self::Disk(_) => None,
        }
    }

    pub fn embedder_object(&self) -> Option<&Py<PyAny>> {
        match self {
            Self::RedisSemantic(cache) => Some(cache.backend().embedder().object()),
            Self::Memory(_) | Self::Redis { .. } | Self::AzureBlob(_) | Self::Disk(_) => None,
        }
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Self::RedisSemantic(cache) = self {
            cache.backend().embedder().traverse(visit)?;
        }
        Ok(())
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

    pub fn directory(&self) -> Option<&Path> {
        match self {
            Self::Disk(cache) => Some(cache.backend().directory()),
            Self::Memory(_) | Self::Redis { .. } | Self::AzureBlob(_) | Self::RedisSemantic(_) => {
                None
            }
        }
    }

    fn exact_requests(requests: &[CacheRequest]) -> Vec<ResponseCacheRequest<ExactCacheContext>> {
        requests.iter().map(CacheRequest::exact).collect()
    }

    pub fn lookup(&self, request: &CacheRequest, now: Duration) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.lookup(&request.exact(), now),
            Self::Redis { cache, .. } => cache.lookup(&request.exact(), now),
            Self::Disk(cache) => cache.lookup(&request.exact(), now),
            Self::AzureBlob(cache) => cache.lookup(&request.exact(), now),
            Self::RedisSemantic(cache) => cache.lookup(&request.semantic(), now),
        }
    }

    pub fn store(
        &self,
        request: &CacheRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.store(&request.exact(), response, now),
            Self::Redis { cache, .. } => cache.store(&request.exact(), response, now),
            Self::Disk(cache) => cache.store(&request.exact(), response, now),
            Self::AzureBlob(cache) => cache.store(&request.exact(), response, now),
            Self::RedisSemantic(cache) => cache.store(&request.semantic(), response, now),
        }
    }

    pub fn lookup_batch(
        &self,
        requests: &[CacheRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Memory(cache) => cache.lookup_batch(&Self::exact_requests(requests), now),
            Self::Redis { cache, .. } => cache.lookup_batch(&Self::exact_requests(requests), now),
            Self::Disk(cache) => cache.lookup_batch(&Self::exact_requests(requests), now),
            Self::AzureBlob(cache) => cache.lookup_batch(&Self::exact_requests(requests), now),
            Self::RedisSemantic(_) => Err(Error::UnsupportedOperation),
        }
    }

    pub async fn async_lookup(
        &self,
        request: &CacheRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.async_lookup(&request.exact(), now).await,
            Self::Redis { cache, .. } => cache.async_lookup(&request.exact(), now).await,
            Self::Disk(cache) => cache.async_lookup(&request.exact(), now).await,
            Self::AzureBlob(cache) => cache.async_lookup(&request.exact(), now).await,
            Self::RedisSemantic(cache) => cache.async_lookup(&request.semantic(), now).await,
        }
    }

    pub async fn async_store(
        &self,
        request: &CacheRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.async_store(&request.exact(), response, now).await,
            Self::Redis {
                cache,
                buffer: None,
            } => cache.async_store(&request.exact(), response, now).await,
            Self::Redis {
                cache,
                buffer: Some(buffer),
            } => {
                buffer
                    .async_store(cache, &request.exact(), response, now)
                    .await
            }
            Self::Disk(cache) => cache.async_store(&request.exact(), response, now).await,
            Self::AzureBlob(cache) => cache.async_store(&request.exact(), response, now).await,
            Self::RedisSemantic(cache) => {
                cache.async_store(&request.semantic(), response, now).await
            }
        }
    }

    pub async fn async_lookup_batch(
        &self,
        requests: &[CacheRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Memory(cache) => {
                cache
                    .async_lookup_batch(&Self::exact_requests(requests), now)
                    .await
            }
            Self::Redis { cache, .. } => {
                cache
                    .async_lookup_batch(&Self::exact_requests(requests), now)
                    .await
            }
            Self::Disk(cache) => {
                cache
                    .async_lookup_batch(&Self::exact_requests(requests), now)
                    .await
            }
            Self::AzureBlob(cache) => {
                cache
                    .async_lookup_batch(&Self::exact_requests(requests), now)
                    .await
            }
            Self::RedisSemantic(_) => Err(Error::UnsupportedOperation),
        }
    }

    pub async fn async_store_batch(
        &self,
        entries: Vec<(CacheRequest, Value)>,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => {
                cache
                    .async_store_batch(
                        entries
                            .into_iter()
                            .map(|(request, value)| (request.exact(), value))
                            .collect(),
                        now,
                    )
                    .await
            }
            Self::Redis { cache, .. } => {
                cache
                    .async_store_batch(
                        entries
                            .into_iter()
                            .map(|(request, value)| (request.exact(), value))
                            .collect(),
                        now,
                    )
                    .await
            }
            Self::Disk(cache) => {
                cache
                    .async_store_batch(
                        entries
                            .into_iter()
                            .map(|(request, value)| (request.exact(), value))
                            .collect(),
                        now,
                    )
                    .await
            }
            Self::AzureBlob(cache) => {
                cache
                    .async_store_batch(
                        entries
                            .into_iter()
                            .map(|(request, value)| (request.exact(), value))
                            .collect(),
                        now,
                    )
                    .await
            }
            Self::RedisSemantic(cache) => {
                cache
                    .async_store_batch(
                        entries
                            .into_iter()
                            .map(|(request, value)| (request.semantic(), value))
                            .collect(),
                        now,
                    )
                    .await
            }
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
            Self::Disk(cache) => cache.async_flush().await,
            Self::RedisSemantic(_) => Err(Error::UnsupportedOperation),
            Self::AzureBlob(cache) => cache.async_flush().await,
        }
    }

    pub async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        match self {
            Self::Memory(cache) => cache.test_connection().await,
            Self::Redis { cache, .. } => cache.test_connection().await,
            Self::Disk(cache) => cache.test_connection().await,
            Self::RedisSemantic(_) => Err(Error::UnsupportedOperation),
            Self::AzureBlob(cache) => cache.test_connection().await,
        }
    }
}
