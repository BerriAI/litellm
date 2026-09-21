use std::{sync::Arc, time::Duration};

use litellm_cache::{CacheCodec, CacheConnectionResult, Error, SemanticCacheContext};
use litellm_cache_azure_blob::AzureBlobCache;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_qdrant_semantic::{Embedder, OpenAiEmbedder, QdrantSemanticCache};
use litellm_cache_redis::{RedisCache, RedisTopology};
use litellm_cache_response::{
    CacheEntry, PartialHits, ResponseCache, ResponseCacheCodec, ResponseCacheRequest, WriteBuffer,
};
use serde_json::Value;

use super::{config::QdrantSemanticCacheConfig, request::exact};

#[derive(Clone)]
pub(super) enum NativeResponseCache {
    Memory(Arc<ResponseCache<InMemoryCache<CacheEntry>>>),
    Redis {
        cache: Arc<ResponseCache<RedisCache<ResponseCacheCodec>>>,
        buffer: Option<Arc<WriteBuffer>>,
    },
    QdrantSemantic(Arc<ResponseCache<QdrantSemanticCache<OpenAiEmbedder, ResponseCacheCodec>>>),
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

    pub async fn qdrant_semantic(
        config: QdrantSemanticCacheConfig,
        client: reqwest::Client,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self, Error> {
        let qdrant = qdrant_client::Qdrant::from_url(&config.grpc_url)
            .skip_compatibility_check()
            .api_key(config.api_key.as_deref())
            .build()
            .map_err(|_| Error::Unavailable)?;
        let qdrant_config = config.to_qdrant_config();
        let embedder = OpenAiEmbedder::new(client, config.embedding);
        let cache = QdrantSemanticCache::connect(
            qdrant,
            embedder,
            ResponseCacheCodec,
            qdrant_config,
            runtime,
        )
        .await?;
        Ok(Self::QdrantSemantic(Arc::new(ResponseCache::new(
            Arc::new(cache),
        ))))
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
            Self::Memory(_) | Self::Redis { .. } | Self::QdrantSemantic(_) => None,
        }
    }
}

impl NativeResponseCache {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Memory(_) => "memory",
            Self::Redis { .. } => "redis",
            Self::QdrantSemantic(_) => "qdrant_semantic",
            Self::AzureBlob(_) => "azure-blob",
        }
    }

    pub fn default_ttl(&self) -> Option<Duration> {
        match self {
            Self::Memory(cache) => cache.default_ttl(),
            Self::Redis { cache, .. } => cache.default_ttl(),
            Self::QdrantSemantic(_) => None,
            Self::AzureBlob(cache) => cache.default_ttl(),
        }
    }

    pub fn namespace(&self) -> Option<&str> {
        match self {
            Self::Memory(_) | Self::AzureBlob(_) => None,
            Self::Redis { cache, .. } => cache.backend().namespace(),
            Self::QdrantSemantic(_) => None,
        }
    }

    pub fn topology(&self) -> Option<&RedisTopology> {
        match self {
            Self::Memory(_) | Self::AzureBlob(_) => None,
            Self::Redis { cache, .. } => Some(cache.backend().topology()),
            Self::QdrantSemantic(_) => None,
        }
    }

    pub fn capacity(&self) -> Option<usize> {
        match self {
            Self::Memory(cache) => Some(cache.backend().max_size_in_memory()),
            Self::Redis { .. } => None,
            Self::QdrantSemantic(_) => None,
            Self::AzureBlob(_) => None,
        }
    }

    pub fn max_entry_bytes(&self) -> Option<usize> {
        match self {
            Self::Memory(cache) => cache.backend().max_entry_bytes(),
            Self::Redis { .. } => None,
            Self::QdrantSemantic(_) => None,
            Self::AzureBlob(_) => None,
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

    pub fn collection_name(&self) -> Option<&str> {
        match self {
            Self::QdrantSemantic(cache) => Some(cache.backend().collection_name()),
            _ => None,
        }
    }

    pub fn similarity_threshold(&self) -> Option<f64> {
        match self {
            Self::QdrantSemantic(cache) => Some(cache.backend().similarity_threshold()),
            _ => None,
        }
    }

    pub fn vector_size(&self) -> Option<u64> {
        match self {
            Self::QdrantSemantic(cache) => Some(cache.backend().vector_size()),
            _ => None,
        }
    }

    pub fn embedding_model(&self) -> Option<&str> {
        match self {
            Self::QdrantSemantic(cache) => Some(cache.backend().embedder().model()),
            _ => None,
        }
    }

    pub fn lookup(
        &self,
        request: &ResponseCacheRequest<SemanticCacheContext>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.lookup(&exact(request), now),
            Self::Redis { cache, .. } => cache.lookup(&exact(request), now),
            Self::QdrantSemantic(cache) => cache.lookup(request, now),
            Self::AzureBlob(cache) => cache.lookup(&exact(request), now),
        }
    }

    pub fn store(
        &self,
        request: &ResponseCacheRequest<SemanticCacheContext>,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.store(&exact(request), response, now),
            Self::Redis { cache, .. } => cache.store(&exact(request), response, now),
            Self::QdrantSemantic(cache) => cache.store(request, response, now),
            Self::AzureBlob(cache) => cache.store(&exact(request), response, now),
        }
    }

    pub fn lookup_batch(
        &self,
        requests: &[ResponseCacheRequest<SemanticCacheContext>],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Memory(cache) => {
                cache.lookup_batch(&requests.iter().map(exact).collect::<Vec<_>>(), now)
            }
            Self::Redis { cache, .. } => {
                cache.lookup_batch(&requests.iter().map(exact).collect::<Vec<_>>(), now)
            }
            Self::QdrantSemantic(_) => Err(Error::UnsupportedOperation),
            Self::AzureBlob(cache) => {
                cache.lookup_batch(&requests.iter().map(exact).collect::<Vec<_>>(), now)
            }
        }
    }

    pub async fn async_lookup(
        &self,
        request: &ResponseCacheRequest<SemanticCacheContext>,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.async_lookup(&exact(request), now).await,
            Self::Redis { cache, .. } => cache.async_lookup(&exact(request), now).await,
            Self::QdrantSemantic(cache) => cache.async_lookup(request, now).await,
            Self::AzureBlob(cache) => cache.async_lookup(&exact(request), now).await,
        }
    }

    pub async fn async_store(
        &self,
        request: &ResponseCacheRequest<SemanticCacheContext>,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.async_store(&exact(request), response, now).await,
            Self::Redis {
                cache,
                buffer: None,
            } => cache.async_store(&exact(request), response, now).await,
            Self::Redis {
                cache,
                buffer: Some(buffer),
            } => {
                let request = exact(request);
                buffer.async_store(cache, &request, response, now).await
            }
            Self::QdrantSemantic(cache) => cache.async_store(request, response, now).await,
            Self::AzureBlob(cache) => cache.async_store(&exact(request), response, now).await,
        }
    }

    pub async fn async_lookup_batch(
        &self,
        requests: &[ResponseCacheRequest<SemanticCacheContext>],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Memory(cache) => {
                let requests = requests.iter().map(exact).collect::<Vec<_>>();
                cache.async_lookup_batch(&requests, now).await
            }
            Self::Redis { cache, .. } => {
                let requests = requests.iter().map(exact).collect::<Vec<_>>();
                cache.async_lookup_batch(&requests, now).await
            }
            Self::QdrantSemantic(_) => Err(Error::UnsupportedOperation),
            Self::AzureBlob(cache) => {
                let requests = requests.iter().map(exact).collect::<Vec<_>>();
                cache.async_lookup_batch(&requests, now).await
            }
        }
    }

    pub async fn async_store_batch(
        &self,
        entries: Vec<(ResponseCacheRequest<SemanticCacheContext>, Value)>,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => {
                cache
                    .async_store_batch(
                        entries
                            .into_iter()
                            .map(|(request, value)| (exact(&request), value))
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
                            .map(|(request, value)| (exact(&request), value))
                            .collect(),
                        now,
                    )
                    .await
            }
            Self::QdrantSemantic(cache) => cache.async_store_batch(entries, now).await,
            Self::AzureBlob(cache) => {
                cache
                    .async_store_batch(
                        entries
                            .into_iter()
                            .map(|(request, value)| (exact(&request), value))
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
            Self::QdrantSemantic(_) => Err(Error::UnsupportedOperation),
            Self::AzureBlob(cache) => cache.async_flush().await,
        }
    }

    pub async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        match self {
            Self::Memory(cache) => cache.test_connection().await,
            Self::Redis { cache, .. } => cache.test_connection().await,
            Self::QdrantSemantic(_) => Err(Error::UnsupportedOperation),
            Self::AzureBlob(cache) => cache.test_connection().await,
        }
    }
}
