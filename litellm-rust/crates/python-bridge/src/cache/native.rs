use std::{path::Path, sync::Arc, time::Duration};

use litellm_cache::{
    CacheCodec, CacheConnectionResult, Error, ExactCacheContext, SemanticCacheContext,
};
use litellm_cache_azure_blob::AzureBlobCache;
use litellm_cache_disk::DiskCache;
use litellm_cache_gcs::{GcsCache, GcsConfig, StaticTokenSource};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_qdrant_semantic::{Embedder, OpenAiEmbedder, QdrantSemanticCache};
use litellm_cache_redis::{RedisCache, RedisTopology};
use litellm_cache_redis_semantic::{RedisSemanticCache, RedisSemanticConfig};
use litellm_cache_response::{
    CacheEntry, CacheKeyField, PartialHits, ResponseCache, ResponseCacheCodec,
    ResponseCacheRequest, WriteBuffer,
};
use litellm_cache_s3::{S3Cache, S3CacheConfig};
use litellm_cache_valkey_semantic::{ValkeySemanticCache, ValkeySemanticConfig};
use pyo3::{PyTraverseError, PyVisit, prelude::*};
use serde_json::Value;

use super::{
    config::QdrantSemanticCacheConfig,
    embedder::PythonEmbedder,
    request::NativeRequest,
    semantic::{SemanticBody, SemanticOperation, drive},
    semantic_step::{SemanticEmbedExecution, drive_semantic},
};

fn semantic_key(request: &NativeRequest, scope: &str) -> litellm_cache_response::CacheKeyInput {
    let mut key = request.key.clone();
    if key.preset.is_some() {
        return key;
    }
    key.fields
        .retain(|field| !matches!(field.name.as_str(), "messages" | "prompt" | "input"));
    const TENANT: [&str; 3] = [
        "user_api_key",
        "user_api_key_team_id",
        "user_api_key_org_id",
    ];
    let end_user = (scope == "end_user").then_some("user_api_key_end_user_id");
    for name in TENANT.into_iter().chain(end_user) {
        let sources = [
            request.metadata.as_ref(),
            request.litellm_metadata.as_ref(),
            request
                .litellm_params
                .as_ref()
                .and_then(|params| params.get("metadata")),
            request
                .litellm_params
                .as_ref()
                .and_then(|params| params.get("litellm_metadata")),
        ];
        let Some(value) = sources.into_iter().flatten().find_map(|source| {
            source
                .as_object()
                .and_then(|values| values.get(name))
                .filter(|value| !value.is_null())
        }) else {
            continue;
        };
        let value = match value {
            Value::Null => continue,
            Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        key.fields.push(CacheKeyField {
            name: name.to_owned(),
            value: Some(value),
            api_parameter: true,
            internal_parameter: false,
        });
    }
    key
}

#[derive(Clone)]
pub(super) enum NativeResponseCache {
    Memory(Arc<ResponseCache<InMemoryCache<CacheEntry>>>),
    Redis {
        cache: Arc<ResponseCache<RedisCache<ResponseCacheCodec>>>,
        buffer: Option<Arc<WriteBuffer>>,
    },
    S3(Arc<ResponseCache<S3Cache<ResponseCacheCodec>>>),
    Gcs(Arc<ResponseCache<GcsCache<ResponseCacheCodec>>>),
    ValkeySemantic {
        cache: Arc<ResponseCache<ValkeySemanticCache<PythonEmbedder, ResponseCacheCodec>>>,
        embedder: PythonEmbedder,
        scope: String,
    },
    RedisSemantic {
        cache: Arc<ResponseCache<RedisSemanticCache<PythonEmbedder>>>,
        embedder: PythonEmbedder,
    },
    QdrantSemantic(Arc<ResponseCache<QdrantSemanticCache<OpenAiEmbedder, ResponseCacheCodec>>>),
    Disk(Arc<ResponseCache<DiskCache<ResponseCacheCodec>>>),
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

    pub fn valkey_semantic(
        url: &str,
        similarity_threshold: f64,
        index_name: String,
        embedder: PythonEmbedder,
    ) -> Result<Self, Error> {
        let backend = ValkeySemanticCache::new(
            url,
            embedder.clone(),
            ResponseCacheCodec,
            ValkeySemanticConfig {
                similarity_threshold,
                index_name,
            },
        )?;
        Ok(Self::ValkeySemantic {
            cache: Arc::new(ResponseCache::new(Arc::new(backend))),
            embedder,
            scope: String::from("key"),
        })
    }

    pub fn redis_semantic(
        url: &str,
        embedder: PythonEmbedder,
        config: RedisSemanticConfig,
    ) -> Result<Self, Error> {
        let backend = RedisSemanticCache::new(url, embedder.clone(), config)?;
        Ok(Self::RedisSemantic {
            cache: Arc::new(ResponseCache::new(Arc::new(backend))),
            embedder,
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

    pub fn disk(directory: &str) -> Result<Self, Error> {
        let cache = DiskCache::open(directory, ResponseCacheCodec)?;
        Ok(Self::Disk(Arc::new(ResponseCache::new(Arc::new(cache)))))
    }

    pub fn gcs(config: GcsConfig, token: Option<String>) -> Result<Self, Error> {
        let backend = match token {
            Some(token) => GcsCache::with_token_source(
                config,
                ResponseCacheCodec,
                Arc::new(StaticTokenSource(token)),
            )?,
            None => GcsCache::new(config, ResponseCacheCodec)?,
        };
        Ok(Self::Gcs(Arc::new(ResponseCache::new(Arc::new(backend)))))
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
            Self::Memory(_)
            | Self::Redis { .. }
            | Self::S3(_)
            | Self::ValkeySemantic { .. }
            | Self::RedisSemantic { .. }
            | Self::QdrantSemantic(_)
            | Self::Disk(_)
            | Self::Gcs(_) => None,
        }
    }

    fn exact(request: &NativeRequest) -> ResponseCacheRequest<ExactCacheContext> {
        ResponseCacheRequest {
            key: request.key.clone(),
            controls: request.controls,
            context: ExactCacheContext { ttl: request.ttl },
            max_age: request.max_age,
        }
    }

    pub(super) fn semantic_request(
        request: &NativeRequest,
    ) -> ResponseCacheRequest<SemanticCacheContext> {
        ResponseCacheRequest {
            key: request.key.clone(),
            controls: request.controls,
            context: SemanticCacheContext {
                input: request.input.clone(),
                messages: request.messages.clone(),
                metadata: request.metadata.clone(),
                scope: request.scope.clone(),
                ttl: request.ttl,
            },
            max_age: request.max_age,
        }
    }

    fn semantic(
        request: &NativeRequest,
        scope: &str,
    ) -> ResponseCacheRequest<SemanticCacheContext> {
        ResponseCacheRequest {
            key: semantic_key(request, scope),
            controls: request.controls,
            context: SemanticCacheContext {
                input: request.input.clone(),
                messages: request.messages.clone(),
                metadata: request.metadata.clone(),
                scope: Some(scope.to_owned()),
                ttl: request.ttl,
            },
            max_age: request.max_age,
        }
    }

    pub fn with_redis_flush_size(self, flush_size: Option<usize>) -> Self {
        match self {
            Self::Redis { cache, .. } => Self::Redis {
                cache,
                buffer: flush_size.map(|size| Arc::new(WriteBuffer::new(size))),
            },
            value => value,
        }
    }

    pub fn with_scope(self, scope: String) -> Self {
        match self {
            Self::ValkeySemantic {
                cache, embedder, ..
            } => Self::ValkeySemantic {
                cache,
                embedder,
                scope,
            },
            value => value,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Memory(_) => "memory",
            Self::Redis { .. } => "redis",
            Self::S3(_) => "s3",
            Self::Gcs(_) => "gcs",
            Self::ValkeySemantic { .. } => "valkey-semantic",
            Self::RedisSemantic { .. } => "redis_semantic",
            Self::QdrantSemantic(_) => "qdrant_semantic",
            Self::Disk(_) => "disk",
            Self::AzureBlob(_) => "azure-blob",
        }
    }

    pub fn default_ttl(&self) -> Option<Duration> {
        match self {
            Self::Memory(cache) => cache.default_ttl(),
            Self::Redis { cache, .. } => cache.default_ttl(),
            Self::S3(cache) => cache.default_ttl(),
            Self::Gcs(cache) => cache.default_ttl(),
            Self::ValkeySemantic { cache, .. } => cache.default_ttl(),
            Self::RedisSemantic { cache, .. } => cache.default_ttl(),
            Self::QdrantSemantic(_) => None,
            Self::Disk(cache) => cache.default_ttl(),
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
            Self::Memory(_)
            | Self::S3(_)
            | Self::ValkeySemantic { .. }
            | Self::RedisSemantic { .. }
            | Self::QdrantSemantic(_)
            | Self::Disk(_)
            | Self::AzureBlob(_)
            | Self::Gcs(_) => None,
            Self::Redis { cache, .. } => cache.backend().namespace(),
        }
    }

    pub fn topology(&self) -> Option<&RedisTopology> {
        match self {
            Self::Memory(_)
            | Self::S3(_)
            | Self::ValkeySemantic { .. }
            | Self::RedisSemantic { .. }
            | Self::QdrantSemantic(_)
            | Self::Disk(_)
            | Self::AzureBlob(_)
            | Self::Gcs(_) => None,
            Self::Redis { cache, .. } => Some(cache.backend().topology()),
        }
    }

    pub fn capacity(&self) -> Option<usize> {
        match self {
            Self::Memory(cache) => Some(cache.backend().max_size_in_memory()),
            Self::Redis { .. }
            | Self::S3(_)
            | Self::ValkeySemantic { .. }
            | Self::RedisSemantic { .. }
            | Self::QdrantSemantic(_)
            | Self::Disk(_)
            | Self::AzureBlob(_)
            | Self::Gcs(_) => None,
        }
    }

    pub fn max_entry_bytes(&self) -> Option<usize> {
        match self {
            Self::Memory(cache) => cache.backend().max_entry_bytes(),
            Self::Redis { .. }
            | Self::S3(_)
            | Self::ValkeySemantic { .. }
            | Self::RedisSemantic { .. }
            | Self::QdrantSemantic(_)
            | Self::Disk(_)
            | Self::AzureBlob(_)
            | Self::Gcs(_) => None,
        }
    }

    pub fn directory(&self) -> Option<&Path> {
        match self {
            Self::Disk(cache) => Some(cache.backend().directory()),
            Self::Memory(_)
            | Self::Redis { .. }
            | Self::S3(_)
            | Self::ValkeySemantic { .. }
            | Self::RedisSemantic { .. }
            | Self::QdrantSemantic(_)
            | Self::AzureBlob(_)
            | Self::Gcs(_) => None,
        }
    }

    pub fn semantic_config(&self) -> Option<(f64, &str)> {
        match self {
            Self::ValkeySemantic { cache, .. } => Some((
                cache.backend().similarity_threshold(),
                cache.backend().index_name(),
            )),
            Self::RedisSemantic { cache, .. } => Some((
                f64::from(cache.backend().similarity_threshold()),
                cache.backend().index_name(),
            )),
            _ => None,
        }
    }

    pub fn index_name(&self) -> Option<&str> {
        match self {
            Self::RedisSemantic { cache, .. } => Some(cache.backend().index_name()),
            _ => None,
        }
    }

    pub fn similarity_threshold(&self) -> Option<f64> {
        match self {
            Self::RedisSemantic { cache, .. } => {
                Some(f64::from(cache.backend().similarity_threshold()))
            }
            Self::QdrantSemantic(cache) => Some(cache.backend().similarity_threshold()),
            _ => None,
        }
    }

    pub fn collection_name(&self) -> Option<&str> {
        match self {
            Self::QdrantSemantic(cache) => Some(cache.backend().collection_name()),
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

    pub fn semantic_embedder(&self) -> Option<&PythonEmbedder> {
        match self {
            Self::RedisSemantic { embedder, .. } => Some(embedder),
            _ => None,
        }
    }

    pub fn embedder_object(&self) -> Option<&Py<PyAny>> {
        match self {
            Self::RedisSemantic { embedder, .. } => Some(embedder.object()),
            _ => None,
        }
    }

    pub fn lookup(&self, request: &NativeRequest, now: Duration) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.lookup(&Self::exact(request), now),
            Self::Redis { cache, .. } => cache.lookup(&Self::exact(request), now),
            Self::S3(cache) => cache.lookup(&Self::exact(request), now),
            Self::ValkeySemantic { cache, scope, .. } => {
                cache.lookup(&Self::semantic(request, scope), now)
            }
            Self::RedisSemantic { cache, .. } => {
                cache.lookup(&Self::semantic_request(request), now)
            }
            Self::Gcs(cache) => cache.lookup(&Self::exact(request), now),
            Self::QdrantSemantic(cache) => cache.lookup(&Self::semantic_request(request), now),
            Self::Disk(cache) => cache.lookup(&Self::exact(request), now),
            Self::AzureBlob(cache) => cache.lookup(&Self::exact(request), now),
        }
    }

    pub fn store(
        &self,
        request: &NativeRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => cache.store(&Self::exact(request), response, now),
            Self::Redis { cache, .. } => cache.store(&Self::exact(request), response, now),
            Self::S3(cache) => cache.store(&Self::exact(request), response, now),
            Self::ValkeySemantic { cache, scope, .. } => {
                cache.store(&Self::semantic(request, scope), response, now)
            }
            Self::RedisSemantic { cache, .. } => {
                cache.store(&Self::semantic_request(request), response, now)
            }
            Self::Gcs(cache) => cache.store(&Self::exact(request), response, now),
            Self::QdrantSemantic(cache) => {
                cache.store(&Self::semantic_request(request), response, now)
            }
            Self::Disk(cache) => cache.store(&Self::exact(request), response, now),
            Self::AzureBlob(cache) => cache.store(&Self::exact(request), response, now),
        }
    }

    pub fn lookup_batch(
        &self,
        requests: &[NativeRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Memory(cache) => {
                let requests = requests.iter().map(Self::exact).collect::<Vec<_>>();
                cache.lookup_batch(&requests, now)
            }
            Self::Redis { cache, .. } => {
                let requests = requests.iter().map(Self::exact).collect::<Vec<_>>();
                cache.lookup_batch(&requests, now)
            }
            Self::S3(cache) => {
                cache.lookup_batch(&requests.iter().map(Self::exact).collect::<Vec<_>>(), now)
            }
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } | Self::QdrantSemantic(_) => {
                Err(Error::UnsupportedOperation)
            }
            Self::Gcs(cache) => {
                cache.lookup_batch(&requests.iter().map(Self::exact).collect::<Vec<_>>(), now)
            }
            Self::Disk(cache) => {
                cache.lookup_batch(&requests.iter().map(Self::exact).collect::<Vec<_>>(), now)
            }
            Self::AzureBlob(cache) => {
                cache.lookup_batch(&requests.iter().map(Self::exact).collect::<Vec<_>>(), now)
            }
        }
    }

    pub async fn async_lookup(
        &self,
        request: &NativeRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        match self {
            Self::Memory(cache) => cache.async_lookup(&Self::exact(request), now).await,
            Self::Redis { cache, .. } => cache.async_lookup(&Self::exact(request), now).await,
            Self::S3(cache) => cache.async_lookup(&Self::exact(request), now).await,
            Self::ValkeySemantic { cache, scope, .. } => {
                cache
                    .async_lookup(&Self::semantic(request, scope), now)
                    .await
            }
            Self::RedisSemantic { cache, .. } => {
                cache
                    .async_lookup(&Self::semantic_request(request), now)
                    .await
            }
            Self::QdrantSemantic(cache) => {
                cache
                    .async_lookup(&Self::semantic_request(request), now)
                    .await
            }
            Self::Gcs(cache) => cache.async_lookup(&Self::exact(request), now).await,
            Self::Disk(cache) => cache.async_lookup(&Self::exact(request), now).await,
            Self::AzureBlob(cache) => cache.async_lookup(&Self::exact(request), now).await,
        }
    }

    pub(super) fn async_lookup_py<'py>(
        &self,
        py: Python<'py>,
        request: NativeRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Memory(_)
            | Self::Redis { .. }
            | Self::S3(_)
            | Self::Disk(_)
            | Self::AzureBlob(_)
            | Self::Gcs(_) => {
                let service = self.clone();
                litellm_host_python::run_async(
                    py,
                    async move { service.async_lookup(&request, super::request::now()).await },
                    super::cache_error,
                )
            }
            Self::ValkeySemantic {
                cache,
                embedder,
                scope,
            } => drive_semantic(
                py,
                SemanticEmbedExecution::lookup(
                    Arc::clone(cache.backend_arc()),
                    embedder.clone(),
                    Self::semantic(&request, scope),
                ),
            ),
            Self::RedisSemantic { .. } => drive(
                py,
                SemanticBody::new(self.clone(), SemanticOperation::Lookup(request)),
            ),
            Self::QdrantSemantic(_) => {
                let service = self.clone();
                litellm_host_python::run_async(
                    py,
                    async move { service.async_lookup(&request, super::request::now()).await },
                    super::cache_error,
                )
            }
        }
    }

    pub async fn async_store(
        &self,
        request: &NativeRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => {
                cache
                    .async_store(&Self::exact(request), response, now)
                    .await
            }
            Self::Redis {
                cache,
                buffer: None,
            } => {
                cache
                    .async_store(&Self::exact(request), response, now)
                    .await
            }
            Self::Redis {
                cache,
                buffer: Some(buffer),
            } => {
                buffer
                    .async_store(cache, &Self::exact(request), response, now)
                    .await
            }
            Self::S3(cache) => {
                cache
                    .async_store(&Self::exact(request), response, now)
                    .await
            }
            Self::ValkeySemantic { cache, scope, .. } => {
                cache
                    .async_store(&Self::semantic(request, scope), response, now)
                    .await
            }
            Self::RedisSemantic { cache, .. } => {
                cache
                    .async_store(&Self::semantic_request(request), response, now)
                    .await
            }
            Self::QdrantSemantic(cache) => {
                cache
                    .async_store(&Self::semantic_request(request), response, now)
                    .await
            }
            Self::Gcs(cache) => {
                cache
                    .async_store(&Self::exact(request), response, now)
                    .await
            }
            Self::Disk(cache) => {
                cache
                    .async_store(&Self::exact(request), response, now)
                    .await
            }
            Self::AzureBlob(cache) => {
                cache
                    .async_store(&Self::exact(request), response, now)
                    .await
            }
        }
    }

    pub(super) fn async_store_py<'py>(
        &self,
        py: Python<'py>,
        request: NativeRequest,
        response: Value,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Memory(_)
            | Self::Redis { .. }
            | Self::S3(_)
            | Self::Disk(_)
            | Self::AzureBlob(_)
            | Self::Gcs(_) => {
                let service = self.clone();
                litellm_host_python::run_async(
                    py,
                    async move {
                        service
                            .async_store(&request, response, super::request::now())
                            .await
                    },
                    super::cache_error,
                )
            }
            Self::ValkeySemantic {
                cache,
                embedder,
                scope,
            } => drive_semantic(
                py,
                SemanticEmbedExecution::store(
                    Arc::clone(cache.backend_arc()),
                    embedder.clone(),
                    Self::semantic(&request, scope),
                    response,
                ),
            ),
            Self::RedisSemantic { .. } => drive(
                py,
                SemanticBody::new(self.clone(), SemanticOperation::Store(request, response)),
            ),
            Self::QdrantSemantic(_) => {
                let service = self.clone();
                litellm_host_python::run_async(
                    py,
                    async move {
                        service
                            .async_store(&request, response, super::request::now())
                            .await
                    },
                    super::cache_error,
                )
            }
        }
    }

    pub async fn async_lookup_batch(
        &self,
        requests: &[NativeRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Memory(cache) => {
                let requests = requests.iter().map(Self::exact).collect::<Vec<_>>();
                cache.async_lookup_batch(&requests, now).await
            }
            Self::Redis { cache, .. } => {
                let requests = requests.iter().map(Self::exact).collect::<Vec<_>>();
                cache.async_lookup_batch(&requests, now).await
            }
            Self::S3(cache) => {
                cache
                    .async_lookup_batch(&requests.iter().map(Self::exact).collect::<Vec<_>>(), now)
                    .await
            }
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } | Self::QdrantSemantic(_) => {
                Err(Error::UnsupportedOperation)
            }
            Self::Gcs(cache) => {
                cache
                    .async_lookup_batch(&requests.iter().map(Self::exact).collect::<Vec<_>>(), now)
                    .await
            }
            Self::Disk(cache) => {
                cache
                    .async_lookup_batch(&requests.iter().map(Self::exact).collect::<Vec<_>>(), now)
                    .await
            }
            Self::AzureBlob(cache) => {
                cache
                    .async_lookup_batch(&requests.iter().map(Self::exact).collect::<Vec<_>>(), now)
                    .await
            }
        }
    }

    pub async fn async_store_batch(
        &self,
        entries: Vec<(NativeRequest, Value)>,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Memory(cache) => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (Self::exact(&request), value))
                    .collect();
                cache.async_store_batch(entries, now).await
            }
            Self::Redis { cache, .. } => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (Self::exact(&request), value))
                    .collect();
                cache.async_store_batch(entries, now).await
            }
            Self::S3(cache) => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (Self::exact(&request), value))
                    .collect();
                cache.async_store_batch(entries, now).await
            }
            Self::ValkeySemantic { cache, scope, .. } => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (Self::semantic(&request, scope), value))
                    .collect();
                cache.async_store_batch(entries, now).await
            }
            Self::RedisSemantic { .. } => Err(Error::UnsupportedOperation),
            Self::QdrantSemantic(cache) => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (Self::semantic_request(&request), value))
                    .collect();
                cache.async_store_batch(entries, now).await
            }
            Self::Gcs(cache) => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (Self::exact(&request), value))
                    .collect();
                cache.async_store_batch(entries, now).await
            }
            Self::Disk(cache) => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (Self::exact(&request), value))
                    .collect();
                cache.async_store_batch(entries, now).await
            }
            Self::AzureBlob(cache) => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (Self::exact(&request), value))
                    .collect();
                cache.async_store_batch(entries, now).await
            }
        }
    }

    pub(super) fn async_store_batch_py<'py>(
        &self,
        py: Python<'py>,
        entries: Vec<(NativeRequest, Value)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Memory(_)
            | Self::Redis { .. }
            | Self::S3(_)
            | Self::Disk(_)
            | Self::AzureBlob(_)
            | Self::Gcs(_) => {
                let service = self.clone();
                litellm_host_python::run_async(
                    py,
                    async move {
                        service
                            .async_store_batch(entries, super::request::now())
                            .await
                    },
                    super::cache_error,
                )
            }
            Self::ValkeySemantic {
                cache,
                embedder,
                scope,
            } => {
                let (requests, responses): (Vec<_>, Vec<_>) = entries
                    .into_iter()
                    .map(|(request, response)| (Self::semantic(&request, scope), response))
                    .unzip();
                drive_semantic(
                    py,
                    SemanticEmbedExecution::store_batch(
                        Arc::clone(cache.backend_arc()),
                        embedder.clone(),
                        requests,
                        responses,
                    ),
                )
            }
            Self::RedisSemantic { .. } => drive(
                py,
                SemanticBody::new(self.clone(), SemanticOperation::StoreBatch(entries.into())),
            ),
            Self::QdrantSemantic(_) => {
                let service = self.clone();
                litellm_host_python::run_async(
                    py,
                    async move {
                        service
                            .async_store_batch(entries, super::request::now())
                            .await
                    },
                    super::cache_error,
                )
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
            Self::S3(cache) => cache.async_flush().await,
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } | Self::QdrantSemantic(_) => {
                Err(Error::UnsupportedOperation)
            }
            Self::Gcs(cache) => cache.async_flush().await,
            Self::Disk(cache) => cache.async_flush().await,
            Self::AzureBlob(cache) => cache.async_flush().await,
        }
    }

    pub async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        match self {
            Self::Memory(cache) => cache.test_connection().await,
            Self::Redis { cache, .. } => cache.test_connection().await,
            Self::S3(cache) => cache.test_connection().await,
            Self::ValkeySemantic { cache, .. } => cache.test_connection().await,
            Self::RedisSemantic { .. } | Self::QdrantSemantic(_) => {
                Err(Error::UnsupportedOperation)
            }
            Self::Gcs(cache) => cache.test_connection().await,
            Self::Disk(cache) => cache.test_connection().await,
            Self::AzureBlob(cache) => cache.test_connection().await,
        }
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::ValkeySemantic { embedder, .. } => embedder.traverse(visit)?,
            Self::RedisSemantic { embedder, .. } => embedder.traverse(visit)?,
            _ => {}
        }
        Ok(())
    }

    pub fn gcs_backend(&self) -> Option<&GcsCache<ResponseCacheCodec>> {
        match self {
            Self::Gcs(cache) => Some(cache.backend()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use litellm_cache_response::{CacheControls, CacheKeyInput, cache_key};
    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::*;

    fn native_request(key: CacheKeyInput, metadata: Value) -> NativeRequest {
        NativeRequest {
            key,
            controls: CacheControls::default(),
            ttl: None,
            max_age: None,
            messages: Some(json!([{"role": "user", "content": "prompt"}])),
            input: None,
            metadata: Some(metadata),
            litellm_metadata: None,
            litellm_params: None,
            scope: None,
        }
    }

    #[test]
    fn semantic_key_matches_python_scope_material() {
        let key = CacheKeyInput {
            fields: vec![
                CacheKeyField {
                    name: "model".to_owned(),
                    value: Some("gpt-4.1".to_owned()),
                    api_parameter: true,
                    internal_parameter: false,
                },
                CacheKeyField {
                    name: "messages".to_owned(),
                    value: Some("prompt".to_owned()),
                    api_parameter: true,
                    internal_parameter: false,
                },
            ],
            ..Default::default()
        };
        let request = native_request(
            key,
            json!({"user_api_key": "k1", "user_api_key_team_id": null}),
        );
        let expected = format!("{:x}", Sha256::digest(b"model: gpt-4.1user_api_key: k1"));
        assert_eq!(cache_key(&semantic_key(&request, "key")), expected);

        let end_user_request = native_request(
            request.key.clone(),
            json!({"user_api_key": "k1", "user_api_key_end_user_id": "u1"}),
        );
        let expected = format!(
            "{:x}",
            Sha256::digest(b"model: gpt-4.1user_api_key: k1user_api_key_end_user_id: u1")
        );
        assert_eq!(
            cache_key(&semantic_key(&end_user_request, "end_user")),
            expected
        );

        let preset_request = native_request(
            CacheKeyInput {
                preset: Some("preset-key".to_owned()),
                ..Default::default()
            },
            json!({"user_api_key": "k1"}),
        );
        assert_eq!(
            semantic_key(&preset_request, "end_user").preset.as_deref(),
            Some("preset-key")
        );
        assert!(semantic_key(&preset_request, "end_user").fields.is_empty());
    }
}
