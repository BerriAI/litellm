use std::{sync::Arc, time::Duration};

use litellm_cache::{CacheCodec, CacheConnectionResult, Error, semantic::SemanticLookup};
use litellm_cache_azure_blob::AzureBlobCache;
use litellm_cache_disk::DiskCache;
use litellm_cache_gcs::{GcsCache, GcsConfig, StaticTokenSource};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_qdrant_semantic::{OpenAiEmbedder, QdrantSemanticCache};
use litellm_cache_redis::{RedisCache, RedisTopology};
use litellm_cache_redis_semantic::{RedisSemanticCache, RedisSemanticConfig};
use litellm_cache_response::{
    ConnectionProbe, ExactResponseCache, PartialHits, ResponseCache, ResponseCacheCodec,
    WriteBuffer,
};
use litellm_cache_s3::{S3Cache, S3CacheConfig};
use litellm_cache_valkey_semantic::{ValkeySemanticCache, ValkeySemanticConfig};
use pyo3::{PyTraverseError, PyVisit, prelude::*};
use serde_json::Value;

use super::{
    config::QdrantSemanticCacheConfig,
    embedder::PythonEmbedder,
    identity::BackendIdentity,
    request::{NativeRequest, now},
    semantic::{EmbeddingFailure, SemanticExecution, SemanticOperation, drive},
};

/// What the Python embedder receives for one semantic request.
pub(super) struct EmbeddingInput {
    pub(super) prompt: String,
    pub(super) metadata: Option<Value>,
}

/// An exact-match backend behind one pointer, with the identity its facade must reproduce.
pub(super) struct ExactService {
    cache: Arc<dyn ExactResponseCache>,
    probe: Option<Arc<dyn ConnectionProbe>>,
    buffer: Option<WriteBuffer>,
    identity: BackendIdentity,
}

#[derive(Clone)]
pub(super) enum NativeResponseCache {
    Exact(Arc<ExactService>),
    ValkeySemantic {
        cache: Arc<ResponseCache<ValkeySemanticCache<PythonEmbedder, ResponseCacheCodec>>>,
        embedder: PythonEmbedder,
        scope: String,
    },
    RedisSemantic {
        cache: Arc<ResponseCache<RedisSemanticCache<PythonEmbedder, ResponseCacheCodec>>>,
        embedder: PythonEmbedder,
    },
    QdrantSemantic(Arc<ResponseCache<QdrantSemanticCache<OpenAiEmbedder, ResponseCacheCodec>>>),
}

impl NativeResponseCache {
    pub fn memory(capacity: usize, ttl: Duration, max_entry_bytes: usize) -> Self {
        let backend = InMemoryCache::with_clock_and_size_measurement(
            Some(capacity),
            Some(ttl),
            Some(max_entry_bytes),
            Some(Arc::new(|entry| {
                ResponseCacheCodec.encode(entry).map(|bytes| bytes.len())
            })),
            now,
        );
        let identity = BackendIdentity::Memory {
            capacity: backend.max_size_in_memory(),
            max_entry_bytes: backend.max_entry_bytes(),
            default_ttl: None,
        };
        Self::exact(ResponseCache::new(Arc::new(backend)), identity)
    }

    pub fn redis(
        url: &str,
        topology: &RedisTopology,
        ttl: Option<Duration>,
        namespace: Option<String>,
    ) -> Result<Self, Error> {
        let backend =
            RedisCache::connect(url, topology, ttl, ResponseCacheCodec)?.with_namespace(namespace);
        let identity = BackendIdentity::Redis {
            topology: backend.topology().clone(),
            namespace: backend.namespace().map(str::to_owned),
            default_ttl: None,
        };
        Ok(Self::exact_probed(
            ResponseCache::new(Arc::new(backend)),
            identity,
        ))
    }

    pub async fn s3(config: S3CacheConfig, http: reqwest::Client) -> Self {
        let runtime = tokio::runtime::Handle::current();
        let backend = S3Cache::new(config, http, ResponseCacheCodec, runtime);
        let identity = BackendIdentity::S3 {
            bucket: backend.bucket().to_owned(),
            key_prefix: backend.key_prefix().to_owned(),
            region: backend.region().to_owned(),
            endpoint: backend.endpoint().map(str::to_owned),
        };
        Self::exact(ResponseCache::new(Arc::new(backend)), identity)
    }

    pub fn disk(directory: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let backend = DiskCache::open(directory, ResponseCacheCodec)?;
        let identity = BackendIdentity::Disk {
            directory: backend.directory().to_path_buf(),
        };
        Ok(Self::exact(ResponseCache::new(Arc::new(backend)), identity))
    }

    pub fn gcs(config: GcsConfig, client: reqwest::Client, token: Option<String>) -> Self {
        let backend = match token {
            Some(token) => GcsCache::with_token_source(
                config,
                client,
                ResponseCacheCodec,
                Arc::new(StaticTokenSource(token)),
            ),
            None => GcsCache::new(config, client, ResponseCacheCodec),
        };
        let identity = BackendIdentity::Gcs {
            bucket_name: backend.bucket_name().to_owned(),
            key_prefix: backend.key_prefix().to_owned(),
            path_service_account: backend.path_service_account().map(str::to_owned),
        };
        Self::exact(ResponseCache::new(Arc::new(backend)), identity)
    }

    pub async fn azure_blob(
        account_url: &str,
        container: &str,
        http: reqwest::Client,
    ) -> Result<Self, Error> {
        let backend = AzureBlobCache::connect(
            account_url,
            container,
            http,
            ResponseCacheCodec,
            tokio::runtime::Handle::current(),
        )
        .await?;
        let identity = BackendIdentity::AzureBlob {
            account_url: backend.account_url().to_owned(),
            container: backend.container_name().to_owned(),
        };
        Ok(Self::exact(ResponseCache::new(Arc::new(backend)), identity))
    }

    /// Wraps a built exact backend; the TTL a facade must match comes from the built cache.
    fn exact<B>(cache: ResponseCache<B>, identity: BackendIdentity) -> Self
    where
        ResponseCache<B>: ExactResponseCache + 'static,
        B: litellm_cache::BaseCache<Value = litellm_cache_response::CacheEntry>,
        B::Context: Default + PartialEq,
    {
        Self::exact_service(Arc::new(cache), None, identity)
    }

    /// Wraps an exact backend whose Python class defines `test_connection`.
    fn exact_probed<B>(cache: ResponseCache<B>, identity: BackendIdentity) -> Self
    where
        ResponseCache<B>: ExactResponseCache + ConnectionProbe + 'static,
        B: litellm_cache::BaseCache<Value = litellm_cache_response::CacheEntry>,
        B::Context: Default + PartialEq,
    {
        let cache = Arc::new(cache);
        Self::exact_service(cache.clone(), Some(cache), identity)
    }

    fn exact_service(
        cache: Arc<dyn ExactResponseCache>,
        probe: Option<Arc<dyn ConnectionProbe>>,
        identity: BackendIdentity,
    ) -> Self {
        let default_ttl = cache.default_ttl();
        let identity = match identity {
            BackendIdentity::Memory {
                capacity,
                max_entry_bytes,
                ..
            } => BackendIdentity::Memory {
                capacity,
                max_entry_bytes,
                default_ttl,
            },
            BackendIdentity::Redis {
                topology,
                namespace,
                ..
            } => BackendIdentity::Redis {
                topology,
                namespace,
                default_ttl,
            },
            other => other,
        };
        Self::Exact(Arc::new(ExactService {
            cache,
            probe,
            buffer: None,
            identity,
        }))
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
        let backend = RedisSemanticCache::new(url, embedder.clone(), ResponseCacheCodec, config)?;
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

    pub fn identity(&self) -> BackendIdentity {
        match self {
            Self::Exact(service) => service.identity.clone(),
            Self::ValkeySemantic { cache, .. } => BackendIdentity::ValkeySemantic {
                index_name: cache.backend().index_name().to_owned(),
                similarity_threshold: cache.backend().similarity_threshold(),
            },
            Self::RedisSemantic { cache, .. } => BackendIdentity::RedisSemantic {
                index_name: cache.backend().index_name().to_owned(),
                similarity_threshold: cache.backend().similarity_threshold(),
            },
            Self::QdrantSemantic(cache) => BackendIdentity::QdrantSemantic {
                collection_name: cache.backend().collection_name().to_owned(),
                similarity_threshold: cache.backend().similarity_threshold(),
                vector_size: cache.backend().vector_size(),
                embedding_model: cache.backend().embedder().model().to_owned(),
            },
        }
    }

    pub fn kind(&self) -> &'static str {
        self.identity().kind()
    }

    pub fn with_redis_flush_size(self, flush_size: Option<usize>) -> Self {
        match self {
            Self::Exact(service) if matches!(service.identity, BackendIdentity::Redis { .. }) => {
                Self::Exact(Arc::new(ExactService {
                    cache: Arc::clone(&service.cache),
                    probe: service.probe.clone(),
                    buffer: flush_size.map(WriteBuffer::new),
                    identity: service.identity.clone(),
                }))
            }
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

    pub fn embedder_object(&self) -> Option<&Py<PyAny>> {
        match self {
            Self::RedisSemantic { embedder, .. } => Some(embedder.object()),
            _ => None,
        }
    }

    /// The prompt and metadata this backend would embed for `request`, if it has a prompt.
    pub(super) fn embedding_input(&self, request: &NativeRequest) -> Option<EmbeddingInput> {
        let context = match self {
            Self::ValkeySemantic { scope, .. } => request.scoped_semantic(scope).context,
            Self::RedisSemantic { .. } => request.semantic().context,
            Self::Exact(_) | Self::QdrantSemantic(_) => return None,
        };
        let prompt = litellm_cache::semantic::prompt_from_context(&context)?;
        Some(EmbeddingInput {
            prompt,
            metadata: context.metadata,
        })
    }

    /// Drives a semantic operation whose embedding comes from Python.
    fn python_semantic<'py>(
        &self,
        py: Python<'py>,
        operation: SemanticOperation,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (embedder, failure) = match self {
            Self::ValkeySemantic { embedder, .. } => (embedder, EmbeddingFailure::Propagate),
            Self::RedisSemantic { embedder, .. } => (embedder, EmbeddingFailure::Unavailable),
            Self::Exact(_) | Self::QdrantSemantic(_) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "semantic execution requires a Python-embedded backend",
                ));
            }
        };
        drive(
            py,
            SemanticExecution::new(self.clone(), embedder.clone(), failure, operation),
        )
    }

    pub fn lookup(&self, request: &NativeRequest, now: Duration) -> Result<Option<Value>, Error> {
        match self {
            Self::Exact(service) => service.cache.lookup(&request.exact(), now),
            Self::ValkeySemantic { cache, scope, .. } => {
                cache.lookup(&request.scoped_semantic(scope), now)
            }
            Self::RedisSemantic { cache, .. } => cache.lookup(&request.semantic(), now),
            Self::QdrantSemantic(cache) => cache.lookup(&request.semantic(), now),
        }
    }

    /// `lookup` plus the similarity Python's semantic backend writes to the request metadata.
    /// Exact backends report none.
    pub fn lookup_semantic(
        &self,
        request: &NativeRequest,
        now: Duration,
    ) -> Result<SemanticLookup<Value>, Error> {
        match self {
            Self::Exact(service) => service
                .cache
                .lookup(&request.exact(), now)
                .map(exact_lookup),
            Self::ValkeySemantic { cache, scope, .. } => {
                redis_family(cache.lookup_semantic(&request.scoped_semantic(scope), now))
            }
            Self::RedisSemantic { cache, .. } => {
                redis_family(cache.lookup_semantic(&request.semantic(), now))
            }
            Self::QdrantSemantic(cache) => cache.lookup_semantic(&request.semantic(), now),
        }
    }

    pub fn store(
        &self,
        request: &NativeRequest,
        response: Value,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Exact(service) => service.cache.store(&request.exact(), response, now),
            Self::ValkeySemantic { cache, scope, .. } => {
                cache.store(&request.scoped_semantic(scope), response, now)
            }
            Self::RedisSemantic { cache, .. } => cache.store(&request.semantic(), response, now),
            Self::QdrantSemantic(cache) => cache.store(&request.semantic(), response, now),
        }
    }

    pub fn lookup_batch(
        &self,
        requests: &[NativeRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Exact(service) => service.cache.lookup_batch(&exact_requests(requests), now),
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } | Self::QdrantSemantic(_) => {
                Err(Error::UnsupportedOperation)
            }
        }
    }

    pub async fn async_lookup(
        &self,
        request: &NativeRequest,
        now: Duration,
    ) -> Result<Option<Value>, Error> {
        match self {
            Self::Exact(service) => service.cache.async_lookup(&request.exact(), now).await,
            Self::ValkeySemantic { cache, scope, .. } => {
                cache
                    .async_lookup(&request.scoped_semantic(scope), now)
                    .await
            }
            Self::RedisSemantic { cache, .. } => cache.async_lookup(&request.semantic(), now).await,
            Self::QdrantSemantic(cache) => cache.async_lookup(&request.semantic(), now).await,
        }
    }

    pub async fn async_lookup_semantic(
        &self,
        request: &NativeRequest,
        now: Duration,
    ) -> Result<SemanticLookup<Value>, Error> {
        match self {
            Self::Exact(service) => service
                .cache
                .async_lookup(&request.exact(), now)
                .await
                .map(exact_lookup),
            Self::ValkeySemantic { cache, scope, .. } => redis_family(
                cache
                    .async_lookup_semantic(&request.scoped_semantic(scope), now)
                    .await,
            ),
            Self::RedisSemantic { cache, .. } => {
                redis_family(cache.async_lookup_semantic(&request.semantic(), now).await)
            }
            Self::QdrantSemantic(cache) => {
                cache.async_lookup_semantic(&request.semantic(), now).await
            }
        }
    }

    pub(super) fn async_lookup_semantic_py<'py>(
        &self,
        py: Python<'py>,
        request: NativeRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Exact(_) | Self::QdrantSemantic(_) => {
                let service = self.clone();
                crate::logger::run_async(
                    py,
                    async move {
                        service
                            .async_lookup_semantic(&request, now())
                            .await
                            .map(SemanticReply::from)
                    },
                    super::cache_error,
                )
            }
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } => {
                self.python_semantic(py, SemanticOperation::LookupSemantic(request))
            }
        }
    }

    pub(super) fn async_lookup_py<'py>(
        &self,
        py: Python<'py>,
        request: NativeRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Exact(_) | Self::QdrantSemantic(_) => {
                let service = self.clone();
                crate::logger::run_async(
                    py,
                    async move { service.async_lookup(&request, now()).await },
                    super::cache_error,
                )
            }
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } => {
                self.python_semantic(py, SemanticOperation::Lookup(request))
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
            Self::Exact(service) => match &service.buffer {
                None => {
                    service
                        .cache
                        .async_store(&request.exact(), response, now)
                        .await
                }
                Some(buffer) => {
                    buffer
                        .async_store(service.cache.as_ref(), &request.exact(), response, now)
                        .await
                }
            },
            Self::ValkeySemantic { cache, scope, .. } => {
                cache
                    .async_store(&request.scoped_semantic(scope), response, now)
                    .await
            }
            Self::RedisSemantic { cache, .. } => {
                cache.async_store(&request.semantic(), response, now).await
            }
            Self::QdrantSemantic(cache) => {
                cache.async_store(&request.semantic(), response, now).await
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
            Self::Exact(_) | Self::QdrantSemantic(_) => {
                let service = self.clone();
                crate::logger::run_async(
                    py,
                    async move { service.async_store(&request, response, now()).await },
                    super::cache_error,
                )
            }
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } => {
                self.python_semantic(py, SemanticOperation::Store(request, response))
            }
        }
    }

    pub async fn async_lookup_batch(
        &self,
        requests: &[NativeRequest],
        now: Duration,
    ) -> Result<PartialHits, Error> {
        match self {
            Self::Exact(service) => {
                service
                    .cache
                    .async_lookup_batch(&exact_requests(requests), now)
                    .await
            }
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } | Self::QdrantSemantic(_) => {
                Err(Error::UnsupportedOperation)
            }
        }
    }

    pub async fn async_store_batch(
        &self,
        entries: Vec<(NativeRequest, Value)>,
        now: Duration,
    ) -> Result<(), Error> {
        match self {
            Self::Exact(service) => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (request.exact(), value))
                    .collect();
                service.cache.async_store_batch(entries, now).await
            }
            Self::ValkeySemantic { cache, scope, .. } => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (request.scoped_semantic(scope), value))
                    .collect();
                cache.async_store_batch(entries, now).await
            }
            Self::RedisSemantic { .. } => Err(Error::UnsupportedOperation),
            Self::QdrantSemantic(cache) => {
                let entries = entries
                    .into_iter()
                    .map(|(request, value)| (request.semantic(), value))
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
            Self::Exact(_) | Self::QdrantSemantic(_) => {
                let service = self.clone();
                crate::logger::run_async(
                    py,
                    async move { service.async_store_batch(entries, now()).await },
                    super::cache_error,
                )
            }
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } => {
                self.python_semantic(py, SemanticOperation::StoreBatch(entries.into()))
            }
        }
    }

    pub async fn async_flush(&self) -> Result<(), Error> {
        match self {
            Self::Exact(service) => {
                if let Some(buffer) = &service.buffer {
                    buffer.clear()?;
                }
                service.cache.async_flush().await
            }
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } | Self::QdrantSemantic(_) => {
                Err(Error::UnsupportedOperation)
            }
        }
    }

    pub async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        match self {
            Self::Exact(service) => match &service.probe {
                Some(probe) => probe.test_connection().await,
                None => Err(Error::UnsupportedOperation),
            },
            Self::ValkeySemantic { .. } | Self::RedisSemantic { .. } | Self::QdrantSemantic(_) => {
                Err(Error::UnsupportedOperation)
            }
        }
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::ValkeySemantic { embedder, .. } | Self::RedisSemantic { embedder, .. } => {
                embedder.traverse(visit)
            }
            Self::Exact(_) | Self::QdrantSemantic(_) => Ok(()),
        }
    }
}

fn exact_requests(requests: &[NativeRequest]) -> Vec<litellm_cache_response::ResponseCacheRequest> {
    requests.iter().map(NativeRequest::exact).collect()
}

/// What `lookup_semantic` hands Python: the response and the similarity to stamp, if any.
#[derive(serde::Serialize)]
pub(super) struct SemanticReply(pub(super) Option<Value>, pub(super) Option<f64>);

impl From<SemanticLookup<Value>> for SemanticReply {
    fn from(lookup: SemanticLookup<Value>) -> Self {
        Self(lookup.value, lookup.similarity)
    }
}

fn exact_lookup(value: Option<Value>) -> SemanticLookup<Value> {
    SemanticLookup {
        value,
        similarity: None,
    }
}

/// Python's Redis and Valkey semantic caches catch every lookup failure and stamp `0.0`.
fn redis_family(
    lookup: Result<SemanticLookup<Value>, Error>,
) -> Result<SemanticLookup<Value>, Error> {
    Ok(lookup.unwrap_or_else(|_| SemanticLookup::miss(Some(0.0))))
}
