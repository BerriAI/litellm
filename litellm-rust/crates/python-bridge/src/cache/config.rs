use std::{path::PathBuf, time::Duration};

use litellm_auth_aws::AwsAuthConfig;
use litellm_cache::CacheType;
use litellm_cache_qdrant_semantic::{OpenAiEmbedderConfig, QdrantSemanticConfig, Quantization};
use litellm_cache_redis::{RedisNode, RedisTopology};
use litellm_cache_s3::{S3CacheConfig, S3Endpoint};
use pyo3::{
    exceptions::{PyAttributeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyDict, PyList, PyString},
};

use super::{identity::BackendIdentity, native::NativeResponseCache, request::duration};

pub(super) struct CachePolicy {
    pub(super) redis_flush_size: Option<usize>,
    pub(super) semantic_cache_scope: String,
}

pub(super) struct MemoryCacheConfig {
    pub(super) default_ttl: Duration,
    pub(super) capacity: usize,
    pub(super) max_entry_bytes: usize,
}

pub(super) struct DiskCacheConfig {
    pub(super) directory: PathBuf,
}

#[derive(Debug, PartialEq)]
pub(super) enum RedisProtocol {
    Resp2,
    Resp3,
}

#[derive(Debug, PartialEq)]
pub(super) enum CertificateRequirement {
    None,
    Optional,
    Required,
}

pub(super) struct RedisTlsConfig {
    pub(super) certificate_requirement: CertificateRequirement,
    pub(super) check_hostname: bool,
    pub(super) ca_certificate: Option<String>,
    pub(super) ca_data: Option<String>,
    pub(super) client_certificate: Option<String>,
    pub(super) client_key: Option<String>,
}

pub(super) struct RedisConnectionConfig {
    pub(super) host: String,
    pub(super) port: u16,
    pub(super) database: i64,
    pub(super) username: Option<String>,
    pub(super) password: Option<String>,
    pub(super) protocol: RedisProtocol,
    pub(super) pool_size: usize,
    pub(super) read_timeout: Option<Duration>,
    pub(super) connect_timeout: Option<Duration>,
    pub(super) socket_keepalive: Option<bool>,
    pub(super) health_check_interval: Duration,
    pub(super) client_name: Option<String>,
    pub(super) tls: Option<RedisTlsConfig>,
}

pub(super) struct RedisCacheConfig {
    pub(super) default_ttl: Duration,
    pub(super) namespace: Option<String>,
    pub(super) flush_size: usize,
    pub(super) topology: RedisTopology,
    pub(super) connection: RedisConnectionConfig,
}

#[derive(Debug, PartialEq)]
pub(super) struct GcsCacheConfig {
    pub(super) bucket_name: String,
    pub(super) key_prefix: String,
    pub(super) path_service_account: Option<String>,
}

pub(super) struct AzureBlobCacheConfig {
    pub(super) account_url: String,
    pub(super) container: String,
}

pub(super) struct RedisSemanticCacheConfig {
    pub(super) redis_url: String,
    pub(super) index_name: String,
    pub(super) similarity_threshold: f64,
}

struct RedisClientProjection<'py> {
    topology: RedisTopology,
    host: String,
    port: u16,
    pool_size: usize,
    resolved: Bound<'py, PyDict>,
    tls: Option<RedisTlsConfig>,
}

const REDIS_PY_DEFAULT_MAX_CONNECTIONS: usize = 1 << 31;

/// The read and write timeout every native Redis connection uses, which is also `RedisCache`'s
/// default `socket_timeout`.
const NATIVE_REDIS_SOCKET_TIMEOUT: Duration = Duration::from_secs(5);

pub(super) struct ValkeySemanticCacheConfig {
    pub(super) similarity_threshold: f64,
    pub(super) index_name: String,
    pub(super) connection: RedisConnectionConfig,
}

pub(super) struct QdrantSemanticCacheConfig {
    pub(super) grpc_url: String,
    pub(super) api_key: Option<String>,
    pub(super) collection_name: String,
    pub(super) similarity_threshold: f64,
    pub(super) vector_size: u64,
    pub(super) embedding: OpenAiEmbedderConfig,
    pub(super) quantization: Quantization,
}

impl QdrantSemanticCacheConfig {
    pub(super) fn to_qdrant_config(&self) -> QdrantSemanticConfig {
        QdrantSemanticConfig {
            collection_name: self.collection_name.clone(),
            similarity_threshold: self.similarity_threshold,
            vector_size: self.vector_size,
            quantization: self.quantization.clone(),
        }
    }
}

impl RedisTlsConfig {
    /// Whether redis-rs with rustls behaves like this redis-py `SSLConnection`: it verifies the
    /// certificate chain against the system roots and always checks the hostname.
    fn native(&self) -> Result<(), UnsupportedCacheConfig> {
        if self.ca_certificate.is_some()
            || self.ca_data.is_some()
            || self.client_certificate.is_some()
            || self.client_key.is_some()
        {
            return Err(UnsupportedCacheConfig::RedisTlsCertificates);
        }
        if self.certificate_requirement == CertificateRequirement::None || !self.check_hostname {
            return Err(UnsupportedCacheConfig::RedisTlsVerification);
        }
        Ok(())
    }
}

impl RedisConnectionConfig {
    /// The redis-rs URL for this connection, or the first setting the native client cannot
    /// honor. The native pool and socket timeouts are fixed, so only redis-py's unbounded pool,
    /// its unset timeouts and `RedisCache`'s five-second `socket_timeout` map onto them.
    pub(super) fn native_url(&self) -> Result<String, UnsupportedCacheConfig> {
        if self.pool_size != REDIS_PY_DEFAULT_MAX_CONNECTIONS {
            return Err(UnsupportedCacheConfig::RedisPoolSize);
        }
        if self
            .read_timeout
            .is_some_and(|timeout| timeout != NATIVE_REDIS_SOCKET_TIMEOUT)
            || self.connect_timeout.is_some()
        {
            return Err(UnsupportedCacheConfig::RedisTimeout);
        }
        if self.socket_keepalive == Some(true) {
            return Err(UnsupportedCacheConfig::RedisKeepalive);
        }
        if !self.health_check_interval.is_zero() {
            return Err(UnsupportedCacheConfig::RedisHealthCheck);
        }
        if self.client_name.is_some() {
            return Err(UnsupportedCacheConfig::RedisClientName);
        }
        let scheme = match &self.tls {
            None => "redis",
            Some(tls) => {
                tls.native()?;
                "rediss"
            }
        };
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        let mut url = url::Url::parse(&format!(
            "{scheme}://{host}:{}/{}",
            self.port, self.database
        ))
        .map_err(|_| UnsupportedCacheConfig::RedisConnection)?;
        if let Some(username) = &self.username {
            url.set_username(username)
                .map_err(|()| UnsupportedCacheConfig::RedisConnection)?;
        }
        if let Some(password) = &self.password {
            url.set_password(Some(password))
                .map_err(|()| UnsupportedCacheConfig::RedisConnection)?;
        }
        if self.protocol == RedisProtocol::Resp3 {
            url.set_query(Some("protocol=resp3"));
        }
        Ok(url.into())
    }
}

impl RedisSemanticCacheConfig {
    /// redisvl hands `redis_url` to redis-py, which reads TLS and socket options from the URL;
    /// redis-rs ignores those, so only a plain URL keeps its meaning.
    pub(super) fn native_url(&self) -> Result<&str, UnsupportedCacheConfig> {
        let url = url::Url::parse(&self.redis_url)
            .map_err(|_| UnsupportedCacheConfig::RedisSemanticUrl)?;
        if !matches!(url.scheme(), "redis" | "unix") || url.query().is_some() {
            return Err(UnsupportedCacheConfig::RedisSemanticUrl);
        }
        Ok(&self.redis_url)
    }
}

pub(super) enum CacheBackendConfig {
    Memory(MemoryCacheConfig),
    Redis(Box<RedisCacheConfig>),
    S3(Box<S3CacheConfig>),
    Gcs(GcsCacheConfig),
    ValkeySemantic(Box<ValkeySemanticCacheConfig>),
    Disk(DiskCacheConfig),
    AzureBlob(AzureBlobCacheConfig),
    RedisSemantic(Box<RedisSemanticCacheConfig>),
    QdrantSemantic(Box<QdrantSemanticCacheConfig>),
}

pub(super) struct NativeCacheConfig {
    pub(super) policy: CachePolicy,
    pub(super) backend: CacheBackendConfig,
}

pub(super) enum UnsupportedCacheConfig {
    Backend,
    RedisTopology,
    RedisCredentials,
    RedisConnection,
    RedisOption,
    S3Client,
    S3Credentials,
    S3Option,
    GcsBucket,
    DiskStore,
    QdrantEndpoint,
    SemanticEmbedding,
    RedisPoolSize,
    RedisTimeout,
    RedisKeepalive,
    RedisHealthCheck,
    RedisClientName,
    RedisTlsCertificates,
    RedisTlsVerification,
    RedisSemanticUrl,
    ValkeyTls,
}

impl UnsupportedCacheConfig {
    pub(super) fn message(&self) -> &'static str {
        match self {
            Self::Backend => "native cache backend is not implemented",
            Self::RedisTopology => "native Redis topology is not implemented",
            Self::RedisCredentials => "native Redis credentials require Python",
            Self::RedisConnection => "native Redis connection type is not implemented",
            Self::RedisOption => "native Redis configuration requires Python",
            Self::S3Client => "native S3 client type is not implemented",
            Self::S3Credentials => "native S3 credentials require Python",
            Self::S3Option => "native S3 configuration requires Python",
            Self::GcsBucket => "native GCS cache requires a configured bucket name",
            Self::DiskStore => "native disk cache requires the built-in diskcache store",
            Self::QdrantEndpoint => {
                "native Qdrant requires the default REST port so the gRPC port can be derived"
            }
            Self::SemanticEmbedding => "native semantic embedding requires Python",
            Self::RedisPoolSize => {
                "native Redis uses a fixed connection pool; max_connections requires Python"
            }
            Self::RedisTimeout => {
                "native Redis uses fixed socket timeouts; socket_timeout and \
                 socket_connect_timeout require Python"
            }
            Self::RedisKeepalive => "native Redis does not support socket_keepalive",
            Self::RedisHealthCheck => "native Redis does not support health_check_interval",
            Self::RedisClientName => "native Redis does not support client_name",
            Self::RedisTlsCertificates => {
                "native Redis TLS does not support ssl_ca_certs, ssl_ca_data, ssl_certfile or \
                 ssl_keyfile"
            }
            Self::RedisTlsVerification => {
                "native Redis TLS always verifies the certificate and hostname; \
                 ssl_cert_reqs=none and ssl_check_hostname=false require Python"
            }
            Self::RedisSemanticUrl => {
                "native Redis semantic cache does not support TLS or query options in redis_url"
            }
            Self::ValkeyTls => "native Valkey semantic cache does not support TLS connections",
        }
    }
}

pub(super) enum CacheConfigProjection {
    Native(Box<NativeCacheConfig>),
    Unsupported(UnsupportedCacheConfig),
}

impl NativeCacheConfig {
    #[inline(never)]
    pub(super) fn project(facade: &Bound<'_, PyAny>) -> PyResult<CacheConfigProjection> {
        let backend_name = facade.getattr("type")?.extract::<String>()?;
        let policy = CachePolicy {
            redis_flush_size: facade
                .getattr("redis_flush_size")?
                .extract::<Option<usize>>()?,
            semantic_cache_scope: facade
                .getattr("semantic_cache_scope")?
                .extract::<String>()?,
        };
        let backend = facade.getattr("cache")?;
        match CacheType::from_python_name(&backend_name) {
            Some(CacheType::Local) => project_memory(&backend).map(|backend| {
                CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::Memory(backend),
                }))
            }),
            Some(CacheType::Redis) => match project_redis(&backend)? {
                Ok(backend) => Ok(CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::Redis(Box::new(backend)),
                }))),
                Err(reason) => Ok(CacheConfigProjection::Unsupported(reason)),
            },
            Some(CacheType::S3) => match project_s3(&backend)? {
                Ok(backend) => Ok(CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::S3(Box::new(backend)),
                }))),
                Err(reason) => Ok(CacheConfigProjection::Unsupported(reason)),
            },
            Some(CacheType::Gcs) => match project_gcs(&backend)? {
                Ok(backend) => Ok(CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::Gcs(backend),
                }))),
                Err(reason) => Ok(CacheConfigProjection::Unsupported(reason)),
            },
            Some(CacheType::ValkeySemantic) => match project_valkey_semantic(&backend)? {
                Ok(backend) => Ok(CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::ValkeySemantic(Box::new(backend)),
                }))),
                Err(reason) => Ok(CacheConfigProjection::Unsupported(reason)),
            },
            Some(CacheType::Disk) => match project_disk(&backend)? {
                Ok(backend) => Ok(CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::Disk(backend),
                }))),
                Err(reason) => Ok(CacheConfigProjection::Unsupported(reason)),
            },
            Some(CacheType::QdrantSemantic) => match project_qdrant_semantic(&backend)? {
                Ok(backend) => Ok(CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::QdrantSemantic(Box::new(backend)),
                }))),
                Err(reason) => Ok(CacheConfigProjection::Unsupported(reason)),
            },
            Some(CacheType::AzureBlob) => project_azure_blob(&backend).map(|backend| {
                CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::AzureBlob(backend),
                }))
            }),
            Some(CacheType::RedisSemantic) => project_redis_semantic(&backend).map(|backend| {
                CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::RedisSemantic(Box::new(backend)),
                }))
            }),
            None => Ok(CacheConfigProjection::Unsupported(
                UnsupportedCacheConfig::Backend,
            )),
        }
    }

    pub(super) fn service_mismatch(&self, service: &NativeResponseCache) -> Option<&'static str> {
        self.backend.identity().mismatch(&service.identity())
    }
}

impl CacheBackendConfig {
    /// The identity a native backend must have for this facade configuration to describe it.
    pub(super) fn identity(&self) -> BackendIdentity {
        match self {
            Self::Memory(config) => BackendIdentity::Memory {
                capacity: config.capacity,
                max_entry_bytes: Some(config.max_entry_bytes),
                default_ttl: Some(config.default_ttl),
            },
            Self::Redis(config) => BackendIdentity::Redis {
                topology: config.topology.clone(),
                namespace: config.namespace.clone(),
                default_ttl: Some(config.default_ttl),
            },
            Self::S3(config) => BackendIdentity::S3 {
                bucket: config.bucket.clone(),
                key_prefix: config.key_prefix.clone(),
                region: config.region.clone(),
                endpoint: config
                    .endpoint
                    .as_ref()
                    .map(|endpoint| endpoint.url.clone()),
            },
            Self::Gcs(config) => BackendIdentity::Gcs {
                bucket_name: config.bucket_name.clone(),
                key_prefix: config.key_prefix.clone(),
                path_service_account: config.path_service_account.clone(),
            },
            Self::ValkeySemantic(config) => BackendIdentity::ValkeySemantic {
                index_name: config.index_name.clone(),
                similarity_threshold: config.similarity_threshold,
            },
            Self::Disk(config) => BackendIdentity::Disk {
                directory: config.directory.clone(),
            },
            Self::AzureBlob(config) => BackendIdentity::AzureBlob {
                account_url: config.account_url.clone(),
                container: config.container.clone(),
            },
            Self::RedisSemantic(config) => BackendIdentity::RedisSemantic {
                index_name: config.index_name.clone(),
                similarity_threshold: config.similarity_threshold as f32,
            },
            Self::QdrantSemantic(config) => BackendIdentity::QdrantSemantic {
                collection_name: config.collection_name.clone(),
                similarity_threshold: config.similarity_threshold,
                vector_size: config.vector_size,
                embedding_model: config.embedding.model.clone(),
            },
        }
    }
}

#[inline(never)]
fn project_qdrant_semantic(
    backend: &Bound<'_, PyAny>,
) -> PyResult<Result<QdrantSemanticCacheConfig, UnsupportedCacheConfig>> {
    let rest_url = backend.getattr("qdrant_api_base")?.extract::<String>()?;
    let parsed = match url::Url::parse(&rest_url) {
        Ok(value) => value,
        Err(_) => return Ok(Err(UnsupportedCacheConfig::QdrantEndpoint)),
    };
    if !matches!(parsed.scheme(), "http" | "https")
        || (!parsed.path().is_empty() && parsed.path() != "/")
        || parsed.query().is_some()
        || parsed.host_str().is_none()
        || parsed.port() != Some(6333)
    {
        return Ok(Err(UnsupportedCacheConfig::QdrantEndpoint));
    }
    let mut grpc_url = parsed;
    if grpc_url.set_port(Some(6334)).is_err() {
        return Ok(Err(UnsupportedCacheConfig::QdrantEndpoint));
    }
    grpc_url.set_path("");
    grpc_url.set_query(None);

    if optional_attribute(backend, "embedding_max_input_tokens")?
        .is_some_and(|value| !value.is_none())
    {
        return Ok(Err(UnsupportedCacheConfig::SemanticEmbedding));
    }
    let configured_model = backend.getattr("embedding_model")?.extract::<String>()?;
    let embedding_model = configured_model
        .strip_prefix("openai/")
        .unwrap_or(&configured_model)
        .to_owned();
    if !embedding_model.starts_with("text-embedding-") {
        return Ok(Err(UnsupportedCacheConfig::SemanticEmbedding));
    }
    let proxy_server = py_sys_module(backend.py())?;
    if let Some(proxy_server) = proxy_server {
        let router = proxy_server.getattr("llm_router")?;
        let model_list = proxy_server.getattr("llm_model_list")?;
        let embedding_router = backend.py().import("litellm.caching._embedding_router")?;
        if !embedding_router
            .getattr("resolve_embedding_router")?
            .call1((configured_model.as_str(), router, model_list))?
            .is_none()
        {
            return Ok(Err(UnsupportedCacheConfig::SemanticEmbedding));
        }
    }
    let litellm = backend.py().import("litellm")?;
    for name in ["api_key", "openai_key", "api_base"] {
        if !litellm.getattr(name)?.is_none() {
            return Ok(Err(UnsupportedCacheConfig::SemanticEmbedding));
        }
    }
    let Ok(embedding_api_key) = std::env::var("OPENAI_API_KEY") else {
        return Ok(Err(UnsupportedCacheConfig::SemanticEmbedding));
    };
    if embedding_api_key.is_empty() {
        return Ok(Err(UnsupportedCacheConfig::SemanticEmbedding));
    }
    let embedding_api_base = std::env::var("OPENAI_BASE_URL")
        .or_else(|_| std::env::var("OPENAI_API_BASE"))
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());
    let timeout = optional_attribute(backend, "embedding_timeout")?
        .map(|value| value.extract::<Option<f64>>())
        .transpose()?
        .flatten()
        .map(duration)
        .transpose()?;
    Ok(Ok(QdrantSemanticCacheConfig {
        grpc_url: grpc_url.to_string().trim_end_matches('/').to_owned(),
        api_key: optional_string(backend.getattr("qdrant_api_key")?)?,
        collection_name: backend.getattr("collection_name")?.extract()?,
        similarity_threshold: backend.getattr("similarity_threshold")?.extract()?,
        vector_size: backend.getattr("vector_size")?.extract::<u64>()?,
        embedding: OpenAiEmbedderConfig {
            api_base: embedding_api_base,
            api_key: embedding_api_key,
            model: embedding_model,
            timeout,
        },
        quantization: Quantization::Binary,
    }))
}

fn py_sys_module(py: Python<'_>) -> PyResult<Option<Bound<'_, PyAny>>> {
    match py
        .import("sys")?
        .getattr("modules")?
        .get_item("litellm.proxy.proxy_server")
    {
        Ok(module) => Ok(Some(module)),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyKeyError>(py) => Ok(None),
        Err(error) => Err(error),
    }
}

#[inline(never)]
fn project_azure_blob(backend: &Bound<'_, PyAny>) -> PyResult<AzureBlobCacheConfig> {
    let client = backend.getattr("container_client")?;
    let container = client.getattr("container_name")?.extract::<String>()?;
    let url = client.getattr("url")?.extract::<String>()?;
    let account_url = url
        .strip_suffix(container.as_str())
        .and_then(|url| url.strip_suffix('/'))
        .ok_or_else(|| PyValueError::new_err("Azure Blob container URL is malformed"))?;
    Ok(AzureBlobCacheConfig {
        account_url: account_url.to_string(),
        container,
    })
}

#[inline(never)]
pub(super) fn project_redis_semantic(
    backend: &Bound<'_, PyAny>,
) -> PyResult<RedisSemanticCacheConfig> {
    Ok(RedisSemanticCacheConfig {
        redis_url: backend.getattr("_redis_url")?.extract::<String>()?,
        index_name: backend
            .getattr("_index_name")?
            .extract::<Option<String>>()?
            .unwrap_or_else(|| "litellm_semantic_cache_index".into()),
        similarity_threshold: backend.getattr("similarity_threshold")?.extract::<f64>()?,
    })
}

#[inline(never)]
fn project_memory(backend: &Bound<'_, PyAny>) -> PyResult<MemoryCacheConfig> {
    let max_size_kib = backend.getattr("max_size_per_item")?.extract::<usize>()?;
    Ok(MemoryCacheConfig {
        default_ttl: duration(backend.getattr("default_ttl")?.extract::<f64>()?)?,
        capacity: backend.getattr("max_size_in_memory")?.extract::<usize>()?,
        max_entry_bytes: max_size_kib
            .checked_mul(1024)
            .ok_or_else(|| PyValueError::new_err("memory cache item limit is too large"))?,
    })
}

#[inline(never)]
fn project_gcs(
    backend: &Bound<'_, PyAny>,
) -> PyResult<Result<GcsCacheConfig, UnsupportedCacheConfig>> {
    let bucket_name = match backend.getattr("bucket_name")?.extract::<Option<String>>() {
        Ok(Some(bucket_name)) if !bucket_name.is_empty() => bucket_name,
        _ => return Ok(Err(UnsupportedCacheConfig::GcsBucket)),
    };
    Ok(Ok(GcsCacheConfig {
        bucket_name,
        key_prefix: backend.getattr("key_prefix")?.extract::<String>()?,
        path_service_account: backend
            .getattr("path_service_account")?
            .extract::<Option<String>>()?,
    }))
}

#[inline(never)]
fn project_disk(
    backend: &Bound<'_, PyAny>,
) -> PyResult<Result<DiskCacheConfig, UnsupportedCacheConfig>> {
    let store = backend.getattr("disk_cache")?;
    if !instance_class_is(&store, "diskcache.core", "Cache")?
        || !instance_class_is(&store.getattr("_disk")?, "diskcache.core", "Disk")?
    {
        return Ok(Err(UnsupportedCacheConfig::DiskStore));
    }
    Ok(Ok(DiskCacheConfig {
        directory: PathBuf::from(store.getattr("directory")?.extract::<String>()?),
    }))
}

#[inline(never)]
fn project_redis(
    backend: &Bound<'_, PyAny>,
) -> PyResult<Result<RedisCacheConfig, UnsupportedCacheConfig>> {
    let source = backend.getattr("redis_kwargs")?.cast_into::<PyDict>()?;
    if has_value(&source, "sentinel_nodes")? {
        return Ok(Err(UnsupportedCacheConfig::RedisTopology));
    }
    for key in ["credential_provider", "redis_connect_func"] {
        if has_value(&source, key)? {
            return Ok(Err(UnsupportedCacheConfig::RedisCredentials));
        }
    }
    if has_value(&source, "connection_pool")? {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    }
    for key in [
        "retry",
        "retry_on_error",
        "socket_keepalive_options",
        "unix_socket_path",
        "cache",
        "cache_config",
        "event_dispatcher",
        "ssl_ca_path",
        "ssl_password",
        "ssl_min_version",
        "ssl_ciphers",
        "ssl_validate_ocsp",
        "ssl_validate_ocsp_stapled",
        "ssl_ocsp_context",
        "ssl_ocsp_expected_cert",
    ] {
        if has_value(&source, key)? {
            return Ok(Err(UnsupportedCacheConfig::RedisOption));
        }
    }
    for key in ["retry_on_timeout", "single_connection_client"] {
        if optional_coerced_bool(&source, key)?.unwrap_or(false) {
            return Ok(Err(UnsupportedCacheConfig::RedisOption));
        }
    }

    let client = backend.getattr("redis_client")?;
    let projection = if has_value(&source, "startup_nodes")? {
        project_cluster_client(&source, &client)?
    } else {
        project_standalone_client(&client)?
    };
    let RedisClientProjection {
        topology,
        host,
        port,
        pool_size,
        resolved,
        tls,
    } = match projection {
        Ok(projection) => projection,
        Err(reason) => return Ok(Err(reason)),
    };
    if has_value(&resolved, "credential_provider")? {
        return Ok(Err(UnsupportedCacheConfig::RedisCredentials));
    }
    Ok(Ok(RedisCacheConfig {
        default_ttl: duration(backend.getattr("default_ttl")?.extract::<f64>()?)?,
        namespace: optional_attribute_string(backend, "namespace")?,
        flush_size: backend.getattr("redis_flush_size")?.extract::<usize>()?,
        topology,
        connection: resolved_connection(&resolved, host, port, pool_size, tls)?,
    }))
}

/// The connection settings redis-py resolved for one client's pool.
#[inline(never)]
fn resolved_connection(
    resolved: &Bound<'_, PyDict>,
    host: String,
    port: u16,
    pool_size: usize,
    tls: Option<RedisTlsConfig>,
) -> PyResult<RedisConnectionConfig> {
    let protocol = match optional_i64(resolved, "protocol")?.unwrap_or(2) {
        2 => RedisProtocol::Resp2,
        3 => RedisProtocol::Resp3,
        _ => return Err(PyValueError::new_err("unsupported Redis protocol version")),
    };
    Ok(RedisConnectionConfig {
        host,
        port,
        database: optional_i64(resolved, "db")?.unwrap_or(0),
        username: optional_dict_string(resolved, "username")?,
        password: optional_dict_string(resolved, "password")?,
        protocol,
        pool_size,
        read_timeout: optional_dict_duration(resolved, "socket_timeout")?,
        connect_timeout: optional_dict_duration(resolved, "socket_connect_timeout")?,
        socket_keepalive: optional_bool(resolved, "socket_keepalive")?,
        health_check_interval: duration(
            optional_f64(resolved, "health_check_interval")?.unwrap_or(0.0),
        )?,
        client_name: optional_dict_string(resolved, "client_name")?,
        tls,
    })
}

#[inline(never)]
fn project_s3(
    backend: &Bound<'_, PyAny>,
) -> PyResult<Result<S3CacheConfig, UnsupportedCacheConfig>> {
    let client = backend.getattr("s3_client")?;
    if !instance_class_is(&client, "botocore.client", "S3")? {
        return Ok(Err(UnsupportedCacheConfig::S3Client));
    }
    let meta = client.getattr("meta")?;
    let Some(region) = optional_string(meta.getattr("region_name")?)? else {
        return Ok(Err(UnsupportedCacheConfig::S3Option));
    };
    let Some(endpoint_url) = optional_string(meta.getattr("endpoint_url")?)? else {
        return Ok(Err(UnsupportedCacheConfig::S3Option));
    };
    let client_config = meta.getattr("config")?;
    for name in ["s3", "proxies", "client_cert"] {
        if optional_attribute(&client_config, name)?.is_some_and(|value| !value.is_none()) {
            return Ok(Err(UnsupportedCacheConfig::S3Option));
        }
    }
    let signature = match optional_attribute(&client_config, "signature_version")? {
        Some(value) => value.extract::<Option<String>>()?,
        None => None,
    };
    if signature.as_deref() != Some("s3v4") {
        return Ok(Err(UnsupportedCacheConfig::S3Option));
    }
    let insecure = endpoint_url.starts_with("http://");
    let verify = optional_attribute_chain(&client, &["_endpoint", "http_session", "_verify"])?;
    let verified = verify
        .and_then(|value| value.cast::<PyBool>().ok().map(|value| value.is_true()))
        .unwrap_or(false);
    if !verified && !insecure {
        return Ok(Err(UnsupportedCacheConfig::S3Option));
    }
    let credentials = optional_attribute_chain(&client, &["_request_signer", "_credentials"])?
        .ok_or(UnsupportedCacheConfig::S3Credentials);
    let credentials = match credentials {
        Ok(credentials) if !credentials.is_none() => credentials,
        _ => return Ok(Err(UnsupportedCacheConfig::S3Credentials)),
    };
    let auth = if credentials.getattr("method")?.extract::<String>()?.as_str() == "explicit" {
        AwsAuthConfig {
            access_key_id: credentials
                .getattr("access_key")?
                .extract::<Option<String>>()?,
            secret_access_key: credentials
                .getattr("secret_key")?
                .extract::<Option<String>>()?,
            session_token: credentials.getattr("token")?.extract::<Option<String>>()?,
            region_name: Some(region.clone()),
            ..Default::default()
        }
    } else {
        AwsAuthConfig {
            region_name: Some(region.clone()),
            ..Default::default()
        }
    };
    let default_endpoint = endpoint_url == format!("https://s3.{region}.amazonaws.com")
        || (region == "us-east-1" && endpoint_url == "https://s3.amazonaws.com");
    Ok(Ok(S3CacheConfig {
        bucket: backend.getattr("bucket_name")?.extract::<String>()?,
        key_prefix: backend.getattr("key_prefix")?.extract::<String>()?,
        region,
        endpoint: (!default_endpoint).then_some(S3Endpoint { url: endpoint_url }),
        auth,
    }))
}

#[inline(never)]
fn project_standalone_client<'py>(
    client: &Bound<'py, PyAny>,
) -> PyResult<Result<RedisClientProjection<'py>, UnsupportedCacheConfig>> {
    let pool = client.getattr("connection_pool")?;
    if !instance_class_is(&pool, "redis.connection", "ConnectionPool")? {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    }
    let resolved = pool.getattr("connection_kwargs")?.cast_into::<PyDict>()?;
    if has_value(&resolved, "redis_connect_func")? {
        return Ok(Err(UnsupportedCacheConfig::RedisCredentials));
    }
    let connection_class = resolved
        .get_item("connection_class")?
        .unwrap_or(pool.getattr("connection_class")?);
    let tls = if class_is(&connection_class, "redis.connection", "Connection")? {
        None
    } else if class_is(&connection_class, "redis.connection", "SSLConnection")? {
        Some(project_tls(&resolved)?)
    } else {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    };
    Ok(Ok(RedisClientProjection {
        topology: RedisTopology::Standalone,
        host: required_string(&resolved, "host")?,
        port: port(required_i64(&resolved, "port")?)?,
        pool_size: pool.getattr("max_connections")?.extract::<usize>()?,
        resolved,
        tls,
    }))
}

#[inline(never)]
fn project_cluster_client<'py>(
    source: &Bound<'_, PyDict>,
    client: &Bound<'py, PyAny>,
) -> PyResult<Result<RedisClientProjection<'py>, UnsupportedCacheConfig>> {
    let Some(startup_nodes) = startup_nodes(source)? else {
        return Ok(Err(UnsupportedCacheConfig::RedisTopology));
    };
    if !instance_class_is(client, "redis.cluster", "RedisCluster")? {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    }
    let nodes = client.getattr("nodes_manager")?;
    if !class_is(
        &nodes.getattr("connection_pool_class")?,
        "redis.connection",
        "ConnectionPool",
    )? {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    }
    let resolved = nodes.getattr("connection_kwargs")?.cast_into::<PyDict>()?;
    if let Some(connect) = resolved.get_item("redis_connect_func")?
        && !connect.is_none()
    {
        let own_hook = connect
            .getattr("__self__")
            .is_ok_and(|owner| owner.is(client))
            && connect
                .getattr("__func__")
                .and_then(|function| Ok(function.is(&client.get_type().getattr("on_connect")?)))
                .unwrap_or(false);
        if !own_hook {
            return Ok(Err(UnsupportedCacheConfig::RedisCredentials));
        }
    }
    let tls = if optional_bool(&resolved, "ssl")?.unwrap_or(false) {
        Some(project_tls(&resolved)?)
    } else {
        None
    };
    let first = &startup_nodes[0];
    Ok(Ok(RedisClientProjection {
        host: first.host.clone(),
        port: first.port,
        pool_size: optional_i64(&resolved, "max_connections")?
            .map(|value| {
                usize::try_from(value).map_err(|_| PyValueError::new_err("invalid Redis pool size"))
            })
            .transpose()?
            .unwrap_or(REDIS_PY_DEFAULT_MAX_CONNECTIONS),
        topology: RedisTopology::Cluster { startup_nodes },
        resolved,
        tls,
    }))
}

#[inline(never)]
fn startup_nodes(source: &Bound<'_, PyDict>) -> PyResult<Option<Vec<RedisNode>>> {
    let Some(nodes) = source.get_item("startup_nodes")? else {
        return Ok(None);
    };
    let Ok(nodes) = nodes.cast_into::<PyList>() else {
        return Ok(None);
    };
    if nodes.is_empty() {
        return Ok(None);
    }
    let mut parsed = Vec::with_capacity(nodes.len());
    for node in nodes.iter() {
        let Ok(node) = node.cast_into::<PyDict>() else {
            return Ok(None);
        };
        if node.len() != 2 || !has_value(&node, "host")? || !has_value(&node, "port")? {
            return Ok(None);
        }
        let (Ok(host), Ok(port)) = (
            required_string(&node, "host"),
            required_i64(&node, "port").and_then(port),
        ) else {
            return Ok(None);
        };
        parsed.push(RedisNode { host, port });
    }
    Ok(Some(parsed))
}

#[inline(never)]
fn port(value: i64) -> PyResult<u16> {
    u16::try_from(value).map_err(|_| PyValueError::new_err("invalid Redis port"))
}

#[inline(never)]
fn project_valkey_semantic(
    backend: &Bound<'_, PyAny>,
) -> PyResult<Result<ValkeySemanticCacheConfig, UnsupportedCacheConfig>> {
    let client = backend.getattr("sync_client")?;
    let pool = client.getattr("connection_pool")?;
    let Ok((resolved, is_tls)) = project_connection_pool(&pool)? else {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    };
    for key in ["credential_provider", "redis_connect_func"] {
        if has_value(&resolved, key)? {
            return Ok(Err(UnsupportedCacheConfig::RedisCredentials));
        }
    }
    if is_tls {
        return Ok(Err(UnsupportedCacheConfig::ValkeyTls));
    }
    let host = required_string(&resolved, "host")?;
    if host.is_empty() {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    }
    let connection = resolved_connection(
        &resolved,
        host,
        port(required_i64(&resolved, "port")?)?,
        pool.getattr("max_connections")?.extract::<usize>()?,
        None,
    )?;
    Ok(Ok(ValkeySemanticCacheConfig {
        similarity_threshold: backend.getattr("similarity_threshold")?.extract()?,
        index_name: backend.getattr("index_name")?.extract()?,
        connection,
    }))
}

#[inline(never)]
fn project_connection_pool<'py>(
    pool: &Bound<'py, PyAny>,
) -> PyResult<Result<(Bound<'py, PyDict>, bool), UnsupportedCacheConfig>> {
    if !instance_class_is(pool, "redis.connection", "ConnectionPool")? {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    }
    let resolved = pool.getattr("connection_kwargs")?.cast_into::<PyDict>()?;
    let connection_class = resolved
        .get_item("connection_class")?
        .unwrap_or(pool.getattr("connection_class")?);
    let is_tls = if class_is(&connection_class, "redis.connection", "Connection")? {
        false
    } else if class_is(&connection_class, "redis.connection", "SSLConnection")? {
        true
    } else {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    };
    Ok(Ok((resolved, is_tls)))
}

#[inline(never)]
fn project_tls(values: &Bound<'_, PyDict>) -> PyResult<RedisTlsConfig> {
    Ok(RedisTlsConfig {
        certificate_requirement: certificate_requirement(values)?,
        check_hostname: optional_bool(values, "ssl_check_hostname")?.unwrap_or(false),
        ca_certificate: optional_dict_string(values, "ssl_ca_certs")?,
        ca_data: optional_dict_string(values, "ssl_ca_data")?,
        client_certificate: optional_dict_string(values, "ssl_certfile")?,
        client_key: optional_dict_string(values, "ssl_keyfile")?,
    })
}

#[inline(never)]
fn certificate_requirement(values: &Bound<'_, PyDict>) -> PyResult<CertificateRequirement> {
    let Some(value) = values.get_item("ssl_cert_reqs")? else {
        return Ok(CertificateRequirement::Required);
    };
    if value.is_none() {
        return Ok(CertificateRequirement::Required);
    }
    if let Ok(number) = value.extract::<i32>() {
        return match number {
            0 => Ok(CertificateRequirement::None),
            1 => Ok(CertificateRequirement::Optional),
            2 => Ok(CertificateRequirement::Required),
            _ => Err(PyValueError::new_err(
                "invalid Redis TLS certificate requirement",
            )),
        };
    }
    let text = value.str()?;
    let text = text.to_str()?;
    if text.eq_ignore_ascii_case("none") || text.eq_ignore_ascii_case("cert_none") {
        return Ok(CertificateRequirement::None);
    }
    if text.eq_ignore_ascii_case("optional") || text.eq_ignore_ascii_case("cert_optional") {
        return Ok(CertificateRequirement::Optional);
    }
    if text.eq_ignore_ascii_case("required") || text.eq_ignore_ascii_case("cert_required") {
        return Ok(CertificateRequirement::Required);
    }
    Err(PyValueError::new_err(
        "invalid Redis TLS certificate requirement",
    ))
}

#[inline(never)]
fn instance_class_is(value: &Bound<'_, PyAny>, module: &str, name: &str) -> PyResult<bool> {
    class_is(value.get_type().as_any(), module, name)
}

#[inline(never)]
fn class_is(value: &Bound<'_, PyAny>, module: &str, name: &str) -> PyResult<bool> {
    Ok(value
        .getattr("__module__")?
        .cast_into::<PyString>()?
        .to_str()?
        == module
        && value
            .getattr("__qualname__")?
            .cast_into::<PyString>()?
            .to_str()?
            == name)
}

#[inline(never)]
fn optional_attribute_string(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    match value.getattr(name) {
        Ok(value) => optional_string(value),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(value.py()) => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

#[inline(never)]
fn optional_attribute<'py>(
    value: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match value.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.is_instance_of::<PyAttributeError>(value.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

#[inline(never)]
fn optional_attribute_chain<'py>(
    value: &Bound<'py, PyAny>,
    names: &[&str],
) -> PyResult<Option<Bound<'py, PyAny>>> {
    names
        .iter()
        .try_fold(Some(value.clone()), |current, name| match current {
            Some(current) => optional_attribute(&current, name),
            None => Ok(None),
        })
}

#[inline(never)]
fn optional_string(value: Bound<'_, PyAny>) -> PyResult<Option<String>> {
    Ok(value
        .extract::<Option<String>>()?
        .filter(|value| !value.is_empty()))
}

#[inline(never)]
fn has_value(values: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    Ok(values.get_item(key)?.is_some_and(|value| !value.is_none()))
}

#[inline(never)]
fn required_string(values: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    values
        .get_item(key)?
        .ok_or_else(|| PyTypeError::new_err("Redis connection is incomplete"))?
        .extract::<String>()
}

#[inline(never)]
fn required_i64(values: &Bound<'_, PyDict>, key: &str) -> PyResult<i64> {
    values
        .get_item(key)?
        .ok_or_else(|| PyTypeError::new_err("Redis connection is incomplete"))?
        .extract::<i64>()
}

#[inline(never)]
fn optional_dict_string(values: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match values.get_item(key)? {
        Some(value) if !value.is_none() => optional_string(value),
        _ => Ok(None),
    }
}

#[inline(never)]
fn optional_f64(values: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    match values.get_item(key)? {
        Some(value) => value.extract::<Option<f64>>(),
        None => Ok(None),
    }
}

#[inline(never)]
fn optional_i64(values: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i64>> {
    match values.get_item(key)? {
        Some(value) => value.extract::<Option<i64>>(),
        None => Ok(None),
    }
}

#[inline(never)]
fn optional_bool(values: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<bool>> {
    match values.get_item(key)? {
        Some(value) => value.extract::<Option<bool>>(),
        None => Ok(None),
    }
}

#[inline(never)]
fn optional_coerced_bool(values: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<bool>> {
    let Some(value) = values.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(text) = value.extract::<String>() {
        return Ok(Some(
            text == "1" || text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("yes"),
        ));
    }
    value.extract::<bool>().map(Some)
}

#[inline(never)]
fn optional_dict_duration(values: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Duration>> {
    optional_f64(values, key)?.map(duration).transpose()
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, time::Duration};

    use litellm_cache_redis::{RedisNode, RedisTopology};
    use litellm_cache_redis_semantic::RedisSemanticConfig;
    use pyo3::{prelude::*, types::PyDict};
    use rstest::{fixture, rstest};

    use super::{
        CacheBackendConfig, CacheConfigProjection, CachePolicy, CertificateRequirement,
        GcsCacheConfig, NativeCacheConfig, REDIS_PY_DEFAULT_MAX_CONNECTIONS, RedisConnectionConfig,
        RedisProtocol, RedisSemanticCacheConfig, RedisTlsConfig, UnsupportedCacheConfig,
    };
    use crate::cache::{embedder::PythonEmbedder, native::NativeResponseCache};

    fn cluster_facade<'py>(py: Python<'py>, startup_nodes: &str, hook: &str) -> Bound<'py, PyAny> {
        facade(
            py,
            &format!(
                "RedisCluster = type('RedisCluster', (), {{'__module__': 'redis.cluster', 'on_connect': lambda self, connection: None}})\n\
                 client = RedisCluster()\n\
                 client.nodes_manager = SimpleNamespace(connection_pool_class=ConnectionPool, connection_kwargs={{'password': 'secret', 'redis_connect_func': {hook}, 'protocol': 3, 'ssl': True, 'ssl_cert_reqs': 'none'}})\n\
                 backend = SimpleNamespace(default_ttl=120, namespace='team', redis_flush_size=100, redis_kwargs={{'startup_nodes': {startup_nodes}, 'password': 'secret'}}, redis_client=client)\n\
                 facade = SimpleNamespace(type='redis', mode='default-on', ttl=None, namespace='team', supported_call_types=None, redis_flush_size=100, semantic_cache_scope='key', cache=backend)"
            ),
        )
    }

    fn facade<'py>(py: Python<'py>, body: &str) -> Bound<'py, PyAny> {
        let locals = PyDict::new(py);
        py.run(
            &CString::new(format!(
                "from types import SimpleNamespace\n\
                 ConnectionPool = type('ConnectionPool', (), {{'__module__': 'redis.connection'}})\n\
                 Connection = type('Connection', (), {{'__module__': 'redis.connection'}})\n\
                 SSLConnection = type('SSLConnection', (), {{'__module__': 'redis.connection'}})\n\
                 {body}"
            ))
            .unwrap(),
            None,
            Some(&locals),
        )
        .unwrap();
        locals.get_item("facade").unwrap().unwrap()
    }

    fn native(facade: &Bound<'_, PyAny>) -> NativeCacheConfig {
        match NativeCacheConfig::project(facade).unwrap() {
            CacheConfigProjection::Native(config) => *config,
            CacheConfigProjection::Unsupported(reason) => panic!("{}", reason.message()),
        }
    }

    fn unsupported(facade: &Bound<'_, PyAny>) -> UnsupportedCacheConfig {
        match NativeCacheConfig::project(facade).unwrap() {
            CacheConfigProjection::Native(_) => panic!("configuration must stay on Python"),
            CacheConfigProjection::Unsupported(reason) => reason,
        }
    }

    #[fixture]
    fn interpreter() {
        Python::initialize();
    }

    #[fixture]
    fn connection() -> RedisConnectionConfig {
        RedisConnectionConfig {
            host: "cache.internal".into(),
            port: 6380,
            database: 4,
            username: None,
            password: None,
            protocol: RedisProtocol::Resp2,
            pool_size: REDIS_PY_DEFAULT_MAX_CONNECTIONS,
            read_timeout: None,
            connect_timeout: None,
            socket_keepalive: None,
            health_check_interval: Duration::ZERO,
            client_name: None,
            tls: None,
        }
    }

    fn verified_tls() -> RedisTlsConfig {
        RedisTlsConfig {
            certificate_requirement: CertificateRequirement::Required,
            check_hostname: true,
            ca_certificate: None,
            ca_data: None,
            client_certificate: None,
            client_key: None,
        }
    }

    #[rstest]
    fn projects_effective_memory_configuration(_interpreter: ()) {
        Python::attach(|py| {
            let facade = facade(
                py,
                "backend = SimpleNamespace(default_ttl=913, max_size_in_memory=37, max_size_per_item=8)\n\
                 facade = SimpleNamespace(type='local', mode='default-on', ttl=11.5, namespace=None, supported_call_types=['completion'], redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
            );
            let config = native(&facade);
            assert_eq!(config.policy.semantic_cache_scope, "key");
            assert_eq!(config.policy.redis_flush_size, None);
            let CacheBackendConfig::Memory(memory) = config.backend else {
                panic!("expected memory configuration");
            };
            assert_eq!(memory.default_ttl, Duration::from_secs(913));
            assert_eq!(memory.capacity, 37);
            assert_eq!(memory.max_entry_bytes, 8192);
            let matching = NativeResponseCache::memory(37, Duration::from_secs(913), 8192);
            let mismatched = NativeResponseCache::memory(37, Duration::from_secs(913), 8191);
            let matching_config = NativeCacheConfig {
                policy: config.policy,
                backend: CacheBackendConfig::Memory(memory),
            };
            assert_eq!(matching_config.service_mismatch(&matching), None);
            assert_eq!(
                matching_config.service_mismatch(&mismatched),
                Some("facade and native backend item limits must match")
            );
        });
    }

    #[rstest]
    fn redis_semantic_service_mismatch_accepts_backend_precision_threshold(_interpreter: ()) {
        Python::attach(|py| {
            let facade = facade(
                py,
                "backend = SimpleNamespace(_redis_url='redis://127.0.0.1/', _index_name='semantic_idx', similarity_threshold=0.8, embedding_model='text-embedding-3-small', embedding_max_input_tokens=None, embedding_timeout=None)\n\
                 facade = SimpleNamespace(type='redis-semantic', mode='default-on', ttl=None, namespace=None, supported_call_types=None, redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
            );
            let backend = facade.getattr("cache").unwrap();
            let embedder = PythonEmbedder::new(backend.clone().unbind());
            let CacheBackendConfig::RedisSemantic(config) = native(&facade).backend else {
                panic!("expected Redis semantic configuration");
            };
            let service = NativeResponseCache::redis_semantic(
                &config.redis_url,
                embedder,
                RedisSemanticConfig {
                    index_name: config.index_name.clone(),
                    similarity_threshold: config.similarity_threshold as f32,
                },
            )
            .unwrap();
            let matching_config = NativeCacheConfig {
                policy: CachePolicy {
                    redis_flush_size: None,
                    semantic_cache_scope: "key".into(),
                },
                backend: CacheBackendConfig::RedisSemantic(config),
            };
            assert_eq!(matching_config.service_mismatch(&service), None);
        });
    }

    #[rstest]
    fn projects_resolved_redis_tls_configuration(_interpreter: ()) {
        Python::attach(|py| {
            let facade = facade(
                py,
                "pool = ConnectionPool()\n\
                 pool.connection_class = SSLConnection\n\
                 pool.max_connections = 29\n\
                 pool.connection_kwargs = {'host': 'cache.internal', 'port': 6380, 'db': 4, 'username': 'user', 'password': 'secret', 'protocol': 3, 'socket_timeout': 7.5, 'socket_connect_timeout': 2, 'socket_keepalive': True, 'health_check_interval': 15, 'client_name': 'litellm', 'ssl_cert_reqs': 'optional', 'ssl_check_hostname': True, 'ssl_ca_certs': '/ca.pem', 'ssl_ca_data': 'CA DATA', 'ssl_certfile': '/client.pem', 'ssl_keyfile': '/client.key'}\n\
                 client = SimpleNamespace(connection_pool=pool)\n\
                 backend = SimpleNamespace(default_ttl=777, namespace='team', redis_flush_size=31, redis_kwargs={}, redis_client=client)\n\
                 facade = SimpleNamespace(type='redis', mode='default-off', ttl=None, namespace='team', supported_call_types=None, redis_flush_size=31, semantic_cache_scope='key', cache=backend)",
            );
            let config = native(&facade);
            assert_eq!(config.policy.redis_flush_size, Some(31));
            let CacheBackendConfig::Redis(redis) = config.backend else {
                panic!("expected Redis configuration");
            };
            assert_eq!(redis.default_ttl, Duration::from_secs(777));
            assert_eq!(redis.namespace.as_deref(), Some("team"));
            assert_eq!(redis.flush_size, 31);
            assert_eq!(redis.connection.host, "cache.internal");
            assert_eq!(redis.connection.port, 6380);
            assert_eq!(redis.connection.database, 4);
            assert_eq!(redis.connection.protocol, RedisProtocol::Resp3);
            assert_eq!(redis.connection.pool_size, 29);
            assert_eq!(
                redis.connection.read_timeout,
                Some(Duration::from_secs_f64(7.5))
            );
            assert_eq!(
                redis.connection.connect_timeout,
                Some(Duration::from_secs(2))
            );
            assert_eq!(redis.connection.socket_keepalive, Some(true));
            assert_eq!(
                redis.connection.health_check_interval,
                Duration::from_secs(15)
            );
            assert_eq!(redis.connection.client_name.as_deref(), Some("litellm"));
            let tls = redis.connection.tls.as_ref().unwrap();
            assert_eq!(
                tls.certificate_requirement,
                CertificateRequirement::Optional
            );
            assert!(tls.check_hostname);
            assert_eq!(tls.ca_certificate.as_deref(), Some("/ca.pem"));
            assert_eq!(tls.ca_data.as_deref(), Some("CA DATA"));
            assert_eq!(tls.client_certificate.as_deref(), Some("/client.pem"));
            assert_eq!(tls.client_key.as_deref(), Some("/client.key"));
            assert!(matches!(
                redis.connection.native_url(),
                Err(UnsupportedCacheConfig::RedisPoolSize)
            ));
        });
    }

    #[rstest]
    fn projects_valkey_semantic_configuration(_interpreter: ()) {
        Python::attach(|py| {
            let facade = facade(
                py,
                "pool = ConnectionPool()\n\
                 pool.connection_class = Connection\n\
                 pool.max_connections = 12\n\
                 pool.connection_kwargs = {'host': 'cache.internal', 'port': 6390, 'db': 2, 'socket_timeout': 3}\n\
                 client = SimpleNamespace(connection_pool=pool)\n\
                 backend = SimpleNamespace(similarity_threshold=0.85, index_name='semantic_idx', embedding_model='text-embedding-3-small', sync_client=client)\n\
                 facade = SimpleNamespace(type='valkey-semantic', mode='default-on', ttl=None, namespace=None, supported_call_types=None, redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
            );
            let CacheBackendConfig::ValkeySemantic(valkey) = native(&facade).backend else {
                panic!("expected Valkey semantic configuration");
            };
            assert_eq!(valkey.similarity_threshold, 0.85);
            assert_eq!(valkey.index_name, "semantic_idx");
            assert_eq!(valkey.connection.host, "cache.internal");
            assert_eq!(valkey.connection.port, 6390);
            assert_eq!(valkey.connection.database, 2);
            assert_eq!(valkey.connection.pool_size, 12);
            assert_eq!(valkey.connection.protocol, RedisProtocol::Resp2);
            assert_eq!(valkey.connection.read_timeout, Some(Duration::from_secs(3)));
            assert!(valkey.connection.tls.is_none());
        });
    }

    #[rstest]
    #[case::valkey_tls(
        "pool = ConnectionPool()\n\
         pool.connection_class = SSLConnection\n\
         pool.connection_kwargs = {'host': 'cache.internal', 'port': 6390}\n\
         client = SimpleNamespace(connection_pool=pool)\n\
         backend = SimpleNamespace(similarity_threshold=0.85, index_name='semantic_idx', embedding_model='text-embedding-3-small', sync_client=client)\n\
         facade = SimpleNamespace(type='valkey-semantic', mode='default-on', ttl=None, namespace=None, supported_call_types=None, redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
        "native Valkey semantic cache does not support TLS connections"
    )]
    #[case::valkey_dynamic_auth(
        "pool = ConnectionPool()\n\
         pool.connection_class = Connection\n\
         pool.connection_kwargs = {'host': 'cache.internal', 'port': 6390, 'credential_provider': object()}\n\
         client = SimpleNamespace(connection_pool=pool)\n\
         backend = SimpleNamespace(similarity_threshold=0.85, index_name='semantic_idx', embedding_model='text-embedding-3-small', sync_client=client)\n\
         facade = SimpleNamespace(type='valkey-semantic', mode='default-on', ttl=None, namespace=None, supported_call_types=None, redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
        "native Redis credentials require Python"
    )]
    #[case::redis_dynamic_auth(
        "backend = SimpleNamespace(redis_kwargs={'credential_provider': object()})\n\
         facade = SimpleNamespace(type='redis', mode='default-on', ttl=None, namespace=None, supported_call_types=[], redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
        "native Redis credentials require Python"
    )]
    #[case::gcs_without_bucket(
        "backend = SimpleNamespace(bucket_name=None, key_prefix='', path_service_account=None)\n\
         facade = SimpleNamespace(type='gcs', mode='default-on', ttl=None, namespace=None, supported_call_types=None, redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
        "native GCS cache requires a configured bucket name"
    )]
    fn configurations_that_stay_on_python(
        _interpreter: (),
        #[case] body: &str,
        #[case] message: &str,
    ) {
        Python::attach(|py| {
            assert_eq!(unsupported(&facade(py, body)).message(), message);
        });
    }

    #[rstest]
    fn projects_cluster_startup_nodes_as_redis_topology(_interpreter: ()) {
        Python::attach(|py| {
            let facade = cluster_facade(
                py,
                "[{'host': 'node-a', 'port': 7000}, {'host': 'node-b', 'port': 7001}]",
                "client.on_connect",
            );
            let CacheBackendConfig::Redis(redis) = native(&facade).backend else {
                panic!("expected Redis configuration");
            };
            let expected = RedisTopology::Cluster {
                startup_nodes: vec![
                    RedisNode {
                        host: "node-a".into(),
                        port: 7000,
                    },
                    RedisNode {
                        host: "node-b".into(),
                        port: 7001,
                    },
                ],
            };
            assert_eq!(redis.topology, expected);
            assert_eq!(redis.connection.host, "node-a");
            assert_eq!(redis.connection.port, 7000);
            assert_eq!(redis.connection.password.as_deref(), Some("secret"));
            assert_eq!(redis.connection.protocol, RedisProtocol::Resp3);
            assert_eq!(
                redis
                    .connection
                    .tls
                    .as_ref()
                    .unwrap()
                    .certificate_requirement,
                CertificateRequirement::None
            );
            assert!(matches!(
                redis.connection.native_url(),
                Err(UnsupportedCacheConfig::RedisTlsVerification)
            ));
        });
    }

    #[rstest]
    fn projects_gcs_configuration(_interpreter: ()) {
        Python::attach(|py| {
            let facade = facade(
                py,
                "backend = SimpleNamespace(bucket_name='bucket', key_prefix='cache/', path_service_account='credentials.json')\n\
                 facade = SimpleNamespace(type='gcs', mode='default-on', ttl=None, namespace=None, supported_call_types=None, redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
            );
            let config = native(&facade);
            let CacheBackendConfig::Gcs(gcs) = config.backend else {
                panic!("expected GCS configuration");
            };
            assert_eq!(
                gcs,
                GcsCacheConfig {
                    bucket_name: "bucket".into(),
                    key_prefix: "cache/".into(),
                    path_service_account: Some("credentials.json".into()),
                }
            );
            let matching = NativeResponseCache::gcs(
                litellm_cache_gcs::GcsConfig {
                    bucket_name: "bucket".into(),
                    gcs_path: Some("cache/".into()),
                    path_service_account: Some("credentials.json".into()),
                    endpoint: litellm_cache_gcs::DEFAULT_ENDPOINT.into(),
                },
                litellm_http::Client::plain_for_test(),
                Some("token".into()),
            );
            let matching_config = NativeCacheConfig {
                policy: config.policy,
                backend: CacheBackendConfig::Gcs(gcs),
            };
            assert_eq!(matching_config.service_mismatch(&matching), None);
        });
    }

    #[rstest]
    #[case::extra_node_field(
        "[{'host': 'node-a', 'port': 7000, 'server_type': 'primary'}]",
        "client.on_connect",
        "native Redis topology is not implemented"
    )]
    #[case::non_numeric_port(
        "[{'host': 'node-a', 'port': 'seven'}]",
        "client.on_connect",
        "native Redis topology is not implemented"
    )]
    #[case::empty("[]", "client.on_connect", "native Redis topology is not implemented")]
    #[case::foreign_hook(
        "[{'host': 'node-a', 'port': 7000}]",
        "lambda connection: None",
        "native Redis credentials require Python"
    )]
    fn malformed_startup_nodes_and_foreign_connect_hooks_stay_on_python(
        _interpreter: (),
        #[case] startup_nodes: &str,
        #[case] hook: &str,
        #[case] message: &str,
    ) {
        Python::attach(|py| {
            let facade = cluster_facade(py, startup_nodes, hook);
            assert_eq!(unsupported(&facade).message(), message);
        });
    }

    #[rstest]
    #[case::plain(|_: &mut RedisConnectionConfig| {}, "redis://cache.internal:6380/4")]
    #[case::credentials(
        |connection: &mut RedisConnectionConfig| {
            connection.username = Some("user".into());
            connection.password = Some("p@ss:word".into());
        },
        "redis://user:p%40ss%3Aword@cache.internal:6380/4"
    )]
    #[case::password_only(
        |connection: &mut RedisConnectionConfig| connection.password = Some("secret".into()),
        "redis://:secret@cache.internal:6380/4"
    )]
    #[case::resp3(
        |connection: &mut RedisConnectionConfig| connection.protocol = RedisProtocol::Resp3,
        "redis://cache.internal:6380/4?protocol=resp3"
    )]
    #[case::ipv6(
        |connection: &mut RedisConnectionConfig| connection.host = "::1".into(),
        "redis://[::1]:6380/4"
    )]
    #[case::verified_tls(
        |connection: &mut RedisConnectionConfig| connection.tls = Some(verified_tls()),
        "rediss://cache.internal:6380/4"
    )]
    #[case::optional_certificate(
        |connection: &mut RedisConnectionConfig| {
            connection.tls = Some(RedisTlsConfig {
                certificate_requirement: CertificateRequirement::Optional,
                ..verified_tls()
            });
        },
        "rediss://cache.internal:6380/4"
    )]
    #[case::keepalive_off(
        |connection: &mut RedisConnectionConfig| connection.socket_keepalive = Some(false),
        "redis://cache.internal:6380/4"
    )]
    #[case::redis_cache_socket_timeout(
        |connection: &mut RedisConnectionConfig| {
            connection.read_timeout = Some(Duration::from_secs(5));
        },
        "redis://cache.internal:6380/4"
    )]
    fn native_url_encodes_the_resolved_connection(
        mut connection: RedisConnectionConfig,
        #[case] configure: fn(&mut RedisConnectionConfig),
        #[case] expected: &str,
    ) {
        configure(&mut connection);
        assert_eq!(connection.native_url().ok().as_deref(), Some(expected));
    }

    #[rstest]
    #[case::pool_size(
        |connection: &mut RedisConnectionConfig| connection.pool_size = 50,
        "native Redis uses a fixed connection pool; max_connections requires Python"
    )]
    #[case::socket_timeout(
        |connection: &mut RedisConnectionConfig| {
            connection.read_timeout = Some(Duration::from_millis(100));
        },
        "native Redis uses fixed socket timeouts; socket_timeout and socket_connect_timeout require Python"
    )]
    #[case::connect_timeout(
        |connection: &mut RedisConnectionConfig| {
            connection.connect_timeout = Some(Duration::from_secs(1));
        },
        "native Redis uses fixed socket timeouts; socket_timeout and socket_connect_timeout require Python"
    )]
    #[case::keepalive(
        |connection: &mut RedisConnectionConfig| connection.socket_keepalive = Some(true),
        "native Redis does not support socket_keepalive"
    )]
    #[case::health_check(
        |connection: &mut RedisConnectionConfig| {
            connection.health_check_interval = Duration::from_secs(25);
        },
        "native Redis does not support health_check_interval"
    )]
    #[case::client_name(
        |connection: &mut RedisConnectionConfig| connection.client_name = Some("litellm".into()),
        "native Redis does not support client_name"
    )]
    #[case::custom_ca(
        |connection: &mut RedisConnectionConfig| {
            connection.tls = Some(RedisTlsConfig {
                ca_certificate: Some("/ca.pem".into()),
                ..verified_tls()
            });
        },
        "native Redis TLS does not support ssl_ca_certs, ssl_ca_data, ssl_certfile or ssl_keyfile"
    )]
    #[case::client_certificate(
        |connection: &mut RedisConnectionConfig| {
            connection.tls = Some(RedisTlsConfig {
                client_certificate: Some("/client.pem".into()),
                client_key: Some("/client.key".into()),
                ..verified_tls()
            });
        },
        "native Redis TLS does not support ssl_ca_certs, ssl_ca_data, ssl_certfile or ssl_keyfile"
    )]
    #[case::unverified(
        |connection: &mut RedisConnectionConfig| {
            connection.tls = Some(RedisTlsConfig {
                certificate_requirement: CertificateRequirement::None,
                check_hostname: false,
                ..verified_tls()
            });
        },
        "native Redis TLS always verifies the certificate and hostname; ssl_cert_reqs=none and ssl_check_hostname=false require Python"
    )]
    #[case::hostname_unchecked(
        |connection: &mut RedisConnectionConfig| {
            connection.tls = Some(RedisTlsConfig {
                check_hostname: false,
                ..verified_tls()
            });
        },
        "native Redis TLS always verifies the certificate and hostname; ssl_cert_reqs=none and ssl_check_hostname=false require Python"
    )]
    fn native_url_declines_settings_the_native_client_cannot_honor(
        mut connection: RedisConnectionConfig,
        #[case] configure: fn(&mut RedisConnectionConfig),
        #[case] message: &str,
    ) {
        configure(&mut connection);
        let Err(reason) = connection.native_url() else {
            panic!("{message}");
        };
        assert_eq!(reason.message(), message);
    }

    #[rstest]
    #[case::plain("redis://:secret@127.0.0.1:6379", true)]
    #[case::database("redis://127.0.0.1:6379/2", true)]
    #[case::unix("unix:///tmp/redis.sock", true)]
    #[case::tls("rediss://cache.internal:6380", false)]
    #[case::query_options("redis://127.0.0.1:6379?socket_timeout=1", false)]
    #[case::malformed("not a url", false)]
    fn redis_semantic_native_url_accepts_only_plain_urls(#[case] url: &str, #[case] native: bool) {
        let config = RedisSemanticCacheConfig {
            redis_url: url.into(),
            index_name: "idx".into(),
            similarity_threshold: 0.8,
        };
        match config.native_url() {
            Ok(value) => {
                assert!(native, "{url} must decline");
                assert_eq!(value, url);
            }
            Err(reason) => {
                assert!(!native, "{url} must be native");
                assert_eq!(
                    reason.message(),
                    "native Redis semantic cache does not support TLS or query options in redis_url"
                );
            }
        }
    }
}
