use std::time::Duration;

use litellm_cache::CacheType;
use litellm_cache_redis::{RedisNode, RedisTopology};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict, PyList, PyString},
};

use super::{native::NativeResponseCache, request::duration};

#[allow(dead_code, reason = "consumed by the cache activation follow-up")]
pub(super) struct CachePolicy {
    pub(super) mode: String,
    pub(super) ttl: Option<Duration>,
    pub(super) namespace: Option<String>,
    pub(super) supported_call_types: Option<Vec<String>>,
    pub(super) redis_flush_size: Option<usize>,
    pub(super) semantic_cache_scope: String,
}

pub(super) struct MemoryCacheConfig {
    pub(super) default_ttl: Duration,
    pub(super) capacity: usize,
    pub(super) max_entry_bytes: usize,
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

#[allow(dead_code, reason = "consumed by the cache activation follow-up")]
pub(super) struct RedisTlsConfig {
    pub(super) certificate_requirement: CertificateRequirement,
    pub(super) check_hostname: bool,
    pub(super) ca_certificate: Option<String>,
    pub(super) ca_data: Option<String>,
    pub(super) client_certificate: Option<String>,
    pub(super) client_key: Option<String>,
}

#[allow(dead_code, reason = "consumed by the cache activation follow-up")]
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

#[allow(dead_code, reason = "consumed by the cache activation follow-up")]
pub(super) struct RedisCacheConfig {
    pub(super) default_ttl: Duration,
    pub(super) namespace: Option<String>,
    pub(super) flush_size: usize,
    pub(super) topology: RedisTopology,
    pub(super) connection: RedisConnectionConfig,
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

#[allow(dead_code, reason = "consumed by the cache activation follow-up")]
pub(super) struct ValkeySemanticCacheConfig {
    pub(super) similarity_threshold: f64,
    pub(super) index_name: String,
    pub(super) embedding_model: String,
    pub(super) connection: RedisConnectionConfig,
}

pub(super) enum CacheBackendConfig {
    Memory(MemoryCacheConfig),
    Redis(Box<RedisCacheConfig>),
    ValkeySemantic(Box<ValkeySemanticCacheConfig>),
}

#[allow(dead_code, reason = "consumed by the cache activation follow-up")]
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
}

impl UnsupportedCacheConfig {
    pub(super) fn message(&self) -> &'static str {
        match self {
            Self::Backend => "native cache backend is not implemented",
            Self::RedisTopology => "native Redis topology is not implemented",
            Self::RedisCredentials => "native Redis credentials require Python",
            Self::RedisConnection => "native Redis connection type is not implemented",
            Self::RedisOption => "native Redis configuration requires Python",
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
            mode: facade.getattr("mode")?.extract::<String>()?,
            ttl: optional_duration(facade.getattr("ttl")?)?,
            namespace: optional_string(facade.getattr("namespace")?)?,
            supported_call_types: facade
                .getattr("supported_call_types")?
                .extract::<Option<Vec<String>>>()?,
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
            Some(CacheType::ValkeySemantic) => match project_valkey_semantic(&backend)? {
                Ok(backend) => Ok(CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::ValkeySemantic(Box::new(backend)),
                }))),
                Err(reason) => Ok(CacheConfigProjection::Unsupported(reason)),
            },
            Some(
                CacheType::RedisSemantic
                | CacheType::S3
                | CacheType::Disk
                | CacheType::QdrantSemantic
                | CacheType::AzureBlob
                | CacheType::Gcs,
            )
            | None => Ok(CacheConfigProjection::Unsupported(
                UnsupportedCacheConfig::Backend,
            )),
        }
    }

    pub(super) fn service_mismatch(&self, service: &NativeResponseCache) -> Option<&'static str> {
        if !matches!(self.backend, CacheBackendConfig::ValkeySemantic(_))
            && service.default_ttl()
                != Some(match &self.backend {
                    CacheBackendConfig::Memory(config) => config.default_ttl,
                    CacheBackendConfig::Redis(config) => config.default_ttl,
                    CacheBackendConfig::ValkeySemantic(_) => Duration::ZERO,
                })
        {
            return Some("facade and native backend default TTLs must match");
        }
        match &self.backend {
            CacheBackendConfig::Memory(config) if service.kind() != "memory" => {
                Some("facade and native backend types must match")
            }
            CacheBackendConfig::Memory(config) if service.capacity() != Some(config.capacity) => {
                Some("facade and native backend capacities must match")
            }
            CacheBackendConfig::Memory(config)
                if service.max_entry_bytes() != Some(config.max_entry_bytes) =>
            {
                Some("facade and native backend item limits must match")
            }
            CacheBackendConfig::Memory(_) => None,
            CacheBackendConfig::Redis(_) if service.kind() != "redis" => {
                Some("facade and native backend types must match")
            }
            CacheBackendConfig::Redis(config) if service.topology() != Some(&config.topology) => {
                Some("facade and native backend topologies must match")
            }
            CacheBackendConfig::Redis(config) => (service.namespace()
                != config.namespace.as_deref())
            .then_some("facade and native backend namespaces must match"),
            CacheBackendConfig::ValkeySemantic(config) => {
                if service.kind() != "valkey-semantic" {
                    return Some("facade and native backend types must match");
                }
                let Some((threshold, index_name)) = service.semantic_config() else {
                    return Some("facade and native backend types must match");
                };
                (threshold != config.similarity_threshold || index_name != config.index_name)
                    .then_some("facade and native semantic settings must match")
            }
        }
    }
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

    let protocol = match optional_i64(&resolved, "protocol")?.unwrap_or(2) {
        2 => RedisProtocol::Resp2,
        3 => RedisProtocol::Resp3,
        _ => return Err(PyValueError::new_err("unsupported Redis protocol version")),
    };
    let health_check_interval =
        duration(optional_f64(&resolved, "health_check_interval")?.unwrap_or(0.0))?;
    Ok(Ok(RedisCacheConfig {
        default_ttl: duration(backend.getattr("default_ttl")?.extract::<f64>()?)?,
        namespace: optional_attribute_string(backend, "namespace")?,
        flush_size: backend.getattr("redis_flush_size")?.extract::<usize>()?,
        topology,
        connection: RedisConnectionConfig {
            host,
            port,
            database: optional_i64(&resolved, "db")?.unwrap_or(0),
            username: optional_dict_string(&resolved, "username")?,
            password: optional_dict_string(&resolved, "password")?,
            protocol,
            pool_size,
            read_timeout: optional_dict_duration(&resolved, "socket_timeout")?,
            connect_timeout: optional_dict_duration(&resolved, "socket_connect_timeout")?,
            socket_keepalive: optional_bool(&resolved, "socket_keepalive")?,
            health_check_interval,
            client_name: optional_dict_string(&resolved, "client_name")?,
            tls,
        },
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
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    }
    let connection = RedisConnectionConfig {
        host: required_string(&resolved, "host")?,
        port: u16::try_from(required_i64(&resolved, "port")?)
            .map_err(|_| PyValueError::new_err("invalid Redis port"))?,
        database: optional_i64(&resolved, "db")?.unwrap_or(0),
        username: optional_dict_string(&resolved, "username")?,
        password: optional_dict_string(&resolved, "password")?,
        protocol: RedisProtocol::Resp2,
        pool_size: pool.getattr("max_connections")?.extract::<usize>()?,
        read_timeout: None,
        connect_timeout: None,
        socket_keepalive: None,
        health_check_interval: Duration::ZERO,
        client_name: None,
        tls: None,
    };
    if connection.host.is_empty() {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    }
    Ok(Ok(ValkeySemanticCacheConfig {
        similarity_threshold: backend.getattr("similarity_threshold")?.extract()?,
        index_name: backend.getattr("index_name")?.extract()?,
        embedding_model: backend.getattr("embedding_model")?.extract()?,
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
fn optional_duration(value: Bound<'_, PyAny>) -> PyResult<Option<Duration>> {
    value.extract::<Option<f64>>()?.map(duration).transpose()
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
    use std::ffi::CString;

    use pyo3::{prelude::*, types::PyDict};

    use litellm_cache_redis::{RedisNode, RedisTopology};

    use super::{
        CacheBackendConfig, CacheConfigProjection, CertificateRequirement, NativeCacheConfig,
        RedisProtocol,
    };
    use crate::cache::native::NativeResponseCache;

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

    #[test]
    fn projects_effective_memory_configuration() {
        Python::initialize();
        Python::attach(|py| {
            let facade = facade(
                py,
                "backend = SimpleNamespace(default_ttl=913, max_size_in_memory=37, max_size_per_item=8)\n\
                 facade = SimpleNamespace(type='local', mode='default-on', ttl=11.5, namespace=None, supported_call_types=['completion'], redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
            );
            let CacheConfigProjection::Native(config) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("memory cache should be supported");
            };
            assert_eq!(
                config.policy.ttl.unwrap(),
                std::time::Duration::from_secs_f64(11.5)
            );
            let CacheBackendConfig::Memory(memory) = config.backend else {
                panic!("expected memory configuration");
            };
            assert_eq!(memory.default_ttl, std::time::Duration::from_secs(913));
            assert_eq!(memory.capacity, 37);
            assert_eq!(memory.max_entry_bytes, 8192);
            let matching =
                NativeResponseCache::memory(37, std::time::Duration::from_secs(913), 8192);
            let mismatched =
                NativeResponseCache::memory(37, std::time::Duration::from_secs(913), 8191);
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

    #[test]
    fn projects_resolved_redis_tls_configuration() {
        Python::initialize();
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
            let CacheConfigProjection::Native(config) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("Redis cache should be supported");
            };
            let CacheBackendConfig::Redis(redis) = config.backend else {
                panic!("expected Redis configuration");
            };
            assert_eq!(redis.default_ttl, std::time::Duration::from_secs(777));
            assert_eq!(redis.namespace.as_deref(), Some("team"));
            assert_eq!(redis.flush_size, 31);
            assert_eq!(redis.connection.host, "cache.internal");
            assert_eq!(redis.connection.port, 6380);
            assert_eq!(redis.connection.database, 4);
            assert_eq!(redis.connection.protocol, RedisProtocol::Resp3);
            assert_eq!(redis.connection.pool_size, 29);
            let tls = redis.connection.tls.unwrap();
            assert_eq!(
                tls.certificate_requirement,
                CertificateRequirement::Optional
            );
            assert!(tls.check_hostname);
            assert_eq!(tls.ca_certificate.as_deref(), Some("/ca.pem"));
            assert_eq!(tls.ca_data.as_deref(), Some("CA DATA"));
            assert_eq!(tls.client_certificate.as_deref(), Some("/client.pem"));
            assert_eq!(tls.client_key.as_deref(), Some("/client.key"));
        });
    }

    #[test]
    fn projects_valkey_semantic_configuration() {
        Python::initialize();
        Python::attach(|py| {
            let facade = facade(
                py,
                "pool = ConnectionPool()\n\
                 pool.connection_class = Connection\n\
                 pool.max_connections = 12\n\
                 pool.connection_kwargs = {'host': 'cache.internal', 'port': 6390, 'db': 2}\n\
                 client = SimpleNamespace(connection_pool=pool)\n\
                 backend = SimpleNamespace(similarity_threshold=0.85, index_name='semantic_idx', embedding_model='text-embedding-3-small', sync_client=client)\n\
                 facade = SimpleNamespace(type='valkey-semantic', mode='default-on', ttl=None, namespace=None, supported_call_types=None, redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
            );
            let CacheConfigProjection::Native(config) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("Valkey semantic cache should be supported");
            };
            let CacheBackendConfig::ValkeySemantic(valkey) = config.backend else {
                panic!("expected Valkey semantic configuration");
            };
            assert_eq!(valkey.similarity_threshold, 0.85);
            assert_eq!(valkey.index_name, "semantic_idx");
            assert_eq!(valkey.embedding_model, "text-embedding-3-small");
            assert_eq!(valkey.connection.host, "cache.internal");
            assert_eq!(valkey.connection.port, 6390);
            assert_eq!(valkey.connection.database, 2);
            assert_eq!(valkey.connection.pool_size, 12);
            assert_eq!(valkey.connection.protocol, RedisProtocol::Resp2);
            assert!(valkey.connection.tls.is_none());
        });
    }

    #[test]
    fn valkey_semantic_tls_stays_on_python() {
        Python::initialize();
        Python::attach(|py| {
            let facade = facade(
                py,
                "pool = ConnectionPool()\n\
                 pool.connection_class = SSLConnection\n\
                 pool.connection_kwargs = {'host': 'cache.internal', 'port': 6390}\n\
                 client = SimpleNamespace(connection_pool=pool)\n\
                 backend = SimpleNamespace(similarity_threshold=0.85, index_name='semantic_idx', embedding_model='text-embedding-3-small', sync_client=client)\n\
                 facade = SimpleNamespace(type='valkey-semantic', mode='default-on', ttl=None, namespace=None, supported_call_types=None, redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
            );
            let CacheConfigProjection::Unsupported(reason) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("TLS Valkey semantic cache should stay on Python");
            };
            assert_eq!(
                reason.message(),
                "native Redis connection type is not implemented"
            );
        });
    }

    #[test]
    fn valkey_semantic_dynamic_auth_stays_on_python() {
        Python::initialize();
        Python::attach(|py| {
            let facade = facade(
                py,
                "pool = ConnectionPool()\n\
                 pool.connection_class = Connection\n\
                 pool.connection_kwargs = {'host': 'cache.internal', 'port': 6390, 'credential_provider': object()}\n\
                 client = SimpleNamespace(connection_pool=pool)\n\
                 backend = SimpleNamespace(similarity_threshold=0.85, index_name='semantic_idx', embedding_model='text-embedding-3-small', sync_client=client)\n\
                 facade = SimpleNamespace(type='valkey-semantic', mode='default-on', ttl=None, namespace=None, supported_call_types=None, redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
            );
            let CacheConfigProjection::Unsupported(reason) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("dynamic Valkey authentication must stay on Python");
            };
            assert_eq!(reason.message(), "native Redis credentials require Python");
        });
    }

    #[test]
    fn dynamic_redis_auth_stays_on_python() {
        Python::initialize();
        Python::attach(|py| {
            let facade = facade(
                py,
                "backend = SimpleNamespace(redis_kwargs={'credential_provider': object()})\n\
                 facade = SimpleNamespace(type='redis', mode='default-on', ttl=None, namespace=None, supported_call_types=[], redis_flush_size=None, semantic_cache_scope='key', cache=backend)",
            );
            let CacheConfigProjection::Unsupported(reason) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("dynamic authentication must stay on Python");
            };
            assert_eq!(reason.message(), "native Redis credentials require Python");
        });
    }

    #[test]
    fn projects_cluster_startup_nodes_as_redis_topology() {
        Python::initialize();
        Python::attach(|py| {
            let facade = cluster_facade(
                py,
                "[{'host': 'node-a', 'port': 7000}, {'host': 'node-b', 'port': 7001}]",
                "client.on_connect",
            );
            let CacheConfigProjection::Native(config) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("cluster startup nodes should project natively");
            };
            let CacheBackendConfig::Redis(redis) = &config.backend else {
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
        });
    }

    #[test]
    fn malformed_startup_nodes_and_foreign_connect_hooks_stay_on_python() {
        Python::initialize();
        Python::attach(|py| {
            for (startup_nodes, hook, message) in [
                (
                    "[{'host': 'node-a', 'port': 7000, 'server_type': 'primary'}]",
                    "client.on_connect",
                    "native Redis topology is not implemented",
                ),
                (
                    "[{'host': 'node-a', 'port': 'seven'}]",
                    "client.on_connect",
                    "native Redis topology is not implemented",
                ),
                (
                    "[]",
                    "client.on_connect",
                    "native Redis topology is not implemented",
                ),
                (
                    "[{'host': 'node-a', 'port': 7000}]",
                    "lambda connection: None",
                    "native Redis credentials require Python",
                ),
            ] {
                let facade = cluster_facade(py, startup_nodes, hook);
                let CacheConfigProjection::Unsupported(reason) =
                    NativeCacheConfig::project(&facade).unwrap()
                else {
                    panic!("{startup_nodes} with {hook} must stay on Python");
                };
                assert_eq!(reason.message(), message, "{startup_nodes} with {hook}");
            }
        });
    }
}
