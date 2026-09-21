use std::time::Duration;

use litellm_auth_aws::AwsAuthConfig;
use litellm_cache::CacheType;
use litellm_cache_s3::{S3CacheConfig, S3Endpoint};
use pyo3::{
    exceptions::{PyAttributeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyBool, PyDict, PyString},
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
    pub(super) connection: RedisConnectionConfig,
}

pub(super) enum CacheBackendConfig {
    Memory(MemoryCacheConfig),
    Redis(Box<RedisCacheConfig>),
    S3(Box<S3CacheConfig>),
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
    S3Client,
    S3Credentials,
    S3Option,
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
            Some(CacheType::S3) => match project_s3(&backend)? {
                Ok(backend) => Ok(CacheConfigProjection::Native(Box::new(Self {
                    policy,
                    backend: CacheBackendConfig::S3(Box::new(backend)),
                }))),
                Err(reason) => Ok(CacheConfigProjection::Unsupported(reason)),
            },
            Some(
                CacheType::RedisSemantic
                | CacheType::ValkeySemantic
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
        if service.default_ttl()
            != match &self.backend {
                CacheBackendConfig::Memory(config) => Some(config.default_ttl),
                CacheBackendConfig::Redis(config) => Some(config.default_ttl),
                CacheBackendConfig::S3(_) => None,
            }
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
            CacheBackendConfig::Redis(config) => (service.namespace()
                != config.namespace.as_deref())
            .then_some("facade and native backend namespaces must match"),
            CacheBackendConfig::S3(_) if service.kind() != "s3" => {
                Some("facade and native backend types must match")
            }
            CacheBackendConfig::S3(config) if service.bucket() != Some(config.bucket.as_str()) => {
                Some("facade and native backend buckets must match")
            }
            CacheBackendConfig::S3(config)
                if service.key_prefix() != Some(config.key_prefix.as_str()) =>
            {
                Some("facade and native backend key prefixes must match")
            }
            CacheBackendConfig::S3(_) => None,
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
    if has_value(&source, "startup_nodes")? {
        return Ok(Err(UnsupportedCacheConfig::RedisTopology));
    }
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
    let pool = client.getattr("connection_pool")?;
    if !instance_class_is(&pool, "redis.connection", "ConnectionPool")? {
        return Ok(Err(UnsupportedCacheConfig::RedisConnection));
    }
    let resolved = pool.getattr("connection_kwargs")?.cast_into::<PyDict>()?;
    for key in ["credential_provider", "redis_connect_func"] {
        if has_value(&resolved, key)? {
            return Ok(Err(UnsupportedCacheConfig::RedisCredentials));
        }
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
        connection: RedisConnectionConfig {
            host: required_string(&resolved, "host")?,
            port: u16::try_from(required_i64(&resolved, "port")?)
                .map_err(|_| PyValueError::new_err("invalid Redis port"))?,
            database: optional_i64(&resolved, "db")?.unwrap_or(0),
            username: optional_dict_string(&resolved, "username")?,
            password: optional_dict_string(&resolved, "password")?,
            protocol,
            pool_size: pool.getattr("max_connections")?.extract::<usize>()?,
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
    let mut current = value.clone();
    for name in names {
        match optional_attribute(&current, name)? {
            Some(next) => current = next,
            None => return Ok(None),
        }
    }
    Ok(Some(current))
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

    use super::{
        CacheBackendConfig, CacheConfigProjection, CertificateRequirement, NativeCacheConfig,
        RedisProtocol,
    };
    use crate::cache::native::NativeResponseCache;

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

    fn s3_facade<'py>(py: Python<'py>, body: &str) -> Bound<'py, PyAny> {
        let locals = PyDict::new(py);
        py.run(
            &CString::new(format!(
                "from types import SimpleNamespace\n\
                 S3Client = type('S3', (), {{'__module__': 'botocore.client'}})\n\
                 client = S3Client()\n\
                 client.meta = SimpleNamespace(region_name='us-east-1', endpoint_url='https://example.test', config=SimpleNamespace(s3=None, proxies=None, client_cert=None, signature_version='s3v4'))\n\
                 client._endpoint = SimpleNamespace(http_session=SimpleNamespace(_verify=True))\n\
                 client._request_signer = SimpleNamespace(_credentials=SimpleNamespace(method='explicit', access_key='key', secret_key='secret', token='token'))\n\
                 backend = SimpleNamespace(bucket_name='bucket', key_prefix='team/', s3_client=client)\n\
                 facade = SimpleNamespace(type='s3', mode='default-on', ttl=None, namespace=None, supported_call_types=[], redis_flush_size=None, semantic_cache_scope='key', cache=backend)\n\
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
    fn projects_s3_configuration_with_explicit_credentials_and_custom_endpoint() {
        Python::initialize();
        Python::attach(|py| {
            let facade = s3_facade(py, "");
            let CacheConfigProjection::Native(config) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("S3 cache should be supported");
            };
            let CacheBackendConfig::S3(s3) = config.backend else {
                panic!("expected S3 configuration");
            };
            assert_eq!(s3.bucket, "bucket");
            assert_eq!(s3.key_prefix, "team/");
            assert_eq!(s3.region, "us-east-1");
            assert_eq!(
                s3.endpoint.map(|endpoint| endpoint.url).as_deref(),
                Some("https://example.test")
            );
            assert_eq!(s3.auth.access_key_id.as_deref(), Some("key"));
            assert_eq!(s3.auth.secret_access_key.as_deref(), Some("secret"));
            assert_eq!(s3.auth.session_token.as_deref(), Some("token"));
            assert_eq!(s3.auth.region_name.as_deref(), Some("us-east-1"));
        });
    }

    #[test]
    fn default_s3_endpoint_projects_no_custom_endpoint() {
        Python::initialize();
        Python::attach(|py| {
            let facade = s3_facade(
                py,
                "facade.cache.s3_client.meta.endpoint_url = 'https://s3.us-east-1.amazonaws.com'",
            );
            let CacheConfigProjection::Native(config) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("S3 cache should be supported");
            };
            let CacheBackendConfig::S3(s3) = config.backend else {
                panic!("expected S3 configuration");
            };
            assert!(s3.endpoint.is_none());
        });
    }

    #[test]
    fn non_sigv4_proxies_and_disabled_verification_stay_on_python() {
        Python::initialize();
        Python::attach(|py| {
            for (body, message) in [
                (
                    "facade.cache.s3_client.meta.config.signature_version = 's3'",
                    "native S3 configuration requires Python",
                ),
                (
                    "facade.cache.s3_client.meta.config.proxies = {'https': 'proxy'}",
                    "native S3 configuration requires Python",
                ),
                (
                    "facade.cache.s3_client._endpoint.http_session._verify = False",
                    "native S3 configuration requires Python",
                ),
                (
                    "del facade.cache.s3_client._endpoint.http_session._verify",
                    "native S3 configuration requires Python",
                ),
            ] {
                let facade = s3_facade(py, body);
                let CacheConfigProjection::Unsupported(reason) =
                    NativeCacheConfig::project(&facade).unwrap()
                else {
                    panic!("{body} must stay on Python");
                };
                assert_eq!(reason.message(), message);
            }
            let facade = s3_facade(py, "facade.cache.s3_client = SimpleNamespace()");
            let CacheConfigProjection::Unsupported(reason) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("non-botocore client must stay on Python");
            };
            assert_eq!(reason.message(), "native S3 client type is not implemented");
            let facade = s3_facade(
                py,
                "facade.cache.s3_client._request_signer._credentials = None",
            );
            let CacheConfigProjection::Unsupported(reason) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("missing credentials must stay on Python");
            };
            assert_eq!(reason.message(), "native S3 credentials require Python");
        });
    }

    #[test]
    fn non_explicit_s3_credentials_use_the_default_chain() {
        Python::initialize();
        Python::attach(|py| {
            let facade = s3_facade(
                py,
                "facade.cache.s3_client._request_signer._credentials = SimpleNamespace(method='sso', access_key=None, secret_key=None, token=None)",
            );
            let CacheConfigProjection::Native(config) =
                NativeCacheConfig::project(&facade).unwrap()
            else {
                panic!("default-chain credentials should be supported");
            };
            let CacheBackendConfig::S3(s3) = config.backend else {
                panic!("expected S3 configuration");
            };
            assert_eq!(s3.auth.access_key_id, None);
            assert_eq!(s3.auth.secret_access_key, None);
            assert_eq!(s3.auth.region_name.as_deref(), Some("us-east-1"));
        });
    }
}
