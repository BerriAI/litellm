use serde::{Deserialize, Deserializer};

use litellm_auth_types::SecretValue;

use crate::{Object, Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct CachingSettings {
    pub cache: Option<bool>,
    pub cache_params: Option<CacheParams>,
    pub enable_redis_auth_cache: Option<Spelled<bool>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct CacheParams {
    #[serde(rename = "type")]
    pub cache_type: Option<String>,
    pub mode: Option<String>,
    pub namespace: Option<String>,
    pub ttl: Option<Spelled<f64>>,
    pub supported_call_types: Option<Box<[String]>>,
    #[serde(deserialize_with = "present")]
    pub max_messages: Option<Option<Spelled<i64>>>,
    #[serde(flatten)]
    pub redis_server: RedisServerCacheParams,
    #[serde(flatten)]
    pub semantic: SemanticCacheParams,
    #[serde(flatten)]
    pub proxy_ttl: ProxyCacheTtlParams,
    #[serde(flatten)]
    pub redis: RedisCacheParams,
    #[serde(flatten)]
    pub valkey: ValkeyCacheParams,
    #[serde(flatten)]
    pub s3: S3CacheParams,
    #[serde(flatten)]
    pub gcs: GcsCacheParams,
    #[serde(flatten)]
    pub azure_blob: AzureBlobCacheParams,
    #[serde(flatten)]
    pub qdrant: QdrantCacheParams,
    #[serde(flatten)]
    pub disk: DiskCacheParams,
    #[serde(flatten)]
    pub additional_fields: Object,
}

/// Read by every backend that talks to a Redis-protocol server: `redis`, `redis-semantic` and `valkey-semantic`.
#[macro_rules_attribute::apply(crate::config_section)]
pub struct RedisServerCacheParams {
    pub host: Option<String>,
    pub port: Option<Spelled<i64>>,
    pub password: Option<SecretValue>,
    pub redis_startup_nodes: Option<Raw>,
    pub default_in_redis_ttl: Option<Spelled<f64>>,
}

/// Read by the semantic backends: `redis-semantic`, `valkey-semantic` and `qdrant-semantic`.
#[macro_rules_attribute::apply(crate::config_section)]
pub struct SemanticCacheParams {
    pub similarity_threshold: Option<Spelled<f64>>,
    pub semantic_cache_embedding_max_input_tokens: Option<Spelled<i64>>,
    pub semantic_cache_embedding_timeout: Option<Spelled<f64>>,
    pub semantic_cache_scope: Option<String>,
}

/// Not response-cache settings. The proxy copies them onto `litellm.default_*_ttl` for its own internal caches.
/// `default_in_memory_ttl` also replaces `ttl` when `type` is `local`.
#[macro_rules_attribute::apply(crate::config_section)]
pub struct ProxyCacheTtlParams {
    pub default_in_memory_ttl: Option<Spelled<f64>>,
    pub default_redis_ttl: Option<Spelled<f64>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct RedisCacheParams {
    pub redis_semantic_cache_embedding_model: Option<String>,
    pub redis_semantic_cache_index_name: Option<String>,
    pub redis_flush_size: Option<Spelled<i64>>,
    pub gcp_service_account: Option<String>,
    pub gcp_ssl_ca_certs: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct ValkeyCacheParams {
    pub valkey_semantic_cache_embedding_model: Option<String>,
    pub valkey_semantic_cache_index_name: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct S3CacheParams {
    pub s3_bucket_name: Option<String>,
    pub s3_region_name: Option<String>,
    pub s3_api_version: Option<String>,
    pub s3_use_ssl: Option<Spelled<bool>>,
    pub s3_verify: Option<Spelled<bool>>,
    pub s3_endpoint_url: Option<SecretValue>,
    pub s3_aws_access_key_id: Option<SecretValue>,
    pub s3_aws_secret_access_key: Option<SecretValue>,
    pub s3_aws_session_token: Option<SecretValue>,
    pub s3_config: Option<Raw>,
    pub s3_path: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct GcsCacheParams {
    pub gcs_bucket_name: Option<String>,
    pub gcs_path_service_account: Option<String>,
    pub gcs_path: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct AzureBlobCacheParams {
    pub azure_account_url: Option<SecretValue>,
    pub azure_blob_container: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct QdrantCacheParams {
    pub qdrant_api_base: Option<SecretValue>,
    pub qdrant_api_key: Option<SecretValue>,
    pub qdrant_collection_name: Option<String>,
    pub qdrant_quantization_config: Option<String>,
    pub qdrant_semantic_cache_embedding_model: Option<String>,
    pub qdrant_semantic_cache_vector_size: Option<Spelled<i64>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct DiskCacheParams {
    pub disk_cache_dir: Option<String>,
}

fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<Option<T>>, D::Error> {
    Ok(Some(Option::deserialize(deserializer)?))
}
