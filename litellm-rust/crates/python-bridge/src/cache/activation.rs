use crate::logger::run_sync_value;
use litellm_cache_gcs::{DEFAULT_ENDPOINT, GcsConfig};
use litellm_cache_redis_semantic::RedisSemanticConfig;
use litellm_host_python::release_gil;
use litellm_http::ClientVariant;
use pyo3::prelude::*;

use super::{
    cache_error,
    config::{CacheBackendConfig, NativeCacheConfig, UnsupportedCacheConfig},
    embedder::PythonEmbedder,
    native::NativeResponseCache,
};
use crate::errors::RustBridgeDeclined;
use crate::http::host_client;

fn declined(reason: UnsupportedCacheConfig) -> PyErr {
    RustBridgeDeclined::new_err(reason.message())
}

/// Builds the native backend a `Cache` facade's projected configuration describes. `backend` is
/// the facade's `.cache` object, which owns embedding for the Python-embedded semantic caches.
pub(super) fn activate(
    py: Python<'_>,
    backend: &Bound<'_, PyAny>,
    config: NativeCacheConfig,
) -> PyResult<NativeResponseCache> {
    let policy = config.policy;
    let service = match config.backend {
        CacheBackendConfig::Memory(memory) => {
            NativeResponseCache::memory(memory.capacity, memory.default_ttl, memory.max_entry_bytes)
        }
        CacheBackendConfig::Redis(redis) => {
            let url = redis.connection.native_url().map_err(declined)?;
            let flush_size = policy.redis_flush_size.map(|_| redis.flush_size);
            release_gil(py, move || {
                NativeResponseCache::redis(
                    &url,
                    &redis.topology,
                    Some(redis.default_ttl),
                    redis.namespace,
                )
            })
            .map_err(cache_error)?
            .with_redis_flush_size(flush_size)
        }
        CacheBackendConfig::S3(s3) => {
            let http = host_client(py, ClientVariant::NoRedirect)?;
            run_sync_value(
                py,
                async move { Ok(NativeResponseCache::s3(*s3, http).await) },
            )?
        }
        CacheBackendConfig::Gcs(gcs) => NativeResponseCache::gcs(
            GcsConfig {
                bucket_name: gcs.bucket_name,
                gcs_path: Some(gcs.key_prefix),
                path_service_account: gcs.path_service_account,
                endpoint: DEFAULT_ENDPOINT.to_owned(),
            },
            host_client(py, ClientVariant::NoRedirect)?,
            None,
        ),
        CacheBackendConfig::Disk(disk) => {
            release_gil(py, move || NativeResponseCache::disk(&disk.directory))
                .map_err(cache_error)?
        }
        CacheBackendConfig::AzureBlob(azure) => {
            let http = host_client(py, ClientVariant::NoRedirect)?;
            run_sync_value(py, async move {
                NativeResponseCache::azure_blob(&azure.account_url, &azure.container, http)
                    .await
                    .map_err(cache_error)
            })?
        }
        CacheBackendConfig::RedisSemantic(semantic) => {
            let url = semantic.native_url().map_err(declined)?.to_owned();
            let embedder = PythonEmbedder::new(backend.clone().unbind());
            let semantic_config = RedisSemanticConfig {
                index_name: semantic.index_name,
                similarity_threshold: semantic.similarity_threshold as f32,
            };
            release_gil(py, move || {
                NativeResponseCache::redis_semantic(&url, embedder, semantic_config)
            })
            .map_err(cache_error)?
        }
        CacheBackendConfig::ValkeySemantic(valkey) => {
            let url = valkey.connection.native_url().map_err(declined)?;
            let embedder = PythonEmbedder::new(backend.clone().unbind());
            release_gil(py, move || {
                NativeResponseCache::valkey_semantic(
                    &url,
                    valkey.similarity_threshold,
                    valkey.index_name,
                    embedder,
                )
            })
            .map_err(cache_error)?
        }
        CacheBackendConfig::QdrantSemantic(qdrant) => {
            let client = host_client(py, ClientVariant::Provider)?;
            run_sync_value(py, async move {
                let runtime = tokio::runtime::Handle::current();
                NativeResponseCache::qdrant_semantic(*qdrant, client, runtime)
                    .await
                    .map_err(cache_error)
            })?
        }
    };
    Ok(service.with_scope(policy.semantic_cache_scope))
}
