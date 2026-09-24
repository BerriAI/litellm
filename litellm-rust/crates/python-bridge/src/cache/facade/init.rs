//! `Cache.__init__`.

use pyo3::{
    PyTypeInfo,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyBool, PyDict, PyList, PyString, PyTuple},
};

use super::{Cache, NativeStorage};
use crate::cache::{binding::ResolvedCache, guard::FacadeGuard};

const PARAMETERS: [&str; 45] = [
    "type",
    "mode",
    "host",
    "port",
    "password",
    "namespace",
    "ttl",
    "default_in_memory_ttl",
    "default_in_redis_ttl",
    "similarity_threshold",
    "supported_call_types",
    "azure_account_url",
    "azure_blob_container",
    "s3_bucket_name",
    "s3_region_name",
    "s3_api_version",
    "s3_use_ssl",
    "s3_verify",
    "s3_endpoint_url",
    "s3_aws_access_key_id",
    "s3_aws_secret_access_key",
    "s3_aws_session_token",
    "s3_config",
    "s3_path",
    "gcs_bucket_name",
    "gcs_path_service_account",
    "gcs_path",
    "redis_semantic_cache_embedding_model",
    "redis_semantic_cache_index_name",
    "valkey_semantic_cache_embedding_model",
    "valkey_semantic_cache_index_name",
    "redis_flush_size",
    "redis_startup_nodes",
    "disk_cache_dir",
    "qdrant_api_base",
    "qdrant_api_key",
    "qdrant_collection_name",
    "qdrant_quantization_config",
    "qdrant_semantic_cache_embedding_model",
    "qdrant_semantic_cache_vector_size",
    "semantic_cache_embedding_max_input_tokens",
    "semantic_cache_embedding_timeout",
    "semantic_cache_scope",
    "gcp_service_account",
    "gcp_ssl_ca_certs",
];

const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-ada-002";

struct Arguments<'py> {
    py: Python<'py>,
    named: Bound<'py, PyDict>,
    extras: Bound<'py, PyDict>,
}

impl<'py> Arguments<'py> {
    fn parse(
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Self> {
        if args.len() > PARAMETERS.len() {
            return Err(PyTypeError::new_err(format!(
                "Cache.__init__() takes at most {} positional arguments ({} given)",
                PARAMETERS.len(),
                args.len()
            )));
        }
        let named = PyDict::new(py);
        let extras = PyDict::new(py);
        for (name, value) in PARAMETERS.iter().zip(args.iter()) {
            named.set_item(name, value)?;
        }
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let key = key.cast_into::<PyString>()?;
                let name = key.to_str()?;
                if !PARAMETERS.contains(&name) {
                    extras.set_item(&key, value)?;
                    continue;
                }
                if named.contains(&key)? {
                    return Err(PyTypeError::new_err(format!(
                        "Cache.__init__() got multiple values for argument '{name}'"
                    )));
                }
                named.set_item(&key, value)?;
            }
        }
        Ok(Self { py, named, extras })
    }

    fn value(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        Ok(self
            .named
            .get_item(name)?
            .unwrap_or_else(|| self.py.None().into_bound(self.py)))
    }

    fn given(&self, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(self.named.get_item(name)?.filter(|value| !value.is_none()))
    }

    fn string_or(&self, name: &str, default: &str) -> PyResult<Bound<'py, PyAny>> {
        Ok(self
            .named
            .get_item(name)?
            .unwrap_or_else(|| PyString::new(self.py, default).into_any()))
    }

    fn kwargs(&self, entries: &[(&str, Bound<'py, PyAny>)]) -> PyResult<Bound<'py, PyDict>> {
        let kwargs = PyDict::new(self.py);
        for (name, value) in entries {
            kwargs.set_item(name, value)?;
        }
        Ok(kwargs)
    }

    fn kwargs_with_extras(
        &self,
        entries: &[(&str, Bound<'py, PyAny>)],
    ) -> PyResult<Bound<'py, PyDict>> {
        let kwargs = self.kwargs(entries)?;
        kwargs.update(self.extras.as_mapping())?;
        Ok(kwargs)
    }
}

fn backend_class<'py>(py: Python<'py>, module: &str, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import(module)?.getattr(name)
}

fn construct<'py>(
    py: Python<'py>,
    module: &str,
    name: &str,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    backend_class(py, module, name)?.call((), Some(kwargs))
}

fn redis_backend<'py>(arguments: &Arguments<'py>) -> PyResult<Bound<'py, PyAny>> {
    let py = arguments.py;
    let mut startup_nodes = arguments.value("redis_startup_nodes")?;
    if !startup_nodes.is_truthy()? {
        let configured = py
            .import("litellm")?
            .getattr("get_secret")?
            .call1(("REDIS_CLUSTER_NODES",))?;
        if configured.is_instance_of::<PyString>() {
            startup_nodes = py.import("json")?.call_method1("loads", (configured,))?;
        }
    }
    if !startup_nodes.is_truthy()? {
        let kwargs = arguments.kwargs_with_extras(&[
            ("host", arguments.value("host")?),
            ("port", arguments.value("port")?),
            ("password", arguments.value("password")?),
            ("redis_flush_size", arguments.value("redis_flush_size")?),
        ])?;
        return construct(py, "litellm.caching.caching", "RedisCache", &kwargs);
    }
    let kwargs = arguments.kwargs_with_extras(&[
        ("host", arguments.value("host")?),
        ("port", arguments.value("port")?),
        ("password", arguments.value("password")?),
        ("redis_flush_size", arguments.value("redis_flush_size")?),
        ("startup_nodes", startup_nodes),
    ])?;
    for name in ["gcp_service_account", "gcp_ssl_ca_certs"] {
        if let Some(value) = arguments.given(name)? {
            kwargs.set_item(name, value)?;
        }
    }
    construct(py, "litellm.caching.caching", "RedisClusterCache", &kwargs)
}

fn backend_for<'py>(
    arguments: &Arguments<'py>,
    cache_type: Option<&str>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = arguments.py;
    let backend = match cache_type {
        Some("redis") => redis_backend(arguments)?,
        Some("redis-semantic") => construct(
            py,
            "litellm.caching.caching",
            "RedisSemanticCache",
            &arguments.kwargs_with_extras(&[
                ("host", arguments.value("host")?),
                ("port", arguments.value("port")?),
                ("password", arguments.value("password")?),
                (
                    "similarity_threshold",
                    arguments.value("similarity_threshold")?,
                ),
                (
                    "embedding_model",
                    arguments.string_or(
                        "redis_semantic_cache_embedding_model",
                        DEFAULT_EMBEDDING_MODEL,
                    )?,
                ),
                (
                    "index_name",
                    arguments.value("redis_semantic_cache_index_name")?,
                ),
                (
                    "embedding_max_input_tokens",
                    arguments.value("semantic_cache_embedding_max_input_tokens")?,
                ),
                (
                    "embedding_timeout",
                    arguments.value("semantic_cache_embedding_timeout")?,
                ),
            ])?,
        )?,
        Some("valkey-semantic") => construct(
            py,
            "litellm.caching.valkey_semantic_cache",
            "ValkeySemanticCache",
            &arguments.kwargs_with_extras(&[
                ("host", arguments.value("host")?),
                ("port", arguments.value("port")?),
                ("password", arguments.value("password")?),
                (
                    "similarity_threshold",
                    arguments.value("similarity_threshold")?,
                ),
                (
                    "embedding_model",
                    arguments.string_or(
                        "valkey_semantic_cache_embedding_model",
                        DEFAULT_EMBEDDING_MODEL,
                    )?,
                ),
                (
                    "index_name",
                    arguments.value("valkey_semantic_cache_index_name")?,
                ),
                ("startup_nodes", arguments.value("redis_startup_nodes")?),
                (
                    "embedding_max_input_tokens",
                    arguments.value("semantic_cache_embedding_max_input_tokens")?,
                ),
                (
                    "embedding_timeout",
                    arguments.value("semantic_cache_embedding_timeout")?,
                ),
            ])?,
        )?,
        Some("qdrant-semantic") => construct(
            py,
            "litellm.caching.caching",
            "QdrantSemanticCache",
            &arguments.kwargs(&[
                ("qdrant_api_base", arguments.value("qdrant_api_base")?),
                ("qdrant_api_key", arguments.value("qdrant_api_key")?),
                (
                    "collection_name",
                    arguments.value("qdrant_collection_name")?,
                ),
                (
                    "similarity_threshold",
                    arguments.value("similarity_threshold")?,
                ),
                (
                    "quantization_config",
                    arguments.value("qdrant_quantization_config")?,
                ),
                (
                    "embedding_model",
                    arguments.string_or(
                        "qdrant_semantic_cache_embedding_model",
                        DEFAULT_EMBEDDING_MODEL,
                    )?,
                ),
                (
                    "vector_size",
                    arguments.value("qdrant_semantic_cache_vector_size")?,
                ),
                (
                    "embedding_max_input_tokens",
                    arguments.value("semantic_cache_embedding_max_input_tokens")?,
                ),
                (
                    "embedding_timeout",
                    arguments.value("semantic_cache_embedding_timeout")?,
                ),
            ])?,
        )?,
        Some("local") => backend_class(py, "litellm.caching.caching", "InMemoryCache")?.call0()?,
        Some("s3") => construct(
            py,
            "litellm.caching.caching",
            "S3Cache",
            &arguments.kwargs_with_extras(&[
                ("s3_bucket_name", arguments.value("s3_bucket_name")?),
                ("s3_region_name", arguments.value("s3_region_name")?),
                ("s3_api_version", arguments.value("s3_api_version")?),
                (
                    "s3_use_ssl",
                    arguments
                        .named
                        .get_item("s3_use_ssl")?
                        .unwrap_or_else(|| PyBool::new(py, true).to_owned().into_any()),
                ),
                ("s3_verify", arguments.value("s3_verify")?),
                ("s3_endpoint_url", arguments.value("s3_endpoint_url")?),
                (
                    "s3_aws_access_key_id",
                    arguments.value("s3_aws_access_key_id")?,
                ),
                (
                    "s3_aws_secret_access_key",
                    arguments.value("s3_aws_secret_access_key")?,
                ),
                (
                    "s3_aws_session_token",
                    arguments.value("s3_aws_session_token")?,
                ),
                ("s3_config", arguments.value("s3_config")?),
                ("s3_path", arguments.value("s3_path")?),
            ])?,
        )?,
        Some("gcs") => construct(
            py,
            "litellm.caching.caching",
            "GCSCache",
            &arguments.kwargs(&[
                ("bucket_name", arguments.value("gcs_bucket_name")?),
                (
                    "path_service_account",
                    arguments.value("gcs_path_service_account")?,
                ),
                ("gcs_path", arguments.value("gcs_path")?),
            ])?,
        )?,
        Some("azure-blob") => construct(
            py,
            "litellm.caching.caching",
            "AzureBlobCache",
            &arguments.kwargs(&[
                ("account_url", arguments.value("azure_account_url")?),
                ("container", arguments.value("azure_blob_container")?),
            ])?,
        )?,
        Some("disk") => construct(
            py,
            "litellm.caching.caching",
            "DiskCache",
            &arguments.kwargs(&[("disk_cache_dir", arguments.value("disk_cache_dir")?)])?,
        )?,
        _ => return Ok(None),
    };
    Ok(Some(backend))
}

fn register_cache_callbacks(py: Python<'_>) -> PyResult<()> {
    let litellm = py.import("litellm")?;
    let input_callback = litellm.getattr("input_callback")?;
    if !input_callback.contains("cache")? {
        input_callback.call_method1("append", ("cache",))?;
    }
    let manager = litellm.getattr("logging_callback_manager")?;
    if !litellm.getattr("success_callback")?.contains("cache")? {
        manager.call_method1("add_litellm_success_callback", ("cache",))?;
    }
    if !litellm
        .getattr("_async_success_callback")?
        .contains("cache")?
    {
        manager.call_method1("add_litellm_async_success_callback", ("cache",))?;
    }
    Ok(())
}

pub(super) fn initialize(
    slf: &Bound<'_, Cache>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let py = slf.py();
    let arguments = Arguments::parse(py, args, kwargs)?;
    let caching_types = py.import("litellm.types.caching")?;
    let cache_type = match arguments.given("type")? {
        Some(value) => value,
        None => caching_types
            .getattr("LiteLLMCacheType")?
            .getattr("LOCAL")?,
    };
    let type_name = cache_type.extract::<String>().ok();
    if let Some(backend) = backend_for(&arguments, type_name.as_deref())? {
        slf.get().set_backend(backend.unbind());
    }
    register_cache_callbacks(py)?;

    let supported_call_types = match arguments.given("supported_call_types")? {
        Some(value) => value,
        None => PyList::type_object(py)
            .call1((caching_types.getattr("DEFAULT_CACHING_SUPPORTED_CALL_TYPES")?,))?,
    };
    let mode = match arguments
        .given("mode")?
        .filter(|mode| mode.is_truthy().unwrap_or(false))
    {
        Some(mode) => mode,
        None => py
            .import("litellm.caching.caching")?
            .getattr("CacheMode")?
            .getattr("default_on")?,
    };
    let scope = caching_types
        .getattr("SemanticCacheScope")?
        .call1((arguments.string_or("semantic_cache_scope", "key")?,))?
        .getattr("value")?;
    let namespace = arguments.value("namespace")?;
    slf.setattr("supported_call_types", supported_call_types)?;
    slf.setattr("type", &cache_type)?;
    slf.setattr("namespace", &namespace)?;
    slf.setattr("redis_flush_size", arguments.value("redis_flush_size")?)?;
    slf.setattr("ttl", arguments.value("ttl")?)?;
    slf.setattr("mode", mode)?;
    slf.setattr("semantic_cache_scope", scope)?;

    let in_memory_ttl = arguments.given("default_in_memory_ttl")?;
    let redis_ttl = arguments.given("default_in_redis_ttl")?;
    match type_name.as_deref() {
        Some("local") => {
            if let Some(ttl) = in_memory_ttl {
                slf.setattr("ttl", ttl)?;
            }
        }
        Some("redis" | "redis-semantic" | "valkey-semantic") => {
            if let Some(ttl) = redis_ttl {
                slf.setattr("ttl", ttl)?;
            }
        }
        _ => {}
    }
    if !namespace.is_none()
        && let Ok(backend) = slf.get().backend(py)
        && backend.bind(py).is_instance(&backend_class(
            py,
            "litellm.caching.caching",
            "RedisCache",
        )?)?
    {
        backend.bind(py).setattr("namespace", namespace)?;
    }
    resolve_native(slf)
}

pub(super) fn resolve_native(slf: &Bound<'_, Cache>) -> PyResult<()> {
    let py = slf.py();
    if slf.get().backend(py).is_err() {
        return Ok(());
    }
    let resolved = py
        .import("litellm.rust_bridge.response_cache")?
        .getattr("resolve_response_cache")?
        .call1((slf,))?;
    if resolved.is_none() {
        return Ok(());
    }
    let runtime = resolved.cast_into::<ResolvedCache>()?;
    let Some(service) = runtime.get().native_service()? else {
        return Ok(());
    };
    let guard = FacadeGuard::capture(py, slf.as_any(), &service)?;
    slf.get()
        .bind_native(py, NativeStorage::new(service, guard))
}
