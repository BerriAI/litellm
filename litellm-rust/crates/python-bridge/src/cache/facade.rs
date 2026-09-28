use litellm_cache_redis::RedisTopology;
use litellm_host_python::from_py;
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyTuple, PyType},
};
use serde_json::Value;

use super::{
    config::{CacheConfigProjection, NativeCacheConfig},
    handle::CacheTestHandle,
    identity::BackendIdentity,
    native::NativeResponseCache,
};

struct ClassGuard {
    class: Py<PyType>,
    attributes: Vec<(String, Py<PyAny>)>,
}

struct ObjectGuard {
    reference: Py<PyAny>,
    classes: Vec<ClassGuard>,
    config_names: &'static [&'static str],
    config: Vec<Value>,
}

struct RedisPoolGuard {
    reference: Py<PyAny>,
    connection_class: Py<PyAny>,
    connection_kwargs: Py<PyAny>,
    max_connections: Option<usize>,
    client_name: &'static str,
    attributes: RedisPoolAttributes,
}

struct DiskStoreGuard {
    reference: Py<PyAny>,
    directory: String,
}

struct AzureBlobClientGuard {
    sync_client: Py<PyAny>,
    async_client: Py<PyAny>,
    url: String,
    container_name: String,
}

struct S3ClientGuard {
    reference: Py<PyAny>,
}

enum ConnectionGuard {
    None,
    RedisPool(RedisPoolGuard),
    AzureBlob(AzureBlobClientGuard),
    S3(S3ClientGuard),
}

#[derive(Clone, Copy)]
struct RedisPoolAttributes {
    pool: &'static str,
    connection_class: &'static str,
    max_connections: Option<&'static str>,
}

const STANDALONE_POOL: RedisPoolAttributes = RedisPoolAttributes {
    pool: "connection_pool",
    connection_class: "connection_class",
    max_connections: Some("max_connections"),
};

const CLUSTER_POOL: RedisPoolAttributes = RedisPoolAttributes {
    pool: "nodes_manager",
    connection_class: "connection_pool_class",
    max_connections: None,
};

const VALKEY_POOL: RedisPoolAttributes = STANDALONE_POOL;

/// Class-level defaults an instance overwrites with its own state rather than behavior:
/// `Cache._native_cache` holds the runtime `Cache.__init__` resolved.
const INSTANCE_STATE: &[&str] = &["_native_cache"];

pub(super) struct FacadeGuard {
    outer: ObjectGuard,
    backend: ObjectGuard,
    disk_store: Option<DiskStoreGuard>,
    connection: ConnectionGuard,
}

impl ObjectGuard {
    fn capture(
        py: Python<'_>,
        object: &Bound<'_, PyAny>,
        config_names: &'static [&'static str],
    ) -> PyResult<Self> {
        let classes = object
            .get_type()
            .getattr("__mro__")?
            .cast_into::<PyTuple>()?
            .iter()
            .map(|class| {
                let class = class.cast_into::<PyType>()?;
                let attributes = class
                    .getattr("__dict__")?
                    .call_method0("items")?
                    .try_iter()?
                    .map(|item| item?.extract::<(String, Py<PyAny>)>())
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(ClassGuard {
                    class: class.unbind(),
                    attributes,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let guard = Self {
            reference: py
                .import("weakref")?
                .getattr("ref")?
                .call1((object,))?
                .unbind(),
            classes,
            config_names,
            config: Self::config(object, config_names)?,
        };
        if !guard.matches(py, object)? {
            return Err(PyTypeError::new_err(
                "native facade registration requires unmodified built-in methods",
            ));
        }
        Ok(guard)
    }

    fn config(object: &Bound<'_, PyAny>, names: &[&str]) -> PyResult<Vec<Value>> {
        names
            .iter()
            .map(|name| match object.getattr(*name) {
                Ok(value) => from_py(&value),
                Err(error)
                    if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(object.py()) =>
                {
                    Ok(Value::Null)
                }
                Err(error) => Err(error),
            })
            .collect()
    }

    fn matches(&self, py: Python<'_>, object: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !self.reference.bind(py).call0()?.is(object) {
            return Ok(false);
        }
        let mro = object
            .get_type()
            .getattr("__mro__")?
            .cast_into::<PyTuple>()?;
        if mro.len() != self.classes.len() {
            return Ok(false);
        }
        let instance = object.getattr("__dict__")?.cast_into::<PyDict>()?;
        for (class, expected) in mro.iter().zip(&self.classes) {
            if !class.is(expected.class.bind(py)) {
                return Ok(false);
            }
            let attributes = class.getattr("__dict__")?;
            if attributes.len()? != expected.attributes.len() {
                return Ok(false);
            }
            for (name, value) in &expected.attributes {
                if (instance.contains(name)?
                    && !self.config_names.contains(&name.as_str())
                    && !INSTANCE_STATE.contains(&name.as_str()))
                    || !attributes.get_item(name)?.is(value.bind(py))
                {
                    return Ok(false);
                }
            }
        }
        Ok(Self::config(object, self.config_names)? == self.config)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reference)?;
        for class in &self.classes {
            visit.call(&class.class)?;
            for (_, value) in &class.attributes {
                visit.call(value)?;
            }
        }
        Ok(())
    }
}

impl RedisPoolGuard {
    fn capture(
        backend: &Bound<'_, PyAny>,
        client_name: &'static str,
        attributes: RedisPoolAttributes,
    ) -> PyResult<Self> {
        let pool = backend.getattr(client_name)?.getattr(attributes.pool)?;
        Ok(Self {
            reference: pool.clone().unbind(),
            connection_class: pool.getattr(attributes.connection_class)?.unbind(),
            connection_kwargs: pool
                .getattr("connection_kwargs")?
                .call_method0("copy")?
                .unbind(),
            max_connections: attributes
                .max_connections
                .map(|name| pool.getattr(name)?.extract::<usize>())
                .transpose()?,
            client_name,
            attributes,
        })
    }

    fn matches(&self, py: Python<'_>, backend: &Bound<'_, PyAny>) -> PyResult<bool> {
        let pool = backend
            .getattr(self.client_name)?
            .getattr(self.attributes.pool)?;
        Ok(self.reference.bind(py).is(&pool)
            && self
                .connection_class
                .bind(py)
                .is(&pool.getattr(self.attributes.connection_class)?)
            && self.max_connections
                == self
                    .attributes
                    .max_connections
                    .map(|name| pool.getattr(name)?.extract::<usize>())
                    .transpose()?
            && self
                .connection_kwargs
                .bind(py)
                .eq(pool.getattr("connection_kwargs")?)?)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reference)?;
        visit.call(&self.connection_class)?;
        visit.call(&self.connection_kwargs)
    }
}

impl DiskStoreGuard {
    fn capture(backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        let store = backend.getattr("disk_cache")?;
        Ok(Self {
            reference: store.clone().unbind(),
            directory: store.getattr("directory")?.extract()?,
        })
    }

    fn matches(&self, py: Python<'_>, backend: &Bound<'_, PyAny>) -> PyResult<bool> {
        let store = backend.getattr("disk_cache")?;
        Ok(self.reference.bind(py).is(&store)
            && self.directory == store.getattr("directory")?.extract::<String>()?)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reference)
    }
}

impl AzureBlobClientGuard {
    fn capture(backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        let sync_client = backend.getattr("container_client")?;
        Ok(Self {
            url: sync_client.getattr("url")?.extract::<String>()?,
            container_name: sync_client.getattr("container_name")?.extract::<String>()?,
            sync_client: sync_client.unbind(),
            async_client: backend.getattr("async_container_client")?.unbind(),
        })
    }

    fn matches(&self, py: Python<'_>, backend: &Bound<'_, PyAny>) -> PyResult<bool> {
        let sync_client = backend.getattr("container_client")?;
        Ok(self.sync_client.bind(py).is(&sync_client)
            && self
                .async_client
                .bind(py)
                .is(&backend.getattr("async_container_client")?)
            && self.url == sync_client.getattr("url")?.extract::<String>()?
            && self.container_name == sync_client.getattr("container_name")?.extract::<String>()?)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.sync_client)?;
        visit.call(&self.async_client)
    }
}

impl S3ClientGuard {
    fn capture(backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            reference: backend.getattr("s3_client")?.unbind(),
        })
    }

    fn matches(&self, py: Python<'_>, backend: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.reference.bind(py).is(&backend.getattr("s3_client")?))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reference)
    }
}

impl ConnectionGuard {
    fn capture(kind: &str, cluster: bool, backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(match (kind, cluster) {
            ("redis", false) => Self::RedisPool(RedisPoolGuard::capture(
                backend,
                "redis_client",
                STANDALONE_POOL,
            )?),
            ("redis", true) => Self::RedisPool(RedisPoolGuard::capture(
                backend,
                "redis_client",
                CLUSTER_POOL,
            )?),
            ("valkey-semantic", _) => Self::RedisPool(RedisPoolGuard::capture(
                backend,
                "sync_client",
                VALKEY_POOL,
            )?),
            ("disk", _) => Self::None,
            ("azure-blob", _) => Self::AzureBlob(AzureBlobClientGuard::capture(backend)?),
            ("s3", _) => Self::S3(S3ClientGuard::capture(backend)?),
            _ => Self::None,
        })
    }

    fn matches(&self, py: Python<'_>, backend: &Bound<'_, PyAny>) -> PyResult<bool> {
        match self {
            Self::None => Ok(true),
            Self::RedisPool(guard) => guard.matches(py, backend),
            Self::AzureBlob(guard) => guard.matches(py, backend),
            Self::S3(guard) => guard.matches(py, backend),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::None => Ok(()),
            Self::RedisPool(guard) => guard.traverse(visit),
            Self::AzureBlob(guard) => guard.traverse(visit),
            Self::S3(guard) => guard.traverse(visit),
        }
    }
}

impl FacadeGuard {
    pub(super) fn capture(
        py: Python<'_>,
        facade: &Bound<'_, PyAny>,
        service: &NativeResponseCache,
    ) -> PyResult<Self> {
        let identity = service.identity();
        let kind = identity.kind();
        let cache_type = py.import("litellm.caching.caching")?.getattr("Cache")?;
        if !facade.get_type().is(&cache_type) {
            return Err(PyTypeError::new_err(
                "only exact built-in Cache facades can be registered",
            ));
        }
        let cluster = matches!(
            identity,
            BackendIdentity::Redis {
                topology: RedisTopology::Cluster { .. },
                ..
            }
        );
        let (module, name) = match (kind, cluster) {
            ("memory", _) => ("litellm.caching.in_memory_cache", "InMemoryCache"),
            ("redis", false) => ("litellm.caching.redis_cache", "RedisCache"),
            ("redis", true) => ("litellm.caching.redis_cluster_cache", "RedisClusterCache"),
            ("redis_semantic", _) => ("litellm.caching.redis_semantic_cache", "RedisSemanticCache"),
            ("qdrant_semantic", _) => (
                "litellm.caching.qdrant_semantic_cache",
                "QdrantSemanticCache",
            ),
            ("gcs", _) => ("litellm.caching.gcs_cache", "GCSCache"),
            ("valkey-semantic", _) => (
                "litellm.caching.valkey_semantic_cache",
                "ValkeySemanticCache",
            ),
            ("disk", _) => ("litellm.caching.disk_cache", "DiskCache"),
            ("azure-blob", _) => ("litellm.caching.azure_blob_cache", "AzureBlobCache"),
            ("s3", _) => ("litellm.caching.s3_cache", "S3Cache"),
            _ => unreachable!(),
        };
        let cache_kind = identity.cache_type();
        let backend = facade.getattr("cache")?;
        if facade.getattr("type")?.extract::<String>()? != cache_kind
            || !backend.get_type().is(&py.import(module)?.getattr(name)?)
        {
            return Err(PyTypeError::new_err(
                "facade and native backend types must match",
            ));
        }
        let config = match NativeCacheConfig::project(facade)? {
            CacheConfigProjection::Native(config) => *config,
            CacheConfigProjection::Unsupported(reason) => {
                return Err(PyTypeError::new_err(reason.message()));
            }
        };
        if let Some(message) = config.service_mismatch(service) {
            return Err(PyTypeError::new_err(message));
        }
        if kind == "redis_semantic"
            && service
                .embedder_object()
                .is_none_or(|embedder| !backend.is(embedder.bind(py)))
        {
            return Err(PyTypeError::new_err(
                "facade backend must be the native embedder",
            ));
        }
        Ok(Self {
            outer: ObjectGuard::capture(
                py,
                facade,
                &[
                    "type",
                    "mode",
                    "ttl",
                    "namespace",
                    "supported_call_types",
                    "redis_flush_size",
                    "semantic_cache_scope",
                ],
            )?,
            backend: ObjectGuard::capture(
                py,
                &backend,
                &[
                    "namespace",
                    "default_ttl",
                    "max_size_in_memory",
                    "max_size_per_item",
                    "redis_kwargs",
                    "redis_flush_size",
                    "similarity_threshold",
                    "distance_threshold",
                    "embedding_model",
                    "embedding_max_input_tokens",
                    "embedding_timeout",
                    "qdrant_api_base",
                    "qdrant_api_key",
                    "collection_name",
                    "vector_size",
                    "_index_name",
                    "_redis_url",
                    "similarity_threshold",
                    "embedding_model",
                    "index_name",
                    "embedding_max_input_tokens",
                    "embedding_timeout",
                    "bucket_name",
                    "key_prefix",
                    "path_service_account",
                ],
            )?,
            disk_store: (kind == "disk")
                .then(|| DiskStoreGuard::capture(&backend))
                .transpose()?,
            connection: ConnectionGuard::capture(kind, cluster, &backend)?,
        })
    }

    pub(super) fn matches(&self, py: Python<'_>, facade: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !self.outer.matches(py, facade)? {
            return Ok(false);
        }
        let backend = facade.getattr("cache")?;
        if !self.backend.matches(py, &backend)? {
            return Ok(false);
        }
        if let Some(guard) = &self.disk_store
            && !guard.matches(py, &backend)?
        {
            return Ok(false);
        }
        self.connection.matches(py, &backend)
    }

    pub(super) fn traverse(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.outer.traverse(&visit)?;
        self.backend.traverse(&visit)?;
        if let Some(guard) = &self.disk_store {
            guard.traverse(&visit)?;
        }
        self.connection.traverse(&visit)
    }
}

pub(super) fn resolve(
    py: Python<'_>,
    facade: &Bound<'_, PyAny>,
) -> PyResult<Option<NativeResponseCache>> {
    let Ok(dict) = facade
        .getattr("__dict__")
        .and_then(|dict| dict.cast_into::<PyDict>().map_err(Into::into))
    else {
        return Ok(None);
    };
    let Some(handle) = dict.get_item("_native_cache_handle")? else {
        return Ok(None);
    };
    let Ok(handle) = handle.extract::<PyRef<'_, CacheTestHandle>>() else {
        return Ok(None);
    };
    let Some(guard) = &handle.guard else {
        return Ok(None);
    };
    if !guard.matches(py, facade).unwrap_or(false) {
        return Ok(None);
    }
    handle.service().map(Some)
}
