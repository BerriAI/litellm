use std::env;

use litellm_cache_redis::{RedisNode, RedisTopology};
use litellm_http::ClientVariant;

use litellm_cache_qdrant_semantic::{OpenAiEmbedderConfig, Quantization};
use litellm_host_python::{release_gil, run_sync_value};
use pyo3::{PyTraverseError, PyVisit, exceptions::PyRuntimeError, prelude::*, types::PyDict};
use url::Url;

use super::{
    cache_error, config::QdrantSemanticCacheConfig, facade::FacadeGuard,
    native::NativeResponseCache, request::duration,
};
use crate::http;

#[pyclass(frozen, name = "_CacheTestHandle")]
pub(crate) struct CacheTestHandle {
    service: NativeResponseCache,
    pub(super) guard: Option<FacadeGuard>,
    pid: u32,
}

impl CacheTestHandle {
    pub(super) fn service(&self) -> PyResult<NativeResponseCache> {
        if self.pid != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "native cache handles must be recreated after fork",
            ));
        }
        Ok(self.service.clone())
    }
}

#[pymethods]
impl CacheTestHandle {
    #[staticmethod]
    #[pyo3(signature = (*, capacity=200, ttl_seconds=600.0, max_entry_bytes=1048576))]
    fn memory(capacity: usize, ttl_seconds: f64, max_entry_bytes: usize) -> PyResult<Self> {
        Ok(Self {
            service: NativeResponseCache::memory(capacity, duration(ttl_seconds)?, max_entry_bytes),
            guard: None,
            pid: std::process::id(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (url, *, ttl_seconds=60.0, namespace=None, startup_nodes=None))]
    fn redis(
        py: Python<'_>,
        url: String,
        ttl_seconds: f64,
        namespace: Option<String>,
        startup_nodes: Option<Vec<(String, u16)>>,
    ) -> PyResult<Self> {
        let ttl = Some(duration(ttl_seconds)?);
        let topology = match startup_nodes {
            None => RedisTopology::Standalone,
            Some(nodes) => RedisTopology::Cluster {
                startup_nodes: nodes
                    .into_iter()
                    .map(|(host, port)| RedisNode { host, port })
                    .collect(),
            },
        };
        let service = release_gil(py, move || {
            NativeResponseCache::redis(&url, &topology, ttl, namespace)
        })
        .map_err(cache_error)?;
        Ok(Self {
            service,
            guard: None,
            pid: std::process::id(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (directory))]
    fn disk(py: Python<'_>, directory: String) -> PyResult<Self> {
        let service =
            release_gil(py, move || NativeResponseCache::disk(&directory)).map_err(cache_error)?;
        Ok(Self {
            service,
            guard: None,
            pid: std::process::id(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (url, *, collection_name, similarity_threshold, vector_size, embedding_model="text-embedding-3-small", api_key=None, embedding_api_key=None, embedding_api_base=None, embedding_timeout_seconds=None, quantization="binary"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the test handle exposes the complete Qdrant constructor"
    )]
    fn qdrant_semantic(
        py: Python<'_>,
        url: String,
        collection_name: String,
        similarity_threshold: f64,
        vector_size: u64,
        embedding_model: &str,
        api_key: Option<String>,
        embedding_api_key: Option<String>,
        embedding_api_base: Option<String>,
        embedding_timeout_seconds: Option<f64>,
        quantization: &str,
    ) -> PyResult<Self> {
        let parsed = Url::parse(&url).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "native Qdrant requires the default REST port so the gRPC port can be derived",
            )
        })?;
        if !matches!(parsed.scheme(), "http" | "https")
            || (!parsed.path().is_empty() && parsed.path() != "/")
            || parsed.query().is_some()
            || parsed.host_str().is_none()
            || parsed.port().is_some_and(|port| port != 6333)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "native Qdrant requires the default REST port so the gRPC port can be derived",
            ));
        }
        let mut grpc_url = parsed;
        grpc_url.set_port(Some(6334)).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "native Qdrant requires the default REST port so the gRPC port can be derived",
            )
        })?;
        grpc_url.set_path("");
        grpc_url.set_query(None);
        let embedding_api_key = embedding_api_key
            .or_else(|| {
                env::var("OPENAI_API_KEY")
                    .ok()
                    .filter(|value| !value.is_empty())
            })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "native semantic embedding requires an OpenAI API key",
                )
            })?;
        let embedding_api_base = embedding_api_base.unwrap_or_else(|| {
            env::var("OPENAI_BASE_URL")
                .or_else(|_| env::var("OPENAI_API_BASE"))
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_owned())
        });
        let quantization = match quantization {
            "binary" => Quantization::Binary,
            "scalar" => Quantization::Scalar,
            "product" => Quantization::Product,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unsupported Qdrant quantization",
                ));
            }
        };
        let config = QdrantSemanticCacheConfig {
            grpc_url: grpc_url.to_string().trim_end_matches('/').to_owned(),
            api_key,
            collection_name,
            similarity_threshold,
            vector_size,
            embedding: OpenAiEmbedderConfig {
                api_base: embedding_api_base,
                api_key: embedding_api_key,
                model: embedding_model.to_owned(),
                timeout: embedding_timeout_seconds.map(duration).transpose()?,
            },
            quantization,
        };
        let http_config = http::call_config(py, &PyDict::new(py), true)?;
        let client = http::pool()
            .client(&http_config, ClientVariant::Provider)
            .map_err(http::client_error)?;
        let service = run_sync_value(py, async move {
            let handle = tokio::runtime::Handle::current();
            NativeResponseCache::qdrant_semantic(config, client, handle)
                .await
                .map_err(cache_error)
        })?;
        Ok(Self {
            service,
            guard: None,
            pid: std::process::id(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (account_url, container))]
    fn azure_blob(py: Python<'_>, account_url: String, container: String) -> PyResult<Self> {
        let service = run_sync_value(py, async move {
            NativeResponseCache::azure_blob(&account_url, &container)
                .await
                .map_err(cache_error)
        })?;
        Ok(Self {
            service,
            guard: None,
            pid: std::process::id(),
        })
    }

    #[getter]
    fn backend(&self) -> &'static str {
        self.service.kind()
    }

    fn _bind_facade(&self, py: Python<'_>, facade: &Bound<'_, PyAny>) -> PyResult<()> {
        let service = self.service()?;
        let guard = FacadeGuard::capture(py, facade, &service)?;
        let service = service.with_redis_flush_size(
            facade
                .getattr("redis_flush_size")?
                .extract::<Option<usize>>()?,
        );
        let handle = Py::new(
            py,
            Self {
                service,
                guard: Some(guard),
                pid: self.pid,
            },
        )?;
        facade.setattr("_native_cache_handle", handle)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(guard) = &self.guard {
            guard.traverse(visit)?;
        }
        Ok(())
    }
}
