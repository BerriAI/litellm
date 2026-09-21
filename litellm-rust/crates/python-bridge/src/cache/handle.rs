use litellm_cache_redis::{RedisNode, RedisTopology};
use litellm_host_python::{release_gil, run_sync_value};
use pyo3::{PyTraverseError, PyVisit, exceptions::PyRuntimeError, prelude::*};

use super::{
    cache_error, embedder::PythonEmbedder, facade::FacadeGuard, native::NativeResponseCache,
    request::duration,
};

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
    #[pyo3(signature = (url, similarity_threshold, index_name, embedder))]
    fn valkey_semantic(
        url: String,
        similarity_threshold: f64,
        index_name: String,
        embedder: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let python_embedder = PythonEmbedder::from_backend(embedder)?;
        let service = NativeResponseCache::valkey_semantic(
            &url,
            similarity_threshold,
            index_name,
            python_embedder,
        )
        .map_err(cache_error)?;
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
        let service = service
            .with_scope(
                facade
                    .getattr("semantic_cache_scope")?
                    .extract::<String>()?,
            )
            .with_redis_flush_size(
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
