use std::time::Duration;

use litellm_host_python::release_gil;
use pyo3::{PyTraverseError, PyVisit, exceptions::PyRuntimeError, prelude::*};

use super::{cache_error, facade::FacadeGuard, native::NativeResponseCache, request::duration};
use crate::python_settings::PythonSettings;

const PYTHON_REDIS_DEFAULT_TTL: Duration = Duration::from_secs(60);

#[derive(FromPyObject)]
struct PythonCacheSettings {
    default_redis_ttl: Option<f64>,
}

fn redis_default_ttl(py: Python<'_>) -> PyResult<Duration> {
    let settings: PythonCacheSettings = PythonSettings::Cache.read(py)?.extract()?;
    settings
        .default_redis_ttl
        .map(duration)
        .transpose()
        .map(|ttl| ttl.unwrap_or(PYTHON_REDIS_DEFAULT_TTL))
}

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
    #[pyo3(signature = (url, *, ttl_seconds=None, namespace=None))]
    fn redis(
        py: Python<'_>,
        url: String,
        ttl_seconds: Option<f64>,
        namespace: Option<String>,
    ) -> PyResult<Self> {
        let ttl = Some(match ttl_seconds {
            Some(seconds) => duration(seconds)?,
            None => redis_default_ttl(py)?,
        });
        let service = release_gil(py, move || NativeResponseCache::redis(&url, ttl, namespace))
            .map_err(cache_error)?;
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
