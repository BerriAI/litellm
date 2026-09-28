use std::{sync::Arc, time::Duration};

use litellm_cache::{DeleteCache, DisconnectCache, PingCache};
use litellm_host_python::{from_py, release_gil, to_py};
use serde_json::Value;

use litellm_cache_memory::InMemoryCache;
use litellm_cache_redis::{RedisCache, RedisTopology};
use litellm_cache_response::{
    CacheEntry, CacheKeyInput, ExactResponseCache, ResponseCache, ResponseCacheCodec,
    ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[pyclass(
    frozen,
    name = "NativeCacheHandle",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeCacheHandle {
    service: Arc<dyn ResponseCacheService>,
    backend: Arc<dyn ExactResponseCache>,
    storage: Storage,
    pid: u32,
}

#[derive(Clone)]
enum Storage {
    Memory(Arc<InMemoryCache<CacheEntry>>),
    Redis(Arc<RedisCache<ResponseCacheCodec>>),
}

impl NativeCacheHandle {
    fn check_process(&self) -> PyResult<()> {
        if self.pid != std::process::id() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "recreate the v2 cache after fork",
            ));
        }
        Ok(())
    }
}

fn request(key: String, ttl: Option<f64>) -> PyResult<ResponseCacheRequest> {
    let mut request: ResponseCacheRequest = ResponseCacheRequest::new(CacheKeyInput {
        preset: Some(key),
        ..Default::default()
    });
    request.context.ttl = ttl.map(duration).transpose()?;
    Ok(request)
}

#[pymethods]
impl NativeCacheHandle {
    #[staticmethod]
    #[pyo3(signature = (*, ttl=600.0, capacity=200, max_entry_bytes=4194304))]
    fn memory(ttl: f64, capacity: usize, max_entry_bytes: usize) -> PyResult<Self> {
        let ttl = duration(ttl)?;
        if capacity == 0 || max_entry_bytes == 0 {
            return Err(PyValueError::new_err("cache limits must be positive"));
        }
        let storage = Arc::new(InMemoryCache::new(Some(capacity), Some(ttl)));
        let backend = Arc::new(ResponseCache::new(storage.clone()).with_config(
            ResponseCacheConfig {
                namespace: "sdk".into(),
                max_entry_bytes,
            },
        ));
        Ok(Self {
            service: backend.clone(),
            backend,
            storage: Storage::Memory(storage),
            pid: std::process::id(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (url, *, namespace, ttl=600.0, max_entry_bytes=4194304))]
    fn redis(
        py: Python<'_>,
        url: &str,
        namespace: String,
        ttl: f64,
        max_entry_bytes: usize,
    ) -> PyResult<Self> {
        let ttl = duration(ttl)?;
        if namespace.is_empty() || max_entry_bytes == 0 {
            return Err(PyValueError::new_err(
                "namespace and a positive cache limit are required",
            ));
        }
        let storage = Arc::new(
            release_gil(py, || {
                RedisCache::connect(
                    url,
                    &RedisTopology::Standalone,
                    Some(ttl),
                    ResponseCacheCodec,
                )
            })
            .map_err(super::cache_error)?,
        );
        let backend = Arc::new(ResponseCache::new(storage.clone()).with_config(
            ResponseCacheConfig {
                namespace,
                max_entry_bytes,
            },
        ));
        Ok(Self {
            service: backend.clone(),
            backend,
            storage: Storage::Redis(storage),
            pid: std::process::id(),
        })
    }
    fn get(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        self.check_process()?;
        let request = request(key, None)?;
        let value = release_gil(py, || self.backend.lookup(&request, super::request::now()))
            .map_err(super::cache_error)?;
        to_py(py, &value)
    }

    #[pyo3(signature = (key, value, *, ttl=None))]
    fn set(
        &self,
        py: Python<'_>,
        key: String,
        value: &Bound<'_, PyAny>,
        ttl: Option<f64>,
    ) -> PyResult<()> {
        self.check_process()?;
        let request = request(key, ttl)?;
        let value: Value = from_py(value)?;
        release_gil(py, || {
            self.backend.store(&request, value, super::request::now())
        })
        .map_err(super::cache_error)
    }

    fn async_get<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        let request = request(key, None)?;
        let backend = self.backend.clone();
        crate::logger::run_async(
            py,
            async move { backend.async_lookup(&request, super::request::now()).await },
            super::cache_error,
        )
    }

    #[pyo3(signature = (key, value, *, ttl=None))]
    fn async_set<'py>(
        &self,
        py: Python<'py>,
        key: String,
        value: &Bound<'_, PyAny>,
        ttl: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        let request = request(key, ttl)?;
        let value: Value = from_py(value)?;
        let backend = self.backend.clone();
        crate::logger::run_async(
            py,
            async move {
                backend
                    .async_store(&request, value, super::request::now())
                    .await
            },
            super::cache_error,
        )
    }

    #[pyo3(signature = (entries, *, ttl=None))]
    fn async_set_many<'py>(
        &self,
        py: Python<'py>,
        entries: &Bound<'_, PyAny>,
        ttl: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        let entries: Vec<(String, Value)> = from_py(entries)?;
        let entries = entries
            .into_iter()
            .map(|(key, value)| Ok((request(key, ttl)?, value)))
            .collect::<PyResult<Vec<_>>>()?;
        let backend = self.backend.clone();
        crate::logger::run_async(
            py,
            async move {
                backend
                    .async_store_batch(entries, super::request::now())
                    .await
            },
            super::cache_error,
        )
    }

    fn flush(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.check_process()?;
        let backend = self.backend.clone();
        crate::logger::run_sync(
            py,
            async move { backend.async_flush().await },
            super::cache_error,
        )
    }

    fn async_flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        let backend = self.backend.clone();
        crate::logger::run_async(
            py,
            async move { backend.async_flush().await },
            super::cache_error,
        )
    }

    fn ping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        let storage = self.storage.clone();
        crate::logger::run_async(
            py,
            async move {
                match storage {
                    Storage::Memory(_) => Ok(true),
                    Storage::Redis(cache) => cache.ping().await,
                }
            },
            super::cache_error,
        )
    }

    fn disconnect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        let storage = self.storage.clone();
        crate::logger::run_async(
            py,
            async move {
                match storage {
                    Storage::Memory(cache) => cache.disconnect().await,
                    Storage::Redis(cache) => cache.disconnect().await,
                }
            },
            super::cache_error,
        )
    }

    fn delete<'py>(&self, py: Python<'py>, keys: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        let storage = self.storage.clone();
        crate::logger::run_async(
            py,
            async move {
                for key in keys {
                    match &storage {
                        Storage::Memory(cache) => cache.async_delete_cache(&key).await?,
                        Storage::Redis(cache) => cache.async_delete_cache(&key).await?,
                    }
                }
                Ok(())
            },
            super::cache_error,
        )
    }
}

fn duration(seconds: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|value| !value.is_zero())
        .ok_or_else(|| PyValueError::new_err("cache durations must be finite and positive"))
}

pub(crate) fn configured(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<(
    Option<Arc<dyn ResponseCacheService>>,
    litellm_cache_response::CacheOptions,
)> {
    let configured = py.import("litellm")?.getattr("cache")?;
    if configured.is_none()
        || kwargs
            .get_item("caching")?
            .is_some_and(|value| value.is(pyo3::types::PyBool::new(py, false)))
    {
        return Ok((
            None,
            litellm_cache_response::CacheOptions::new(litellm_cache_response::CacheScope::Shared),
        ));
    }
    let handle = configured
        .getattr_opt("cache")?
        .map(|backend| backend.getattr_opt("native_handle"))
        .transpose()?
        .flatten()
        .ok_or_else(|| {
            crate::errors::RustBridgeDeclined::new_err(
                "the configured cache requires Python inference",
            )
        })?;
    let Ok(cache) = handle.extract::<PyRef<'_, NativeCacheHandle>>() else {
        return Ok((
            None,
            litellm_cache_response::CacheOptions::new(litellm_cache_response::CacheScope::Shared),
        ));
    };
    cache.check_process()?;
    let controls = kwargs.get_item("cache")?.filter(|value| !value.is_none());
    let controls = controls
        .as_ref()
        .map(|value| value.cast::<PyDict>())
        .transpose()?;
    if let Some(controls) = controls {
        for name in controls.keys() {
            let name = name.extract::<String>()?;
            if !matches!(
                name.as_str(),
                "no-cache" | "no-store" | "ttl" | "s-maxage" | "s-max-age"
            ) {
                return Err(PyValueError::new_err(format!(
                    "unsupported v2 cache control: {name}"
                )));
            }
        }
    }
    let boolean = |name: &str| -> PyResult<bool> {
        controls
            .map(|values| values.get_item(name))
            .transpose()?
            .flatten()
            .map(|value| value.extract())
            .transpose()
            .map(|value| value.unwrap_or(false))
    };
    let seconds = |name: &str| -> PyResult<Option<Duration>> {
        controls
            .map(|values| values.get_item(name))
            .transpose()?
            .flatten()
            .map(|value| duration(value.extract()?))
            .transpose()
    };
    Ok((
        Some(cache.service.clone()),
        litellm_cache_response::CacheOptions {
            caching: kwargs
                .get_item("caching")?
                .filter(|value| !value.is_none())
                .map(|value| value.extract())
                .transpose()?,
            no_cache: boolean("no-cache")?,
            no_store: boolean("no-store")?,
            ttl: seconds("ttl")?,
            max_age: seconds("s-max-age")?.or(seconds("s-maxage")?),
            scope: litellm_cache_response::CacheScope::Shared,
        },
    ))
}
