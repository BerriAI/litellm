use litellm_cache_response::PartialHits;
use litellm_host_python::{ExecutionStep, from_py, release_gil, to_py};
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

use super::{
    activation::activate,
    cache_error,
    callback::PythonCallback,
    config::{CacheConfigProjection, NativeCacheConfig},
    facade::{Cache, is_unmodified},
    future::{ready_none, ready_value},
    native::{NativeResponseCache, SemanticReply},
    request::{now, request, requests},
};
use crate::{errors::RustBridgeDeclined, logger::run_async};

pub(super) enum CacheBinding {
    Disabled,
    Native(NativeResponseCache),
    PythonCallback(PythonCallback),
}

#[pyclass(frozen, name = "_ResponseCacheRuntime")]
pub(crate) struct ResolvedCache {
    binding: CacheBinding,
    guard: Option<super::guard::FacadeGuard>,
    pid: u32,
}

impl ResolvedCache {
    pub(super) fn new(binding: CacheBinding) -> Self {
        Self {
            binding,
            guard: None,
            pid: std::process::id(),
        }
    }

    pub(super) fn with_guard(mut self, guard: super::guard::FacadeGuard) -> Self {
        self.guard = Some(guard);
        self
    }

    pub(super) fn native_service(&self) -> PyResult<Option<NativeResponseCache>> {
        self.check_process()?;
        Ok(match &self.binding {
            CacheBinding::Native(service) => Some(service.clone()),
            _ => None,
        })
    }

    fn check_process(&self) -> PyResult<()> {
        if matches!(self.binding, CacheBinding::Native(_)) && self.pid != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "native cache bindings must be resolved again after fork",
            ));
        }
        Ok(())
    }

    pub(crate) fn lookup_step(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<ExecutionStep> {
        self.check_process()?;
        let awaitable = match &self.binding {
            CacheBinding::Disabled => ready_none(py)?,
            CacheBinding::Native(service) => {
                let request = request(input)?;
                service.async_lookup_py(py, request)?
            }
            CacheBinding::PythonCallback(callback) => callback.async_lookup(py, kwargs)?,
        };
        Ok(ExecutionStep::Await(awaitable.unbind()))
    }
}

#[pymethods]
impl ResolvedCache {
    #[staticmethod]
    pub(crate) fn from_selected(cache: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = cache.py();
        let binding = if cache.is_none() {
            CacheBinding::Disabled
        } else if let Ok(facade) = cache.cast::<Cache>() {
            match is_unmodified(py, cache)?
                .then(|| facade.get().native_service(py, cache))
                .transpose()?
                .flatten()
            {
                Some(service) => CacheBinding::Native(service),
                None => CacheBinding::PythonCallback(PythonCallback::new(cache.clone().unbind())),
            }
        } else if let Some(runtime) = cache
            .getattr_opt("_native_cache")?
            .filter(|value| !value.is_none())
        {
            let resolved = runtime
                .getattr("native")?
                .extract::<PyRef<'_, ResolvedCache>>()?;
            match resolved.native_service()? {
                Some(service) => {
                    if !resolved
                        .guard
                        .as_ref()
                        .is_some_and(|guard| guard.matches(py, cache).unwrap_or(false))
                    {
                        return Err(RustBridgeDeclined::new_err(
                            "native cache runtime no longer matches its facade",
                        ));
                    }
                    CacheBinding::Native(service)
                }
                None => CacheBinding::PythonCallback(PythonCallback::new(cache.clone().unbind())),
            }
        } else {
            CacheBinding::PythonCallback(PythonCallback::new(cache.clone().unbind()))
        };
        Ok(Self::new(binding))
    }

    #[staticmethod]
    fn from_cache(cache: &Bound<'_, PyAny>) -> PyResult<Self> {
        let config = match NativeCacheConfig::project(cache)? {
            CacheConfigProjection::Native(config) => *config,
            CacheConfigProjection::Unsupported(reason) => {
                return Err(RustBridgeDeclined::new_err(reason.message()));
            }
        };
        let backend = cache.getattr("cache")?;
        let service = activate(cache.py(), &backend, config)?;
        let resolved = Self::new(CacheBinding::Native(service.clone()));
        Ok(
            match super::guard::FacadeGuard::capture(cache.py(), cache, &service) {
                Ok(guard) => resolved.with_guard(guard),
                Err(_) => resolved,
            },
        )
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.binding {
            CacheBinding::Disabled => "disabled",
            CacheBinding::Native(_) => "native",
            CacheBinding::PythonCallback(_) => "python_callback",
        }
    }

    #[pyo3(signature = (request, *, callback_kwargs=None))]
    fn lookup(
        &self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        callback_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Disabled => Ok(py.None()),
            CacheBinding::Native(service) => {
                let request = self::request(request)?;
                let service = service.clone();
                let response = release_gil(py, move || service.lookup(&request, now()))
                    .map_err(cache_error)?;
                to_py(py, &response)
            }
            CacheBinding::PythonCallback(callback) => {
                callback.lookup(py, callback_kwargs).map(Bound::unbind)
            }
        }
    }

    /// `(response, similarity)`: the similarity is `None` when the backend reports none.
    fn lookup_semantic(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Native(service) => {
                let request = self::request(request)?;
                let service = service.clone();
                let lookup = release_gil(py, move || service.lookup_semantic(&request, now()))
                    .map_err(cache_error)?;
                to_py(py, &SemanticReply::from(lookup))
            }
            CacheBinding::Disabled => to_py(py, &SemanticReply(None, None)),
            CacheBinding::PythonCallback(_) => Err(PyRuntimeError::new_err(
                "semantic lookups require a native cache binding",
            )),
        }
    }

    fn async_lookup_semantic<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Native(service) => {
                service.async_lookup_semantic_py(py, self::request(request)?)
            }
            CacheBinding::Disabled => ready_value(py, &SemanticReply(None, None)),
            CacheBinding::PythonCallback(_) => Err(PyRuntimeError::new_err(
                "semantic lookups require a native cache binding",
            )),
        }
    }

    #[pyo3(signature = (request, response, *, callback_kwargs=None))]
    fn store(
        &self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        response: &Bound<'_, PyAny>,
        callback_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Disabled => Ok(()),
            CacheBinding::Native(service) => {
                let request = self::request(request)?;
                let response: Value = from_py(response)?;
                let service = service.clone();
                release_gil(py, move || service.store(&request, response, now()))
                    .map_err(cache_error)
            }
            CacheBinding::PythonCallback(callback) => callback.store(py, response, callback_kwargs),
        }
    }

    #[pyo3(signature = (requests, *, callback_kwargs=None))]
    fn lookup_batch(
        &self,
        py: Python<'_>,
        requests: &Bound<'_, PyAny>,
        callback_kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Disabled => {
                let requests = self::requests(requests)?;
                to_py(py, &PartialHits::new(vec![None; requests.len()]))
            }
            CacheBinding::Native(service) => {
                let requests = self::requests(requests)?;
                let service = service.clone();
                let response = release_gil(py, move || service.lookup_batch(&requests, now()))
                    .map_err(cache_error)?;
                to_py(py, &response)
            }
            CacheBinding::PythonCallback(callback) => callback
                .lookup_batch(py, requests, callback_kwargs)
                .map(Bound::unbind),
        }
    }

    #[pyo3(signature = (request, *, callback_kwargs=None))]
    fn async_lookup<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
        callback_kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ExecutionStep::Await(awaitable) = self.lookup_step(py, request, callback_kwargs)?
        else {
            unreachable!()
        };
        Ok(awaitable.into_bound(py))
    }

    #[pyo3(signature = (request, response, *, callback_kwargs=None))]
    fn async_store<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
        response: &Bound<'py, PyAny>,
        callback_kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Disabled => ready_none(py),
            CacheBinding::Native(service) => {
                let request = self::request(request)?;
                let response: Value = from_py(response)?;
                service.async_store_py(py, request, response)
            }
            CacheBinding::PythonCallback(callback) => {
                callback.async_store(py, response, callback_kwargs)
            }
        }
    }

    #[pyo3(signature = (requests, *, callback_kwargs=None))]
    fn async_lookup_batch<'py>(
        &self,
        py: Python<'py>,
        requests: &Bound<'py, PyAny>,
        callback_kwargs: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Disabled => {
                let requests = self::requests(requests)?;
                ready_value(py, &PartialHits::new(vec![None; requests.len()]))
            }
            CacheBinding::Native(service) => {
                let requests = self::requests(requests)?;
                let service = service.clone();
                run_async(
                    py,
                    async move { service.async_lookup_batch(&requests, now()).await },
                    cache_error,
                )
            }
            CacheBinding::PythonCallback(callback) => {
                callback.async_lookup_batch(py, requests, callback_kwargs)
            }
        }
    }

    #[pyo3(signature = (requests, responses, *, callback_result=None, callback_kwargs=None))]
    fn async_store_batch<'py>(
        &self,
        py: Python<'py>,
        requests: &Bound<'py, PyAny>,
        responses: &Bound<'py, PyAny>,
        callback_result: Option<&Bound<'py, PyAny>>,
        callback_kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Disabled => ready_none(py),
            CacheBinding::Native(service) => {
                let requests = self::requests(requests)?;
                let responses: Vec<Value> = from_py(responses)?;
                if requests.len() != responses.len() {
                    return Err(PyValueError::new_err(
                        "batch cache requests and responses must have equal lengths",
                    ));
                }
                let entries = requests.into_iter().zip(responses).collect();
                service.async_store_batch_py(py, entries)
            }
            CacheBinding::PythonCallback(callback) => {
                callback.async_store_batch(py, callback_result, callback_kwargs)
            }
        }
    }

    fn async_flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Disabled => ready_none(py),
            CacheBinding::Native(service) => {
                let service = service.clone();
                run_async(py, async move { service.async_flush().await }, cache_error)
            }
            CacheBinding::PythonCallback(callback) => callback.async_flush(py),
        }
    }

    fn ping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.check_process()?;
        match &self.binding {
            CacheBinding::Disabled => ready_none(py),
            CacheBinding::Native(service) => {
                let service = service.clone();
                run_async(
                    py,
                    async move { service.test_connection().await },
                    cache_error,
                )
            }
            CacheBinding::PythonCallback(callback) => callback.ping(py),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let CacheBinding::PythonCallback(callback) = &self.binding {
            callback.traverse(&visit)?;
        }
        if let Some(guard) = &self.guard {
            guard.traverse(visit)?;
        }
        Ok(())
    }
}
