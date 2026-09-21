mod facade;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_cache::{CacheControls, CacheKeyInput, Error};
use litellm_cache_response::{NativeResponseCache, ResponseCacheRequest};
use litellm_host_python::{ExecutionStep, from_py, release_gil, run_async, to_py};
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use serde::Deserialize;
use serde_json::Value;

use facade::FacadeGuard;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestInput {
    key: CacheKeyInput,
    controls: Option<CacheControls>,
    ttl_seconds: Option<f64>,
    max_age_seconds: Option<f64>,
}

fn request(value: &Bound<'_, PyAny>) -> PyResult<ResponseCacheRequest> {
    let input: RequestInput = from_py(value)?;
    let mut request = ResponseCacheRequest::new(input.key);
    if let Some(controls) = input.controls {
        request.controls = controls;
    }
    request.kwargs.ttl = input.ttl_seconds.map(duration).transpose()?;
    request.max_age = input.max_age_seconds.map(duration).transpose()?;
    Ok(request)
}

fn duration(seconds: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| PyValueError::new_err("cache durations must be finite and nonnegative"))
}

fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}

fn cache_error(error: Error) -> PyErr {
    match error {
        Error::InvalidEntry => PyValueError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyclass(frozen)]
pub(crate) struct NativeCacheHandle {
    service: NativeResponseCache,
    guard: Option<FacadeGuard>,
    pid: u32,
}

impl NativeCacheHandle {
    fn service(&self) -> PyResult<NativeResponseCache> {
        if self.pid != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "native cache handles must be recreated after fork",
            ));
        }
        Ok(self.service.clone())
    }
}

#[pymethods]
impl NativeCacheHandle {
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
        let ttl = ttl_seconds.map(duration).transpose()?;
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

    fn bind_facade(&self, py: Python<'_>, facade: &Bound<'_, PyAny>) -> PyResult<()> {
        let service = self.service()?;
        let guard = FacadeGuard::capture(py, facade, self.backend())?;
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

enum CacheBinding {
    Disabled,
    Native(NativeResponseCache),
    PythonCallback(Py<PyAny>),
}

#[pyclass(frozen, name = "CacheBinding")]
pub(crate) struct ResolvedCache {
    binding: CacheBinding,
    pid: u32,
}

impl ResolvedCache {
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
                let service = service.clone();
                run_async(
                    py,
                    async move { service.async_lookup(&request, now()).await },
                    cache_error,
                )?
            }
            CacheBinding::PythonCallback(object) => object.bind(py).call_method(
                "async_get_cache",
                (),
                Some(callback_kwargs(kwargs)?),
            )?,
        };
        Ok(ExecutionStep::Await(awaitable.unbind()))
    }
}

#[pymethods]
impl ResolvedCache {
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
            CacheBinding::PythonCallback(object) => object
                .bind(py)
                .call_method(
                    "get_cache",
                    (),
                    Some(self::callback_kwargs(callback_kwargs)?),
                )
                .map(Bound::unbind),
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
            CacheBinding::PythonCallback(object) => object
                .bind(py)
                .call_method(
                    "add_cache",
                    (response,),
                    Some(self::callback_kwargs(callback_kwargs)?),
                )
                .map(|_| ()),
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
                let service = service.clone();
                run_async(
                    py,
                    async move { service.async_store(&request, response, now()).await },
                    cache_error,
                )
            }
            CacheBinding::PythonCallback(object) => object.bind(py).call_method(
                "async_add_cache",
                (response,),
                Some(self::callback_kwargs(callback_kwargs)?),
            ),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let CacheBinding::PythonCallback(object) = &self.binding {
            visit.call(object)?;
        }
        Ok(())
    }
}

fn callback_kwargs<'a, 'py>(
    kwargs: Option<&'a Bound<'py, PyDict>>,
) -> PyResult<&'a Bound<'py, PyDict>> {
    kwargs.ok_or_else(|| {
        PyTypeError::new_err("Python cache callbacks require their original callback_kwargs")
    })
}

fn ready_none(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let future = py
        .import("asyncio")?
        .call_method0("get_running_loop")?
        .call_method0("create_future")?;
    future.call_method1("set_result", (py.None(),))?;
    Ok(future)
}

#[pyclass(frozen)]
pub(crate) struct CacheResolver {
    namespace: Py<PyAny>,
}

#[pymethods]
impl CacheResolver {
    #[new]
    fn new(namespace: Py<PyAny>) -> Self {
        Self { namespace }
    }

    pub(crate) fn resolve(&self, py: Python<'_>) -> PyResult<ResolvedCache> {
        let object = self.namespace.bind(py).getattr("cache")?;
        let binding = if object.is_none() {
            CacheBinding::Disabled
        } else if let Ok(handle) = object.extract::<PyRef<'_, NativeCacheHandle>>() {
            CacheBinding::Native(handle.service()?)
        } else if let Some(service) = facade::resolve(py, &object)? {
            CacheBinding::Native(service)
        } else {
            CacheBinding::PythonCallback(object.unbind())
        };
        Ok(ResolvedCache {
            binding,
            pid: std::process::id(),
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.namespace)
    }
}
