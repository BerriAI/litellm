use std::sync::{Mutex, MutexGuard};

use litellm_host_python::{from_py_argument, to_py};
use litellm_router::{Error, Router, config::RouterConfig};
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
};

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub struct RouterHandle {
    router: Mutex<Router>,
}

impl RouterHandle {
    fn lock(&self) -> PyResult<MutexGuard<'_, Router>> {
        self.router
            .lock()
            .map_err(|_| PyRuntimeError::new_err("router state is unavailable"))
    }
}

fn public_error(error: Error) -> PyErr {
    match error {
        Error::Closed => PyRuntimeError::new_err(error.to_string()),
        Error::NotImplemented => PyNotImplementedError::new_err(error.to_string()),
        Error::DuplicateDeployment { .. }
        | Error::UnknownModel { .. }
        | Error::InvalidSelection { .. }
        | Error::Unauthorized => PyValueError::new_err(error.to_string()),
    }
}

#[pyfunction]
pub fn routing_create(config: &Bound<'_, PyAny>) -> PyResult<RouterHandle> {
    let config: RouterConfig = from_py_argument(config)?;
    Ok(RouterHandle {
        router: Mutex::new(Router::new(config).map_err(public_error)?),
    })
}

#[pyfunction]
pub fn routing_snapshot(py: Python<'_>, router: &RouterHandle) -> PyResult<Py<PyAny>> {
    let snapshot = router.lock()?.snapshot();
    to_py(py, &snapshot)
}

#[pyfunction]
pub fn routing_reconfigure(router: &RouterHandle, config: &Bound<'_, PyAny>) -> PyResult<()> {
    let config: RouterConfig = from_py_argument(config)?;
    router.lock()?.reconfigure(config).map_err(public_error)
}

#[pyfunction]
pub fn routing_close(router: &RouterHandle) -> PyResult<()> {
    router.lock()?.close();
    Ok(())
}
