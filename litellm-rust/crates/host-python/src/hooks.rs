mod guard;

use futures_util::future::BoxFuture;
use litellm_host::hooks::{AnyGroup, Cancelled, Surface};
use pyo3::exceptions::{PyBaseException, PyException};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::PythonOwned;

/// Where a hook or host call awaits a Python awaitable: the caller's task, which the call
/// suspends to and resumes from with the awaitable's result.
pub trait Awaiter: Sync {
    fn awaited(&self, awaitable: Py<PyAny>) -> BoxFuture<'_, PyResult<Py<PyAny>>>;
}

/// The Python public surface: keyword arguments in, Python objects out, `PyErr` for every
/// failure, and the caller's task to await on.
pub struct PythonSurface;

impl Surface for PythonSurface {
    type Task<'a> = &'a dyn Awaiter;
    type Arguments = Py<PyDict>;
    type Response = Py<PyAny>;
    type Chunk = Py<PyAny>;
    type Error = PyErr;
    type Failure = Py<PyBaseException>;
    type Group = dyn PythonGroup;
}

/// A group mounted on the Python surface: the four facets plus Python ownership.
pub trait PythonGroup: AnyGroup<PythonSurface> + PythonOwned {}

impl<H: AnyGroup<PythonSurface> + PythonOwned> PythonGroup for H {}

pub type Hooks = litellm_host::hooks::Hooks<PythonSurface>;

pub(crate) fn is_cancellation(py: Python<'_>, error: &PyErr) -> bool {
    !error.is_instance_of::<PyException>(py)
}

/// How a Python outcome hook settles its own result: an ordinary exception is reported as
/// unraisable and the call goes on, anything else is a cancellation that stops it.
pub fn report_unless_cancelled(result: PyResult<()>) -> Result<(), Cancelled<PyErr>> {
    let Err(error) = result else {
        return Ok(());
    };
    Python::attach(|py| {
        if is_cancellation(py, &error) {
            return Err(Cancelled(error));
        }
        error.write_unraisable(py, None);
        Ok(())
    })
}
