mod chain;
pub use chain::HookChain;

use crate::PythonOwned;
use futures_util::future::BoxFuture;
use litellm_host::hooks::{CallHooks, HookRuntime, RuntimeCallEvent};
use pyo3::exceptions::PyBaseException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Where a hook or host call awaits a Python awaitable: the caller's task, which the call
/// suspends to and resumes from with the awaitable's result.
pub trait Awaiter: Sync {
    fn awaited(&self, awaitable: Py<PyAny>) -> BoxFuture<'_, PyResult<Py<PyAny>>>;
}

pub struct PythonRuntime;

impl HookRuntime for PythonRuntime {
    type Context<'a> = &'a dyn Awaiter;
    type Arguments = Py<PyDict>;
    type Response = Py<PyAny>;
    type Chunk = Py<PyAny>;
    type Error = PyErr;
    type Failure = Py<PyBaseException>;
}

pub type PythonCallEvent<'a> = RuntimeCallEvent<'a, PythonRuntime>;

pub trait PythonCallHooks: CallHooks<PythonRuntime> + PythonOwned {}

impl<H: CallHooks<PythonRuntime> + PythonOwned> PythonCallHooks for H {}
