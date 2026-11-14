mod chain;
pub use chain::HookChain;

use crate::PythonOwned;
use litellm_host::hooks::{CallHooks, HookRuntime, RuntimeCallEvent};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// What a hook step produced: either the value the driver asked for, or a Python
/// awaitable the driver hands back to the caller's task before asking again.
pub type HookResume<L, T> = fn(&mut L, Python<'_>, PyResult<Py<PyAny>>) -> PyResult<HookStep<L, T>>;

pub enum HookStep<L, T> {
    Await(Py<PyAny>, HookResume<L, T>),
    Ready(T),
}

pub struct PythonRuntime;

impl HookRuntime for PythonRuntime {
    type Context<'a> = Python<'a>;
    type Arguments = Py<PyDict>;
    type Response = Py<PyAny>;
    type Chunk = Py<PyAny>;
    type Error = PyErr;
    type Step<H, T> = HookStep<H, T>;

    fn ready<H, T>(value: T) -> Self::Step<H, T> {
        HookStep::Ready(value)
    }
}

pub type PythonCallEvent<'a> = RuntimeCallEvent<'a, PythonRuntime>;

pub trait PythonCallHooks: CallHooks<PythonRuntime> + PythonOwned {}

impl<H: CallHooks<PythonRuntime> + PythonOwned> PythonCallHooks for H {}
