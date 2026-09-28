use crate::PythonOwned;
use litellm_host::hooks::{CallHooks, HookRuntime, RuntimeCallEvent};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The SDK's request policy, run by the driver on the keyword view `prepare_arguments` returned and
/// before the binding decodes from it. It rewrites that view in place, so the
/// hooks that returned it see the rewrite too; a rejection fails the call as a host
/// failure, so the hooks still observe it.
pub type Preflight = fn(Python<'_>, &Bound<'_, PyDict>) -> PyResult<()>;

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
}

pub type PythonCallEvent<'a> = RuntimeCallEvent<'a, PythonRuntime>;

pub trait PythonCallHooks: CallHooks<PythonRuntime> + PythonOwned {}

impl<H: CallHooks<PythonRuntime> + PythonOwned> PythonCallHooks for H {}
