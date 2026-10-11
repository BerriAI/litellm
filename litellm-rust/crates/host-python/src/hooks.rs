mod chain;
pub use chain::HookChain;

use crate::{PythonOwned, PythonTask};
use litellm_host::hooks::{CallHooks, HookRuntime, RuntimeCallEvent};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python hooks run on the caller's task: each stage gets the [`PythonTask`] and awaits any
/// Python awaitable through it, so a hook is a plain `async fn`.
pub struct PythonRuntime;

impl HookRuntime for PythonRuntime {
    type Context = PythonTask;
    type Arguments = Py<PyDict>;
    type Response = Py<PyAny>;
    type Chunk = Py<PyAny>;
    type Error = PyErr;
}

pub type PythonCallEvent<'a> = RuntimeCallEvent<'a, PythonRuntime>;

pub trait PythonCallHooks: CallHooks<PythonRuntime> + PythonOwned {}

impl<H: CallHooks<PythonRuntime> + PythonOwned> PythonCallHooks for H {}
