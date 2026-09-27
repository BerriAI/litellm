use crate::PythonOwned;
use litellm_host::event::{FailureOrigin, MachineEvent, RequestContext, Timing, WireRequest};
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

/// Events dispatched to call hooks: the driver's start, the machine's own events, and one
/// terminal event carrying the public value the caller receives.
pub enum HookEvent<'a> {
    Started {
        start_time: f64,
    },
    Machine(&'a MachineEvent),
    Succeeded {
        timing: Timing,
        response: &'a Py<PyAny>,
    },
    Failed {
        timing: Timing,
        origin: FailureOrigin,
        error: &'a PyErr,
    },
}

/// Active Python hooks that can transform values or fail execution. The driver calls the steps in
/// order: `prepare_arguments` before the machine starts, `before_provider_request` and `on_event` while it runs,
/// `transform_response` and one terminal `on_event` after it completes. Whenever a step returns
/// [`HookStep::Await`], the driver awaits it in the caller's task and continues the
/// same step through its typed continuation.
///
/// A step that fails with an ordinary exception fails the call with that exception,
/// except on a terminal event, where the hooks are expected to report and swallow their
/// own errors. An exception that is not a `PyException`, such as a cancellation, ends
/// the call without further dispatch.
pub trait PythonCallHooks: Sized + PythonOwned {
    fn prepare_arguments(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<HookStep<Self, Py<PyDict>>>;

    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>>;

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>>;

    fn on_event(&mut self, py: Python<'_>, event: HookEvent<'_>) -> PyResult<HookStep<Self, ()>>;

    /// The call streams and its stream was handed to the caller. The caller is not
    /// inside an await here, so this step and `on_stream_chunk` cannot suspend.
    fn on_stream_open(&mut self, py: Python<'_>) -> PyResult<()>;

    /// One chunk of an open stream is about to reach the caller.
    fn on_stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()>;
}
