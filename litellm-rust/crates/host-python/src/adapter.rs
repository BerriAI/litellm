use litellm_host::event::{FailureOrigin, MachineEvent, RequestContext, Timing, WireRequest};
use litellm_host::route::Route;
use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Whether an error ends the call outright: anything that is not a `PyException`, such
/// as `CancelledError` or `KeyboardInterrupt`. A cancellation is never reported, never
/// swallowed and never followed by further dispatch.
pub fn is_cancellation(py: Python<'_>, error: &PyErr) -> bool {
    !error.is_instance_of::<PyException>(py)
}

pub fn missing_state() -> PyErr {
    PyRuntimeError::new_err("missing native call state")
}

/// What a lifecycle step produced: either the value the driver asked for, or a Python
/// awaitable the driver hands back to the caller's task before asking again.
pub enum LifecycleStep {
    Await(Py<PyAny>),
    Arguments(Py<PyDict>),
    Wire(Box<WireRequest>),
    Response(Py<PyAny>),
    Done,
}

/// What a lifecycle observes: the machine's own events, and one terminal event
/// carrying the public value the caller receives.
#[derive(Clone, Copy)]
pub enum LifecycleEvent<'a> {
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

/// One consumer of a call's lifecycle on the Python side: the callback capability of the
/// Python host, next to [`RouteHost`], the route capability.
///
/// The steps come in two kinds. A threading step receives a value and returns the value
/// the call continues with, so a lifecycle may rewrite it; each defaults to returning
/// what it was given. An observing step receives an event or a notice and returns
/// nothing to the call. Two steps answer an operation the machine raised
/// ([`HostOp`](litellm_host::host::HostOp)); the rest the driver originates around it:
///
/// | step | kind | raised by |
/// |---|---|---|
/// | `begin` | threads the keyword view | driver, before the machine starts |
/// | `before_send` | threads the wire request | machine, `HostOp::BeforeSend` |
/// | `emit(Machine)` | observes | machine, `HostOp::Emit` |
/// | `opened`, `delivered` | observe | driver, answering `HostOp::Open` and `Deliver` |
/// | `after_success` | threads the public response | driver, after the machine completes |
/// | `emit(Succeeded \| Failed)` | observes | driver, exactly once, last |
///
/// Whenever a step returns [`LifecycleStep::Await`], the driver awaits it in the caller's
/// task and continues the same step through `resume`.
///
/// A step that fails with an ordinary exception fails the call with that exception,
/// except on a terminal event, where the lifecycle is expected to report and swallow its
/// own errors. A cancellation ([`is_cancellation`]) ends the call without further
/// dispatch.
pub trait PythonLifecycle: Send + Sync {
    /// `start_time` is the call's start in epoch seconds, the same value the terminal
    /// event's `Timing` carries.
    fn begin(
        &mut self,
        _py: Python<'_>,
        arguments: Py<PyDict>,
        _start_time: f64,
    ) -> PyResult<LifecycleStep> {
        Ok(LifecycleStep::Arguments(arguments))
    }

    fn before_send(
        &mut self,
        _py: Python<'_>,
        wire: Box<WireRequest>,
        _context: &RequestContext,
    ) -> PyResult<LifecycleStep> {
        Ok(LifecycleStep::Wire(wire))
    }

    fn after_success(
        &mut self,
        _py: Python<'_>,
        response: Py<PyAny>,
        _timing: Timing,
    ) -> PyResult<LifecycleStep> {
        Ok(LifecycleStep::Response(response))
    }

    fn emit(&mut self, py: Python<'_>, event: LifecycleEvent<'_>) -> PyResult<LifecycleStep>;

    /// The call streams and its stream was handed to the caller. The caller is not
    /// inside an await here, so this step and `delivered` cannot suspend.
    fn opened(&mut self, _py: Python<'_>) -> PyResult<()> {
        Ok(())
    }

    /// One chunk of an open stream is about to reach the caller.
    fn delivered(&mut self, _py: Python<'_>, _chunk: &Py<PyAny>) -> PyResult<()> {
        Ok(())
    }

    fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep>;

    fn close(&mut self, py: Python<'_>);

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

/// Why a route operation the host answered did not produce a result: the route's own code
/// rejected it, which the route classifies like any other native failure, or Python code
/// raised, which reaches the caller as it was raised.
#[derive(Debug)]
pub enum InvokeError<E> {
    Native(E),
    Python(PyErr),
}

impl<E> From<PyErr> for InvokeError<E> {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

/// The Python side of one route: answers the route's own operations, builds the public
/// response and classifies native failures into public exceptions.
pub trait RouteHost: Send + Sync {
    type Route: Route<Error: std::fmt::Display>;

    /// The public exception a native failure maps to, kept as a value until the driver
    /// raises it.
    type Failure: Into<PyErr>;

    /// `arguments` is the keyword view the lifecycle's `begin` produced, not the
    /// caller's own dict. A route host that projects from it inherits whatever that
    /// adapter rewrote.
    fn invoke(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        op: <Self::Route as Route>::Op,
    ) -> Result<<Self::Route as Route>::OpResult, InvokeError<<Self::Route as Route>::Error>>;

    fn complete(
        &mut self,
        py: Python<'_>,
        response: <Self::Route as Route>::Response,
    ) -> PyResult<Py<PyAny>>;

    /// One streamed chunk as the caller receives it.
    fn chunk(
        &mut self,
        py: Python<'_>,
        chunk: <Self::Route as Route>::Chunk,
    ) -> PyResult<Py<PyAny>>;

    fn classify(
        &self,
        py: Python<'_>,
        error: <Self::Route as Route>::Error,
    ) -> PyResult<Self::Failure>;

    fn host_error(error: &PyErr) -> <Self::Route as Route>::Error;

    fn close(&mut self, py: Python<'_>);

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}
