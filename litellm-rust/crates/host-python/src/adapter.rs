use litellm_callbacks::event::{FailureOrigin, MachineEvent, RequestContext, Timing, WireRequest};
use litellm_callbacks::route::Route;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub fn missing_state() -> PyErr {
    PyRuntimeError::new_err("missing native call state")
}

/// What an adapter step produced: either the value the driver asked for, or a Python
/// awaitable the driver hands back to the caller's task before asking again.
pub enum LifecycleStep {
    Await(Py<PyAny>),
    Arguments(Py<PyDict>),
    Wire(Box<WireRequest>),
    Response(Py<PyAny>),
    Done,
}

/// What a lifecycle observes: the driver's start, the machine's own events, and one
/// terminal event carrying the public value the caller receives.
pub enum LifecycleEvent<'a> {
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

/// One consumer of a call's lifecycle on the Python side. The driver calls the steps in
/// order: `begin` before the machine starts, `before_send` and `emit` while it runs,
/// `after_success` and one terminal `emit` after it completes. Whenever a step returns
/// [`LifecycleStep::Await`], the driver awaits it in the caller's task and continues the
/// same step through `resume`.
///
/// A step that fails with an ordinary exception fails the call with that exception,
/// except on a terminal event, where the adapter is expected to report and swallow its
/// own errors. An exception that is not a `PyException`, such as a cancellation, ends
/// the call without further dispatch.
pub trait PythonLifecycle: Send + Sync {
    fn begin(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<LifecycleStep>;

    fn before_send(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<LifecycleStep>;

    fn after_success(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<LifecycleStep>;

    fn emit(
        &mut self,
        py: Python<'_>,
        event: LifecycleEvent<'_>,
    ) -> PyResult<LifecycleStep>;

    /// The call streams and its stream was handed to the caller. The caller is not
    /// inside an await here, so this step and `delivered` cannot suspend.
    fn opened(&mut self, py: Python<'_>) -> PyResult<()>;

    /// One chunk of an open stream is about to reach the caller.
    fn delivered(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()>;

    fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep>;

    fn close(&mut self, py: Python<'_>);

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

/// Why a route operation the host answered did not produce a result: the route's own code
/// rejected it, which the route classifies like any other native failure, or Python code
/// raised, which reaches the caller as it was raised.
#[derive(Debug)]
pub enum HostOpError<E> {
    Native(E),
    Python(PyErr),
}

impl<E> From<PyErr> for HostOpError<E> {
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
    ) -> Result<<Self::Route as Route>::OpResult, HostOpError<<Self::Route as Route>::Error>>;

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
