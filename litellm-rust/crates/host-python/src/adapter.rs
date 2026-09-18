use litellm_callbacks::event::{CallEvent, RequestContext, Timing, WireRequest};
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
pub enum AdapterStep {
    Await(Py<PyAny>),
    Arguments(Py<PyDict>),
    Wire(Box<WireRequest>),
    Response(Py<PyAny>),
    Done,
}

/// The host-typed value the driver attaches to a terminal event.
pub enum PublicValue<'a> {
    Response(&'a Py<PyAny>),
    Error(&'a PyErr),
}

/// One consumer of a call's lifecycle on the Python side. The driver calls the steps in
/// order: `begin` before the machine starts, `before_send` and `emit` while it runs,
/// `after_success` and one terminal `emit` after it completes. Whenever a step returns
/// [`AdapterStep::Await`], the driver awaits it in the caller's task and continues the
/// same step through `resume`.
///
/// A step that fails with an ordinary exception fails the call with that exception,
/// except on a terminal event, where the adapter is expected to report and swallow its
/// own errors. An exception that is not a `PyException`, such as a cancellation, ends
/// the call without further dispatch.
pub trait CallbackAdapter: Send + Sync {
    fn begin(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        started_at: f64,
    ) -> PyResult<AdapterStep>;

    fn before_send(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<AdapterStep>;

    fn after_success(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<AdapterStep>;

    fn emit(
        &mut self,
        py: Python<'_>,
        event: &CallEvent,
        public: Option<PublicValue<'_>>,
    ) -> PyResult<AdapterStep>;

    fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<AdapterStep>;

    fn close(&mut self, py: Python<'_>);

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

/// The Python side of one route: answers the route's own operations, builds the public
/// response and maps failures to public exceptions.
pub trait RouteHost: Send + Sync {
    type Route: Route;

    fn invoke(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        op: <Self::Route as Route>::Op,
    ) -> PyResult<<Self::Route as Route>::OpResult>;

    fn complete(
        &mut self,
        py: Python<'_>,
        response: <Self::Route as Route>::Response,
    ) -> PyResult<Py<PyAny>>;

    fn native_error(error: <Self::Route as Route>::Error) -> PyErr;

    fn host_error(error: &PyErr) -> <Self::Route as Route>::Error;

    fn map_failure(&self, py: Python<'_>, error: &PyErr) -> PyResult<PyErr>;

    fn close(&mut self, py: Python<'_>);

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}
