use litellm_core::call_lifecycle::host::{
    HostCall as NativeCall, HostCallFuture, HostCallStep as NativeCallStep, HostFailure,
};
use pyo3::exceptions::PyBaseException;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::dispatch::{PendingLogging, PendingSuccess};
use super::handle::{Execution, ExecutionBody, ExecutionStep};
use super::*;
use pyo3::types::PyDict;
use std::sync::Mutex;

static PYTHON_GLOBALS: Mutex<()> = Mutex::new(());

fn install_logging_worker(py: Python<'_>, worker: &Bound<'_, PyAny>) -> PyResult<()> {
    py.import("litellm.litellm_core_utils.logging_worker")?
        .setattr("GLOBAL_LOGGING_WORKER", worker)
}

struct RetainingHost {
    retained: Option<Py<PyAny>>,
}

impl ExecutionBody for RetainingHost {
    fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| Ok(ExecutionStep::Return(py.None())))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.retained)
    }
}

#[pyfunction]
fn retaining_coroutine(py: Python<'_>, retained: Py<PyAny>) -> PyResult<Py<Execution>> {
    Py::new(
        py,
        Execution::new(RetainingHost {
            retained: Some(retained),
        }),
    )
}

struct AwaitBody(Option<Py<PyAny>>);

impl ExecutionBody for AwaitBody {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        match self.0.take() {
            Some(awaitable) => Ok(ExecutionStep::Await(awaitable)),
            None => result
                .expect("selected await completed")
                .map(ExecutionStep::Return),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }
}

#[pyfunction]
fn await_execution(awaitable: Py<PyAny>) -> Execution {
    Execution::new(AwaitBody(Some(awaitable)))
}

struct CallingBody(Py<PyAny>);

impl ExecutionBody for CallingBody {
    fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| self.0.call0(py).map(ExecutionStep::Return))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }
}

#[pyfunction]
fn calling_execution(callback: Py<PyAny>) -> Execution {
    Execution::new(CallingBody(callback))
}

struct SyntheticCall(bool);

impl NativeCall for SyntheticCall {
    type Operation = ();
    type Result = ();
    type Complete = ();

    fn resume(
        &mut self,
        result: Option<Self::Result>,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete> {
        Box::pin(async move {
            match (self.0, result) {
                (false, None) => {
                    self.0 = true;
                    Ok(NativeCallStep::Host(()))
                }
                (true, Some(())) => Ok(NativeCallStep::Complete(())),
                _ => Err(litellm_core::Error::InvalidRequest(
                    "invalid synthetic lifecycle state".into(),
                )),
            }
        })
    }

    fn interrupt(&mut self, _: HostFailure) -> HostCallFuture<'_, Self::Operation, Self::Complete> {
        Box::pin(async { Ok(NativeCallStep::Complete(())) })
    }
}

struct SyntheticRoute(PythonCallState);

impl PythonRoute for SyntheticRoute {
    type Call = SyntheticCall;

    fn state(&self) -> &PythonCallState {
        &self.0
    }

    fn state_mut(&mut self) -> &mut PythonCallState {
        &mut self.0
    }

    fn classify(_: &()) -> OperationClass {
        OperationClass::Route
    }

    fn lifecycle_result() {}

    fn map_error(error: litellm_core::Error) -> PyErr {
        crate::errors::core_error_to_pyerr(error)
    }

    fn invoke(&mut self, py: Python<'_>, _: ()) -> PyResult<()> {
        self.0.response = Some(
            pyo3::types::PyString::new(py, "shared lifecycle")
                .into_any()
                .unbind(),
        );
        Ok(())
    }

    fn cleanup(&mut self) {}

    fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}

#[test]
fn shared_runner_executes_a_non_ocr_adapter() {
    Python::initialize();
    Python::attach(|py| {
        let route = SyntheticRoute(
            PythonCallState::new(
                py,
                PyTuple::empty(py).unbind(),
                PyDict::new(py).unbind(),
                false,
                "synthetic",
            )
            .unwrap(),
        );
        let value: String = run_call(py, SyntheticCall(false), route)
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(value, "shared lifecycle");
    });
}

#[test]
fn ready_native_lifecycle_completes_without_scheduling() {
    let _guard = PYTHON_GLOBALS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    Python::initialize();
    Python::attach(|py| {
        let source = std::ffi::CString::new(include_str!(
            "../../../../../litellm/rust_bridge/lifecycle.py"
        ))
        .unwrap();
        PyModule::from_code(
            py,
            &source,
            pyo3::ffi::c_str!("lifecycle.py"),
            pyo3::ffi::c_str!("litellm.rust_bridge.lifecycle"),
        )
        .unwrap();
        let route = SyntheticRoute(
            PythonCallState::new(
                py,
                PyTuple::empty(py).unbind(),
                PyDict::new(py).unbind(),
                true,
                "synthetic",
            )
            .unwrap(),
        );
        let coroutine = run_call(py, SyntheticCall(false), route).unwrap();
        let completed = coroutine
            .call_method1(py, "send", (py.None(),))
            .unwrap_err();
        assert!(completed.is_instance_of::<pyo3::exceptions::PyStopIteration>(py));
        assert_eq!(
            completed
                .value(py)
                .getattr("value")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "shared lifecycle",
        );
    });
}

#[test]
fn python_driver_preserves_inline_await_and_native_ownership() {
    let _guard = PYTHON_GLOBALS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    Python::initialize();
    Python::attach(|py| {
        py.import("asyncio").unwrap();
        let source = std::ffi::CString::new(include_str!(
            "../../../../../litellm/rust_bridge/lifecycle.py"
        ))
        .unwrap();
        let module = PyModule::from_code(
            py,
            &source,
            pyo3::ffi::c_str!("lifecycle.py"),
            pyo3::ffi::c_str!("litellm.rust_bridge.lifecycle"),
        )
        .unwrap();
        let locals = PyDict::new(py);
        locals
            .set_item("drive", module.getattr("drive").unwrap())
            .unwrap();
        locals
            .set_item(
                "await_execution",
                wrap_pyfunction!(await_execution, py).unwrap(),
            )
            .unwrap();
        locals
            .set_item(
                "calling_execution",
                wrap_pyfunction!(calling_execution, py).unwrap(),
            )
            .unwrap();
        let probe = std::ffi::CString::new(include_str!("../../tests/lifecycle.py")).unwrap();
        py.run(&probe, Some(&locals), Some(&locals)).unwrap();
    });
}

struct ErrorBody(PythonCallState);

impl ExecutionBody for ErrorBody {
    fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| {
            Err(PyErr::from_value(
                self.0.error.take().unwrap().into_bound(py).into_any(),
            ))
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.0.traverse(visit)
    }
}

#[pyfunction]
fn error_execution(py: Python<'_>, error: Bound<'_, PyBaseException>) -> Execution {
    let mut state = PythonCallState::new(
        py,
        PyTuple::empty(py).unbind(),
        PyDict::new(py).unbind(),
        true,
        "test",
    )
    .unwrap();
    state.retain_error(py, PyErr::from_value(error.into_any()));
    Execution::new(ErrorBody(state))
}

#[test]
fn retained_exception_frames_and_duplicate_argument_edges_are_collectable() {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "error_execution",
                wrap_pyfunction!(error_execution, py).unwrap(),
            )
            .unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import gc
import weakref

class Retained:
    pass

def cycle():
    retained = Retained()
    try:
        raise ValueError('retained traceback')
    except ValueError as error:
        retained.owner = error_execution(error)
    return weakref.ref(retained)

reference = cycle()
gc.collect()
assert reference() is None
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}

fn state(
    py: Python<'_>,
    logger: Py<PyAny>,
    response: Py<PyAny>,
    asynchronous: bool,
) -> PythonCallState {
    PythonCallState {
        args: PyTuple::empty(py).unbind(),
        kwargs: PyDict::new(py).unbind(),
        logger: Some(logger.extract(py).unwrap()),
        start: py.None(),
        end: Some(py.None()),
        response: Some(response),
        error: None,
        asynchronous,
        internal: false,
        call_type: "test",
    }
}

#[test]
fn success_dispatch_reports_ordinary_failures_without_replacing_response() {
    let _guard = PYTHON_GLOBALS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                r#"
import sys

response = object()
failure = ValueError('terminal diagnostic')
diagnostics = []
old_hook = sys.unraisablehook
sys.unraisablehook = lambda event: diagnostics.append(event.exc_value)

class Logger:
    def handle_sync_success_callbacks_for_async_calls(self, *args):
        raise failure

logger = Logger()
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        let response = locals.get_item("response").unwrap().unwrap().unbind();
        let mut lifecycle_state = state(
            py,
            locals.get_item("logger").unwrap().unwrap().unbind(),
            response.clone_ref(py),
            true,
        );
        lifecycle_state.internal = true;
        lifecycle_state.dispatch_success(py).unwrap();
        assert!(lifecycle_state.response.as_ref().unwrap().is(&response));
        py.run(
            pyo3::ffi::c_str!(
                r#"
assert diagnostics == [failure]
sys.unraisablehook = old_hook
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}

#[test]
fn retained_failure_preserves_exception_identity() {
    Python::initialize();
    Python::attach(|py| {
        let logger = PyDict::new(py).into_any().unbind();
        let response = py.None();
        let failure = pyo3::exceptions::PyValueError::new_err("identity");
        let failure_value = failure.value(py).clone().unbind();
        let mut lifecycle_state = state(py, logger, response, false);
        lifecycle_state.retain_error(py, failure);
        let retained = lifecycle_state.error.take().unwrap();
        assert!(retained.is(&failure_value));
    });
}

#[test]
fn deferred_release_uses_release_context_and_allows_reentry_once() {
    let _guard = PYTHON_GLOBALS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                r#"
import sys
import types
from contextvars import ContextVar

litellm = types.ModuleType('litellm')
core_utils = types.ModuleType('litellm.litellm_core_utils')
logging_worker = types.ModuleType('litellm.litellm_core_utils.logging_worker')
litellm.litellm_core_utils = core_utils
core_utils.logging_worker = logging_worker
sys.modules['litellm'] = litellm
sys.modules['litellm.litellm_core_utils'] = core_utils
sys.modules['litellm.litellm_core_utils.logging_worker'] = logging_worker

marker = ContextVar('marker', default='unset')
observed = []

class Coroutine:
    def close(self):
        observed.append('closed')

class Worker:
    def ensure_initialized_and_enqueue(self, coroutine):
        observed.append(marker.get())
        pending.release(True)
        coroutine.close()

class Logger:
    def async_success_handler(self, *args):
        observed.append('created')
        return Coroutine()

worker = Worker()
logger = Logger()
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        install_logging_worker(py, &locals.get_item("worker").unwrap().unwrap()).unwrap();
        let pending = Py::new(
            py,
            PendingLogging::new(PendingSuccess {
                logger: locals
                    .get_item("logger")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap(),
                response: Some(py.None()),
                start: py.None(),
                end: Some(py.None()),
            }),
        )
        .unwrap();
        locals.set_item("pending", &pending).unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
marker.set('release')
pending.release(True)
pending.release(True)
assert observed == ['created', 'release', 'closed']
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}

#[test]
fn deferred_logging_collects_cycles_through_typed_logger() {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!("class Logger: pass\nlogger = Logger()"),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        let pending = Py::new(
            py,
            PendingLogging::new(PendingSuccess {
                logger: locals
                    .get_item("logger")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap(),
                response: None,
                start: py.None(),
                end: None,
            }),
        )
        .unwrap();
        locals.set_item("pending", pending).unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import gc
import weakref
logger.pending = pending
reference = weakref.ref(logger)
del logger, pending
gc.collect()
assert reference() is None
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}

#[test]
fn coroutine_collects_cycles_retained_by_bridge_host() {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "retaining_coroutine",
                wrap_pyfunction!(retaining_coroutine, py).unwrap(),
            )
            .unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import gc
import weakref

class Retained:
    pass

def cycle():
    retained = Retained()
    coroutine = retaining_coroutine(retained)
    retained.coroutine = coroutine
    return weakref.ref(retained)

retained_ref = cycle()
gc.collect()
assert retained_ref() is None
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}
