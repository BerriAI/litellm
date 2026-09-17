use std::sync::Arc;
use std::task::Poll;

use futures_util::future::{AbortHandle, Abortable};
#[cfg(test)]
use litellm_callbacks::protocol::NativeCallFuture;
use litellm_callbacks::protocol::{HostFailure, HostPhase, HostStep, NativeCall, NativeCallStep};
pub(crate) use litellm_callbacks_legacy::{PythonCallState, PythonLogger, missing_state, now};
use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use tokio::sync::Mutex;

use crate::execution::{poll_async_value, run_async_value, run_sync_value};

mod handle;

use handle::{Execution, ExecutionBody, ExecutionStep};

pub(crate) enum OperationClass {
    Phase(HostPhase),
    Route,
}

pub(crate) trait PythonHost: Send + Sync {
    type Call: NativeCall + 'static;

    fn state(&self) -> &PythonCallState;
    fn state_mut(&mut self) -> &mut PythonCallState;
    fn classify(operation: &<Self::Call as NativeCall>::Operation) -> OperationClass;
    fn lifecycle_result() -> <Self::Call as NativeCall>::Result;
    fn map_error(error: <Self::Call as NativeCall>::Error) -> PyErr;
    fn host_error(message: String) -> <Self::Call as NativeCall>::Error;
    fn invoke(
        &mut self,
        py: Python<'_>,
        operation: <Self::Call as NativeCall>::Operation,
    ) -> PyResult<<Self::Call as NativeCall>::Result>;
    fn cleanup(&mut self);
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

type NativeStep<C> = NativeCallStep<<C as NativeCall>::Operation, <C as NativeCall>::Complete>;
type NativeResult<C> = Result<NativeStep<C>, <C as NativeCall>::Error>;
type HostResumeStep<R> = HostStep<NativeStep<<R as PythonHost>::Call>, Py<PyAny>>;
type NativeResume<C> =
    Option<Result<<C as NativeCall>::Result, HostFailure<<C as NativeCall>::Error>>>;

struct NativeCallState<C: NativeCall> {
    call: C,
    result: Option<NativeResult<C>>,
}

enum PendingOperation {
    Native,
    Host(HostPhase),
}

struct PythonLifecycle<R: PythonHost> {
    route: R,
    call: Option<Arc<Mutex<NativeCallState<R::Call>>>>,
    pending: Option<PendingOperation>,
    native_abort: Option<AbortHandle>,
}

pub(crate) fn run_call<R: PythonHost + 'static>(
    py: Python<'_>,
    call: R::Call,
    route: R,
) -> PyResult<Py<PyAny>> {
    let asynchronous = route.state().asynchronous;
    let mut lifecycle = PythonLifecycle {
        route,
        call: Some(Arc::new(Mutex::new(NativeCallState { call, result: None }))),
        pending: None,
        native_abort: None,
    };
    if asynchronous {
        let execution = Py::new(py, Execution::new(lifecycle))?;
        return py
            .import("litellm.rust_bridge.lifecycle")?
            .getattr("drive")?
            .call1((execution,))
            .map(Bound::unbind);
    }
    match lifecycle.resume(None)? {
        ExecutionStep::Return(value) => Ok(value),
        ExecutionStep::Await(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "sync call suspended",
        )),
    }
}

impl<R: PythonHost> PythonLifecycle<R> {
    fn resume_core(
        &mut self,
        py: Python<'_>,
        result: NativeResume<R::Call>,
    ) -> PyResult<HostResumeStep<R>> {
        let call = Arc::clone(self.call.as_ref().ok_or_else(missing_state)?);
        let future = async move {
            let mut call = call.lock().await;
            let result = match result {
                Some(Err(failure)) => call.call.interrupt(failure).await,
                Some(Ok(result)) => call.call.resume(Some(result)).await,
                None => call.call.resume(None).await,
            };
            call.result = Some(result);
            Ok(())
        };
        if self.route.state().asynchronous {
            let mut future = Box::pin(future);
            if let Poll::Ready(()) = poll_async_value(py, future.as_mut())? {
                return Ok(HostStep::Ready(self.take_native_result()?));
            }
            let (abort, registration) = AbortHandle::new_pair();
            self.native_abort = Some(abort);
            self.pending = Some(PendingOperation::Native);
            Ok(HostStep::Suspend(
                run_async_value(py, async move {
                    Abortable::new(future, registration)
                        .await
                        .map_err(|_| PyRuntimeError::new_err("native execution closed"))?
                })?
                .unbind(),
            ))
        } else {
            run_sync_value(py, future)?;
            Ok(HostStep::Ready(self.take_native_result()?))
        }
    }

    fn take_native_result(&self) -> PyResult<NativeStep<R::Call>> {
        self.call
            .as_ref()
            .ok_or_else(missing_state)?
            .try_lock()
            .map_err(|_| missing_state())?
            .result
            .take()
            .ok_or_else(missing_state)?
            .map_err(R::map_error)
    }

    fn host_failure(
        &mut self,
        py: Python<'_>,
        error: PyErr,
        phase: Option<HostPhase>,
    ) -> HostFailure<<R::Call as NativeCall>::Error> {
        let native = R::host_error(error.to_string());
        let cancelled = !error.is_instance_of::<PyException>(py);
        let failure = if !cancelled {
            HostFailure::Error(native)
        } else {
            HostFailure::Cancelled(native)
        };
        self.route
            .state_mut()
            .record_failure(py, error, cancelled, phase);
        failure
    }

    fn drive(
        &mut self,
        py: Python<'_>,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<ExecutionStep> {
        let mut step = match (self.pending.take(), result) {
            (None, None) => self.resume_core(py, None)?,
            (Some(PendingOperation::Native), Some(result)) => match result {
                Ok(_) => HostStep::Ready(self.take_native_result()?),
                Err(error) => {
                    let failure = self.host_failure(py, error, None);
                    self.resume_core(py, Some(Err(failure)))?
                }
            },
            (Some(PendingOperation::Host(phase)), Some(result)) => {
                let result =
                    result.and_then(|value| self.route.state_mut().accept(py, phase, value));
                let result = match result {
                    Ok(()) => Ok(R::lifecycle_result()),
                    Err(error) => Err(self.host_failure(py, error, Some(phase))),
                };
                self.resume_core(py, Some(result))?
            }
            _ => return Err(missing_state()),
        };
        loop {
            let operation = match step {
                HostStep::Suspend(awaitable) => return Ok(ExecutionStep::Await(awaitable)),
                HostStep::Ready(NativeCallStep::Complete(_)) => {
                    return self
                        .route
                        .state_mut()
                        .take_response()
                        .map(ExecutionStep::Return)
                        .ok_or_else(missing_state);
                }
                HostStep::Ready(NativeCallStep::Host(operation)) => operation,
            };
            let phase = match R::classify(&operation) {
                OperationClass::Phase(phase) => Some(phase),
                OperationClass::Route => None,
            };
            let result = match phase {
                Some(phase) => match self.route.state_mut().invoke(py, phase) {
                    Ok(HostStep::Suspend(awaitable)) => {
                        self.pending = Some(PendingOperation::Host(phase));
                        return Ok(ExecutionStep::Await(awaitable));
                    }
                    Ok(HostStep::Ready(value)) => self
                        .route
                        .state_mut()
                        .accept(py, phase, value)
                        .map(|()| R::lifecycle_result()),
                    Err(error) => Err(error),
                },
                None => self.route.invoke(py, operation),
            };
            let result = match result {
                Ok(result) => Ok(result),
                Err(error) => Err(self.host_failure(py, error, phase)),
            };
            step = self.resume_core(py, Some(result))?;
        }
    }
}

impl<R: PythonHost> ExecutionBody for PythonLifecycle<R> {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        let result = Python::attach(|py| self.drive(py, result));
        match result {
            Ok(ExecutionStep::Await(value)) => Ok(ExecutionStep::Await(value)),
            result => result.map_err(|error| {
                Python::attach(|py| self.route.state_mut().take_error(py).unwrap_or(error))
            }),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.route.state().traverse(visit)?;
        self.route.traverse(visit)
    }
}

impl<R: PythonHost> PythonLifecycle<R> {
    fn clear(&mut self) {
        if let Some(abort) = self.native_abort.take() {
            abort.abort();
        }
        if self.call.take().is_some() {
            Python::attach(|py| self.route.state_mut().cleanup(py));
            self.route.cleanup();
        }
    }
}

impl<R: PythonHost> Drop for PythonLifecycle<R> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyBaseException;
    use pyo3::types::{PyDict, PyTuple};
    use std::sync::Mutex;

    static PYTHON_GLOBALS: Mutex<()> = Mutex::new(());

    fn install_lifecycle_module(py: Python<'_>) -> Bound<'_, PyModule> {
        py.run(
            pyo3::ffi::c_str!(
                r#"
import sys
import types

sys.modules.setdefault('litellm', types.ModuleType('litellm'))
sys.modules.setdefault('litellm.rust_bridge', types.ModuleType('litellm.rust_bridge'))
"#
            ),
            None,
            None,
        )
        .unwrap();
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
        .unwrap()
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
        type Error = litellm_core::messages::Error;
        type Operation = ();
        type Result = ();
        type Complete = ();

        fn resume(
            &mut self,
            result: Option<Self::Result>,
        ) -> NativeCallFuture<'_, Self::Operation, Self::Complete, Self::Error> {
            Box::pin(async move {
                match (self.0, result) {
                    (false, None) => {
                        self.0 = true;
                        Ok(NativeCallStep::Host(()))
                    }
                    (true, Some(())) => Ok(NativeCallStep::Complete(())),
                    _ => Err(litellm_core::messages::Error::InvalidRequest(
                        "invalid synthetic lifecycle state".into(),
                    )),
                }
            })
        }

        fn interrupt(
            &mut self,
            _: HostFailure<Self::Error>,
        ) -> NativeCallFuture<'_, Self::Operation, Self::Complete, Self::Error> {
            Box::pin(async { Ok(NativeCallStep::Complete(())) })
        }
    }

    struct SyntheticRoute(PythonCallState);

    impl PythonHost for SyntheticRoute {
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

        fn map_error(error: litellm_core::messages::Error) -> PyErr {
            crate::errors::messages_error_to_pyerr(error)
        }

        fn host_error(message: String) -> litellm_core::messages::Error {
            litellm_core::messages::Error::InvalidRequest(message)
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
            install_lifecycle_module(py);
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
            let module = install_lifecycle_module(py);
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
            Python::attach(|py| Err(self.0.take_error(py).unwrap()))
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
}
