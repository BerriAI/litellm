use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::future::{AbortHandle, Abortable};
use litellm_core::call_lifecycle::host::{HostFailure, HostPhase, HostStep};
use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use tokio::sync::Mutex;

use crate::execution::{run_async_value, run_sync_value};

mod bindings;
mod handle;
mod preparation;

use bindings::DeploymentHooks;
pub(crate) use bindings::PythonLogger;
use handle::{Execution, ExecutionBody, ExecutionStep};

pub(crate) enum NativeCallStep<O> {
    Host(O),
    Complete,
}

pub(crate) trait NativeCall: Send + Sync {
    type Operation: Send + 'static;
    type Result: Send + 'static;

    fn resume(
        &mut self,
        result: Option<Self::Result>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<NativeCallStep<Self::Operation>, litellm_core::Error>>
                + Send
                + '_,
        >,
    >;

    fn interrupt(
        &mut self,
        failure: HostFailure,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<NativeCallStep<Self::Operation>, litellm_core::Error>>
                + Send
                + '_,
        >,
    >;
}

pub(crate) enum OperationClass {
    Phase(HostPhase),
    Route,
}

pub(crate) trait PythonRoute: Send + Sync {
    type Call: NativeCall + 'static;

    fn state(&self) -> &PythonCallState;
    fn state_mut(&mut self) -> &mut PythonCallState;
    fn classify(operation: &<Self::Call as NativeCall>::Operation) -> OperationClass;
    fn lifecycle_result() -> <Self::Call as NativeCall>::Result;
    fn map_error(error: litellm_core::Error) -> PyErr;
    fn invoke(
        &mut self,
        py: Python<'_>,
        operation: <Self::Call as NativeCall>::Operation,
    ) -> PyResult<<Self::Call as NativeCall>::Result>;
    fn cleanup(&mut self);
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

struct NativeCallState<C: NativeCall> {
    call: C,
    result: Option<Result<NativeCallStep<C::Operation>, litellm_core::Error>>,
}

enum PendingOperation {
    Native,
    Host(HostPhase),
}

struct PythonLifecycle<R: PythonRoute> {
    route: R,
    call: Option<Arc<Mutex<NativeCallState<R::Call>>>>,
    pending: Option<PendingOperation>,
    native_abort: Option<AbortHandle>,
}

pub(crate) fn run_call<R: PythonRoute + 'static>(
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

pub(crate) fn missing_state() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("missing native call state")
}

impl<R: PythonRoute> PythonLifecycle<R> {
    fn resume_core(
        &mut self,
        py: Python<'_>,
        result: Option<Result<<R::Call as NativeCall>::Result, HostFailure>>,
    ) -> PyResult<HostStep<NativeCallStep<<R::Call as NativeCall>::Operation>, Py<PyAny>>> {
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

    fn take_native_result(&self) -> PyResult<NativeCallStep<<R::Call as NativeCall>::Operation>> {
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
    ) -> HostFailure {
        let native = litellm_core::Error::InvalidRequest(error.to_string());
        let cancelled = !error.is_instance_of::<PyException>(py);
        let failure = if !cancelled {
            HostFailure::Error(native)
        } else {
            HostFailure::Cancelled(native)
        };
        let state = self.route.state_mut();
        if state.error.is_none() || (cancelled && phase != Some(HostPhase::DeploymentFailure)) {
            state.retain_error(py, error);
        }
        if state.end.is_none() {
            state.end = now(py).ok();
        }
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
                HostStep::Ready(NativeCallStep::Complete) => {
                    return self
                        .route
                        .state_mut()
                        .response
                        .take()
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

impl<R: PythonRoute> ExecutionBody for PythonLifecycle<R> {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        let result = Python::attach(|py| self.drive(py, result));
        match result {
            Ok(ExecutionStep::Await(value)) => Ok(ExecutionStep::Await(value)),
            result => result.map_err(|error| {
                Python::attach(|py| {
                    self.route
                        .state_mut()
                        .error
                        .take()
                        .map(|value| PyErr::from_value(value.into_bound(py).into_any()))
                        .unwrap_or(error)
                })
            }),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.route.state().traverse(visit)?;
        self.route.traverse(visit)
    }
}

impl<R: PythonRoute> PythonLifecycle<R> {
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

impl<R: PythonRoute> Drop for PythonLifecycle<R> {
    fn drop(&mut self) {
        self.clear();
    }
}

pub(crate) struct PythonCallState {
    pub args: Py<PyTuple>,
    pub kwargs: Py<PyDict>,
    pub logger: Option<PythonLogger>,
    pub start: Py<PyAny>,
    pub end: Option<Py<PyAny>>,
    pub response: Option<Py<PyAny>>,
    pub error: Option<Py<PyBaseException>>,
    pub asynchronous: bool,
    pub internal: bool,
    pub call_type: &'static str,
}

pub(crate) fn now(py: Python<'_>) -> PyResult<Py<PyAny>> {
    py.import("datetime")?
        .getattr("datetime")?
        .call_method0("now")
        .map(Bound::unbind)
}

impl PythonCallState {
    fn invoke(
        &mut self,
        py: Python<'_>,
        phase: HostPhase,
    ) -> PyResult<HostStep<Py<PyAny>, Py<PyAny>>> {
        match phase {
            HostPhase::Setup => self.setup(py)?,
            HostPhase::DeploymentPreCall => {
                return Ok(HostStep::Suspend(DeploymentHooks::before_call(
                    py,
                    &self.kwargs,
                    self.call_type,
                )?));
            }
            HostPhase::Prepare => self.prepare(py)?,
            HostPhase::DeploymentPostCall => {
                return Ok(HostStep::Suspend(DeploymentHooks::after_success(
                    py,
                    &self.kwargs,
                    &self.response,
                    self.call_type,
                )?));
            }
            HostPhase::Finalize => self.finalize(py)?,
            HostPhase::Success => self.dispatch_success(py)?,
            HostPhase::DeploymentFailure => {
                if let Some(error) = &self.error {
                    return Ok(HostStep::Suspend(DeploymentHooks::after_failure(
                        py,
                        &self.kwargs,
                        error,
                        self.call_type,
                    )?));
                }
            }
            HostPhase::Failure | HostPhase::AsyncFailure => {
                if let Some(awaitable) =
                    self.dispatch_failure(py, phase == HostPhase::AsyncFailure)?
                {
                    return Ok(HostStep::Suspend(awaitable));
                }
            }
            HostPhase::Execute
            | HostPhase::ConstructResponse
            | HostPhase::MapFailure
            | HostPhase::Complete => return Err(missing_state()),
        }
        Ok(HostStep::Ready(py.None()))
    }

    fn accept(&mut self, py: Python<'_>, phase: HostPhase, value: Py<PyAny>) -> PyResult<()> {
        match phase {
            HostPhase::DeploymentPreCall => {
                self.kwargs = value.into_bound(py).cast_into::<PyDict>()?.unbind()
            }
            HostPhase::DeploymentPostCall => self.response = Some(value),
            _ => {}
        }
        Ok(())
    }

    pub fn new(
        py: Python<'_>,
        args: Py<PyTuple>,
        kwargs: Py<PyDict>,
        asynchronous: bool,
        call_type: &'static str,
    ) -> PyResult<Self> {
        Ok(Self {
            args,
            kwargs,
            logger: None,
            start: py.None(),
            end: None,
            response: None,
            error: None,
            asynchronous,
            internal: false,
            call_type,
        })
    }

    pub fn logger(&self) -> PyResult<&PythonLogger> {
        self.logger.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("call logging is not initialized")
        })
    }

    pub fn setup(&mut self, py: Python<'_>) -> PyResult<()> {
        self.start = now(py)?;
        self.internal = bindings::is_internal_call(py)?;
        let result = bindings::setup(
            py,
            self.call_type,
            &self.args,
            &self.kwargs,
            &self.start,
            self.asynchronous,
        )?;
        self.logger = Some(result.logger()?);
        self.kwargs = result.kwargs()?;
        Ok(())
    }

    pub fn prepare(&mut self, py: Python<'_>) -> PyResult<()> {
        self.kwargs = preparation::prepare(py, self.kwargs.bind(py), self.logger()?)?.unbind();
        Ok(())
    }

    pub fn finalize(&self, py: Python<'_>) -> PyResult<()> {
        bindings::finalize(
            py,
            &self.response,
            self.logger()?,
            &self.kwargs,
            &self.start,
            &self.end,
        )
    }

    pub fn dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        match self.try_dispatch_success(py) {
            Err(error) if error.is_instance_of::<PyException>(py) => {
                error.write_unraisable(py, self.logger.as_ref().map(|logger| logger.object(py)));
                Ok(())
            }
            result => result,
        }
    }

    fn try_dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        let logger = self.logger()?;
        let pending = PendingSuccess {
            logger: logger.clone_ref(py),
            response: self.response.as_ref().map(|value| value.clone_ref(py)),
            start: self.start.clone_ref(py),
            end: self.end.as_ref().map(|value| value.clone_ref(py)),
        };
        if !self.asynchronous {
            pending.sync(py)
        } else {
            if !self.internal
                && self
                    .kwargs
                    .bind(py)
                    .get_item("fallbacks")?
                    .is_none_or(|value| value.is_none())
            {
                if logger.defers_async_logging(py) {
                    logger.defer_success(
                        py,
                        Py::new(
                            py,
                            PendingLogging {
                                pending: Some(pending),
                            },
                        )?,
                    )?;
                } else {
                    pending.asynchronous(py)?;
                }
            }
            logger.sync_success_for_async_call(py, &self.response, &self.start, &self.end)
        }
    }

    pub fn dispatch_failure(
        &self,
        py: Python<'_>,
        asynchronous: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        if self.logger.is_none() || (self.asynchronous && self.internal) {
            return Ok(None);
        }
        let Some(error) = &self.error else {
            return Ok(None);
        };
        self.logger()?
            .failure(py, error, &self.start, &self.end, asynchronous)
    }

    pub fn cleanup(&mut self, py: Python<'_>) {
        if let Some(logger) = self.logger.take()
            && let Err(error) = logger.restore_context(py)
        {
            error.write_unraisable(py, None);
        }
    }

    pub fn retain_error(&mut self, py: Python<'_>, error: PyErr) {
        self.error = Some(error.into_value(py));
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.args)?;
        visit.call(&self.kwargs)?;
        if let Some(logger) = &self.logger {
            logger.traverse(visit)?;
        }
        visit.call(&self.start)?;
        visit.call(&self.end)?;
        visit.call(&self.response)?;
        visit.call(&self.error)
    }
}

struct PendingSuccess {
    logger: PythonLogger,
    response: Option<Py<PyAny>>,
    start: Py<PyAny>,
    end: Option<Py<PyAny>>,
}

impl PendingSuccess {
    fn sync(&self, py: Python<'_>) -> PyResult<()> {
        self.logger
            .submit_success(py, &self.response, &self.start, &self.end)
    }

    fn asynchronous(&self, py: Python<'_>) -> PyResult<()> {
        self.logger
            .enqueue_success(py, &self.response, &self.start, &self.end)
    }
}

#[pyclass]
struct PendingLogging {
    pending: Option<PendingSuccess>,
}

#[pymethods]
impl PendingLogging {
    fn release(slf: &Bound<'_, Self>, py: Python<'_>, success: bool) -> PyResult<()> {
        let pending = slf.borrow_mut().pending.take();
        if let Some(pending) = pending
            && success
        {
            match pending.asynchronous(py) {
                Err(error) if error.is_instance_of::<PyException>(py) => {
                    error.write_unraisable(py, Some(pending.logger.object(py)));
                }
                result => return result,
            }
        }
        Ok(())
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        if let Some(pending) = &self.pending {
            pending.logger.traverse(&visit)?;
            visit.call(&pending.response)?;
            visit.call(&pending.start)?;
            visit.call(&pending.end)?;
        }
        Ok(())
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let pending = slf.borrow_mut().pending.take();
        drop(pending);
    }
}

#[cfg(test)]
mod tests {
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

        fn resume(
            &mut self,
            result: Option<Self::Result>,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<NativeCallStep<Self::Operation>, litellm_core::Error>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async move {
                match (self.0, result) {
                    (false, None) => {
                        self.0 = true;
                        Ok(NativeCallStep::Host(()))
                    }
                    (true, Some(())) => Ok(NativeCallStep::Complete),
                    _ => Err(litellm_core::Error::InvalidRequest(
                        "invalid synthetic lifecycle state".into(),
                    )),
                }
            })
        }

        fn interrupt(
            &mut self,
            _: HostFailure,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<NativeCallStep<Self::Operation>, litellm_core::Error>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(NativeCallStep::Complete) })
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
                PendingLogging {
                    pending: Some(PendingSuccess {
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
                },
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
                PendingLogging {
                    pending: Some(PendingSuccess {
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
                },
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
}
