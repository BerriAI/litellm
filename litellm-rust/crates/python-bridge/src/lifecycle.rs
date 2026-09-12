use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use futures_util::future::{AbortHandle, Abortable};
use litellm_core::call_lifecycle::host::{HostFailure, HostPhase, HostStep};
use litellm_core::ocr::{OcrCall, OcrCallStep, OcrHostOperation, OcrHostResult};
use litellm_python_interop::panic_to_pyerr;
use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use tokio::sync::Mutex;

use crate::errors::ocr_error_to_pyerr;
use crate::execution::{run_async_value, run_sync_value};

mod preparation;

pub(crate) trait PythonRoute: Send + Sync {
    fn state(&self) -> &PythonCallState;
    fn state_mut(&mut self) -> &mut PythonCallState;
    fn invoke(&mut self, py: Python<'_>, operation: OcrHostOperation) -> PyResult<OcrHostResult>;
    fn cleanup(&mut self);
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

enum ExecutionStep {
    Return(Py<PyAny>),
    Await(Py<PyAny>),
}

trait ExecutionBody: Send + Sync {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep>;
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

enum ExecutionState {
    Created(Box<dyn ExecutionBody>),
    Running,
    Suspended(Box<dyn ExecutionBody>),
    Closed,
}

#[pyclass]
struct Execution {
    state: ExecutionState,
}

impl Execution {
    fn new(body: impl ExecutionBody + 'static) -> Self {
        Self {
            state: ExecutionState::Created(Box::new(body)),
        }
    }

    fn advance(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<Py<PyAny>> {
        let mut body = {
            let mut execution = slf.borrow_mut();
            match (&execution.state, result.is_some()) {
                (ExecutionState::Created(_), false) | (ExecutionState::Suspended(_), true) => {}
                (ExecutionState::Running, _) => {
                    return Err(PyRuntimeError::new_err("execution is already running"));
                }
                (ExecutionState::Closed, _) => {
                    return Err(PyRuntimeError::new_err("execution is closed"));
                }
                _ => {
                    return Err(PyRuntimeError::new_err(
                        "execution requires start before resume and can only start once",
                    ));
                }
            }
            match std::mem::replace(&mut execution.state, ExecutionState::Running) {
                ExecutionState::Created(body) | ExecutionState::Suspended(body) => body,
                _ => unreachable!(),
            }
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let step = body.resume(result)?;
            let (tag, value, suspended) = match step {
                ExecutionStep::Await(value) => ("Await", value, true),
                ExecutionStep::Return(value) => ("Complete", value, false),
            };
            let step = py
                .import("litellm.rust_bridge.lifecycle")?
                .getattr(tag)?
                .call1((value,))?
                .unbind();
            Ok((step, suspended))
        }))
        .map_err(panic_to_pyerr)
        .and_then(|result| result);
        match outcome {
            Ok((step, true)) if matches!(slf.borrow().state, ExecutionState::Running) => {
                slf.borrow_mut().state = ExecutionState::Suspended(body);
                Ok(step)
            }
            outcome => {
                slf.borrow_mut().state = ExecutionState::Closed;
                drop(body);
                outcome.and_then(|(step, suspended)| {
                    if suspended {
                        Err(PyRuntimeError::new_err(
                            "execution was closed while running",
                        ))
                    } else {
                        Ok(step)
                    }
                })
            }
        }
    }
}

#[pymethods]
impl Execution {
    fn start(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::advance(slf, py, None)
    }

    fn resume_value(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        value: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Self::advance(slf, py, Some(Ok(value)))
    }

    fn resume_error(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        error: Bound<'_, PyBaseException>,
    ) -> PyResult<Py<PyAny>> {
        Self::advance(slf, py, Some(Err(PyErr::from_value(error.into_any()))))
    }

    fn close(slf: &Bound<'_, Self>) {
        let state = std::mem::replace(&mut slf.borrow_mut().state, ExecutionState::Closed);
        drop(state);
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        match &self.state {
            ExecutionState::Created(body) | ExecutionState::Suspended(body) => {
                body.traverse(&visit)
            }
            _ => Ok(()),
        }
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        Self::close(slf);
    }
}

struct NativeCall {
    call: OcrCall,
    result: Option<Result<OcrCallStep, litellm_core::Error>>,
}

enum PendingOperation {
    Native,
    Host(HostPhase),
}

struct PythonLifecycle<R: PythonRoute> {
    route: R,
    call: Option<Arc<Mutex<NativeCall>>>,
    pending: Option<PendingOperation>,
    native_abort: Option<AbortHandle>,
}

pub(crate) fn run_call<R: PythonRoute + 'static>(
    py: Python<'_>,
    call: OcrCall,
    route: R,
) -> PyResult<Py<PyAny>> {
    let asynchronous = route.state().asynchronous;
    let mut lifecycle = PythonLifecycle {
        route,
        call: Some(Arc::new(Mutex::new(NativeCall { call, result: None }))),
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
        result: Option<OcrHostResult>,
    ) -> PyResult<HostStep<OcrCallStep, Py<PyAny>>> {
        let call = Arc::clone(self.call.as_ref().ok_or_else(missing_state)?);
        let future = async move {
            let mut call = call.lock().await;
            let result = match result {
                Some(OcrHostResult::Lifecycle(Err(failure))) => call.call.interrupt(failure).await,
                result => call.call.resume(result).await,
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

    fn take_native_result(&self) -> PyResult<OcrCallStep> {
        self.call
            .as_ref()
            .ok_or_else(missing_state)?
            .try_lock()
            .map_err(|_| missing_state())?
            .result
            .take()
            .ok_or_else(missing_state)?
            .map_err(ocr_error_to_pyerr)
    }

    fn host_failure(
        &mut self,
        py: Python<'_>,
        error: PyErr,
        phase: Option<HostPhase>,
    ) -> OcrHostResult {
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
        OcrHostResult::Lifecycle(Err(failure))
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
                    let result = self.host_failure(py, error, None);
                    self.resume_core(py, Some(result))?
                }
            },
            (Some(PendingOperation::Host(phase)), Some(result)) => {
                let result =
                    result.and_then(|value| self.route.state_mut().accept(py, phase, value));
                let result = match result {
                    Ok(()) => OcrHostResult::Lifecycle(Ok(())),
                    Err(error) => self.host_failure(py, error, Some(phase)),
                };
                self.resume_core(py, Some(result))?
            }
            _ => return Err(missing_state()),
        };
        loop {
            let operation = match step {
                HostStep::Suspend(awaitable) => return Ok(ExecutionStep::Await(awaitable)),
                HostStep::Ready(OcrCallStep::Complete(_)) => {
                    return self
                        .route
                        .state_mut()
                        .response
                        .take()
                        .map(ExecutionStep::Return)
                        .ok_or_else(missing_state);
                }
                HostStep::Ready(OcrCallStep::Host(operation)) => operation,
            };
            let phase = match &operation {
                OcrHostOperation::Lifecycle(phase) => Some(*phase),
                OcrHostOperation::Failure { .. } => Some(HostPhase::Failure),
                OcrHostOperation::Success { .. } => Some(HostPhase::Success),
                _ => None,
            };
            let result = match operation {
                OcrHostOperation::Success { .. } => self
                    .route
                    .state_mut()
                    .invoke(py, HostPhase::Success)
                    .map(|_| OcrHostResult::Lifecycle(Ok(()))),
                OcrHostOperation::Failure { .. } => self
                    .route
                    .state_mut()
                    .invoke(py, HostPhase::Failure)
                    .map(|_| OcrHostResult::Lifecycle(Ok(()))),
                OcrHostOperation::Lifecycle(phase) => {
                    match self.route.state_mut().invoke(py, phase) {
                        Ok(HostStep::Suspend(awaitable)) => {
                            self.pending = Some(PendingOperation::Host(phase));
                            return Ok(ExecutionStep::Await(awaitable));
                        }
                        Ok(HostStep::Ready(value)) => self
                            .route
                            .state_mut()
                            .accept(py, phase, value)
                            .map(|()| OcrHostResult::Lifecycle(Ok(()))),
                        Err(error) => Err(error),
                    }
                }
                operation => self.route.invoke(py, operation),
            };
            let result = match result {
                Ok(result) => result,
                Err(error) => self.host_failure(py, error, phase),
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
    pub logger: Option<Py<PyAny>>,
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
                return Ok(HostStep::Suspend(
                    py.import("litellm.utils")?
                        .getattr("async_pre_call_deployment_hook")?
                        .call1((&self.kwargs, self.call_type))?
                        .unbind(),
                ));
            }
            HostPhase::Prepare => self.prepare(py)?,
            HostPhase::DeploymentPostCall => {
                return Ok(HostStep::Suspend(
                    py.import("litellm.utils")?
                        .getattr("async_post_call_success_deployment_hook")?
                        .call1((&self.kwargs, &self.response, self.call_type))?
                        .unbind(),
                ));
            }
            HostPhase::Finalize => self.finalize(py)?,
            HostPhase::Success => self.dispatch_success(py)?,
            HostPhase::DeploymentFailure => {
                if let Some(error) = &self.error {
                    return Ok(HostStep::Suspend(
                        py.import("litellm.utils")?
                            .getattr("async_post_call_failure_deployment_hook")?
                            .call1((&self.kwargs, error, self.call_type))?
                            .unbind(),
                    ));
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

    pub fn logger<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.logger
            .as_ref()
            .map(|value| value.bind(py).clone())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("call logging is not initialized")
            })
    }

    pub fn setup(&mut self, py: Python<'_>) -> PyResult<()> {
        self.start = now(py)?;
        self.internal = py
            .import("litellm._internal_context")?
            .getattr("is_internal_call")?
            .call_method0("get")?
            .extract()?;
        let result = py
            .import("litellm.rust_bridge.lifecycle")?
            .getattr("setup")?
            .call1((
                self.call_type,
                &self.args,
                &self.kwargs,
                &self.start,
                self.asynchronous,
            ))?;
        self.logger = Some(result.getattr("logger")?.unbind());
        self.kwargs = result.getattr("kwargs")?.cast_into::<PyDict>()?.unbind();
        Ok(())
    }

    pub fn prepare(&mut self, py: Python<'_>) -> PyResult<()> {
        self.kwargs = preparation::prepare(py, self.kwargs.bind(py), &self.logger(py)?)?.unbind();
        Ok(())
    }

    pub fn finalize(&self, py: Python<'_>) -> PyResult<()> {
        py.import("litellm.rust_bridge.lifecycle")?
            .getattr("finalize")?
            .call1((
                &self.response,
                self.logger(py)?,
                &self.kwargs,
                &self.start,
                &self.end,
            ))?;
        Ok(())
    }

    pub fn dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        match self.try_dispatch_success(py) {
            Err(error) if error.is_instance_of::<PyException>(py) => {
                error.write_unraisable(py, self.logger.as_ref().map(|logger| logger.bind(py)));
                Ok(())
            }
            result => result,
        }
    }

    fn try_dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        let logger = self.logger(py)?;
        let pending = PendingSuccess {
            logger: logger.clone().unbind(),
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
                if logger
                    .getattr("_defer_async_logging")
                    .is_ok_and(|value| value.is_truthy().unwrap_or(false))
                {
                    logger.setattr(
                        "_native_pending_logging",
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
            logger
                .call_method1(
                    "handle_sync_success_callbacks_for_async_calls",
                    (&self.response, &self.start, &self.end),
                )
                .map(|_| ())
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
        let trace = py
            .import("traceback")?
            .getattr("format_exception")?
            .call1((error,))?;
        let trace = pyo3::types::PyString::new(py, "").call_method1("join", (trace,))?;
        let value = self.logger(py)?.call_method1(
            if asynchronous {
                "async_failure_handler"
            } else {
                "failure_handler"
            },
            (error, trace, &self.start, &self.end),
        )?;
        Ok(asynchronous.then(|| value.unbind()))
    }

    pub fn cleanup(&mut self, py: Python<'_>) {
        if let Some(logger) = self.logger.take()
            && let Err(error) = py.import("litellm.utils").and_then(|utils| {
                utils
                    .getattr("_restore_correlation_context_if_supported")?
                    .call1((logger,))
            })
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
        visit.call(&self.logger)?;
        visit.call(&self.start)?;
        visit.call(&self.end)?;
        visit.call(&self.response)?;
        visit.call(&self.error)
    }
}

struct PendingSuccess {
    logger: Py<PyAny>,
    response: Option<Py<PyAny>>,
    start: Py<PyAny>,
    end: Option<Py<PyAny>>,
}

impl PendingSuccess {
    fn context(py: Python<'_>) -> PyResult<Py<PyAny>> {
        py.import("contextvars")?
            .call_method0("copy_context")
            .map(Bound::unbind)
    }

    fn sync(&self, py: Python<'_>) -> PyResult<()> {
        let context = Self::context(py)?;
        py.import("litellm.litellm_core_utils.litellm_logging")?
            .getattr("executor")?
            .call_method1(
                "submit",
                (
                    context.getattr(py, "run")?,
                    self.logger.getattr(py, "success_handler")?,
                    &self.response,
                    &self.start,
                    &self.end,
                ),
            )?;
        Ok(())
    }

    fn asynchronous(&self, py: Python<'_>) -> PyResult<()> {
        let context = Self::context(py)?;
        let worker = py
            .import("litellm.litellm_core_utils.logging_worker")?
            .getattr("GLOBAL_LOGGING_WORKER")?
            .getattr("ensure_initialized_and_enqueue")?;
        let coroutine = self.logger.call_method1(
            py,
            "async_success_handler",
            (&self.response, &self.start, &self.end),
        )?;
        let enqueue = context.call_method1(py, "run", (worker, &coroutine));
        if enqueue.is_err()
            && let Err(error) = coroutine.call_method0(py, "close")
        {
            error.write_unraisable(py, Some(coroutine.bind(py)));
        }
        enqueue.map(|_| ())
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
                    error.write_unraisable(py, Some(pending.logger.bind(py)));
                }
                result => return result,
            }
        }
        Ok(())
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        if let Some(pending) = &self.pending {
            visit.call(&pending.logger)?;
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

    #[test]
    fn python_driver_preserves_inline_await_and_native_ownership() {
        let _guard = PYTHON_GLOBALS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        Python::initialize();
        Python::attach(|py| {
            py.import("asyncio").unwrap();
            let source = std::ffi::CString::new(include_str!(
                "../../../../litellm/rust_bridge/lifecycle.py"
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
            let probe = std::ffi::CString::new(include_str!("../tests/lifecycle.py")).unwrap();
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
            logger: Some(logger),
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
                        logger: locals.get_item("logger").unwrap().unwrap().unbind(),
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
