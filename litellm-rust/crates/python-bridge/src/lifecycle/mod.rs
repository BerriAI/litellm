use std::sync::Arc;
use std::task::Poll;

use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use futures_util::future::{AbortHandle, Abortable};
use tokio::sync::Mutex;

use litellm_core::call_lifecycle::host::{
    HostCall as NativeCall, HostCallStep as NativeCallStep, HostFailure, HostPhase, HostStep,
};
use litellm_core::call_lifecycle::{
    CallbackFamily, Delivery, ReleaseGate, SuccessFacts, plan_failure, plan_success,
};

mod arguments;
mod bindings;
mod compat;
mod dispatch;
mod handle;
mod preparation;
mod setup;

use crate::execution::{poll_async_value, run_async_value, run_sync_value};
pub(crate) use arguments::{BoundArguments, Signature};
pub(crate) use bindings::PythonLogger;
use handle::{Execution, ExecutionBody, ExecutionStep};

pub(crate) trait PythonRoute: Send + Sync {
    type Call: NativeCall + 'static;

    fn state(&self) -> &PythonCallState;
    fn state_mut(&mut self) -> &mut PythonCallState;
    fn phase(operation: &<Self::Call as NativeCall>::Operation) -> Option<HostPhase>;
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
type HostResult<C> = Result<<C as NativeCall>::Result, HostFailure<<C as NativeCall>::Error>>;
type HostResumeStep<R> = HostStep<NativeStep<<R as PythonRoute>::Call>, Py<PyAny>>;

struct NativeCallState<C: NativeCall> {
    call: C,
    result: Option<NativeResult<C>>,
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

pub(crate) fn debug_setup(
    py: Python<'_>,
    call_type: &'static str,
    args: &Py<PyTuple>,
    kwargs: &Py<PyDict>,
    start: &Py<PyAny>,
    asynchronous: bool,
) -> PyResult<(Py<PyAny>, Py<PyDict>)> {
    let result = setup::setup(py, call_type, args, kwargs, start, asynchronous)?;
    Ok((result.logger.object(py).clone().unbind(), result.kwargs))
}

pub(crate) fn missing_state() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("missing native call state")
}

impl<R: PythonRoute> PythonLifecycle<R> {
    fn resume_core(
        &mut self,
        py: Python<'_>,
        result: Option<HostResult<R::Call>>,
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
        let state = self.route.state_mut();
        state.retain_first_error(
            py,
            error,
            cancelled && phase != Some(HostPhase::DeploymentFailure),
        );
        let _ = state.finish(py);
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
                        .response
                        .take()
                        .map(ExecutionStep::Return)
                        .ok_or_else(missing_state);
                }
                HostStep::Ready(NativeCallStep::Host(operation)) => operation,
            };
            let phase = R::phase(&operation);
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
    pub supplied: bool,
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
                let copied = self.kwargs.bind(py).copy()?.into_any().unbind();
                return Ok(HostStep::Suspend(self.deployment(
                    py,
                    CallbackFamily::DeploymentPreCall,
                    copied,
                )?));
            }
            HostPhase::Prepare => self.prepare(py)?,
            HostPhase::DeploymentPostCall => {
                let response = self
                    .response
                    .as_ref()
                    .map(|value| value.clone_ref(py))
                    .unwrap_or_else(|| py.None());
                return Ok(HostStep::Suspend(self.deployment(
                    py,
                    CallbackFamily::DeploymentPostCall,
                    response,
                )?));
            }
            HostPhase::Finalize => self.finalize(py)?,
            HostPhase::Success => self.dispatch_success(py)?,
            HostPhase::DeploymentFailure => {
                if self.error.is_some() {
                    return Ok(HostStep::Suspend(self.deployment(
                        py,
                        CallbackFamily::DeploymentFailure,
                        py.None(),
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

    fn deployment(
        &self,
        py: Python<'_>,
        family: CallbackFamily,
        current: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let body = dispatch::DeploymentBody::start(
            py,
            self.logger()?,
            family,
            self.call_type,
            &self.kwargs,
            current,
            self.error.as_ref(),
        )?;
        dispatch::deployment_coroutine(py, body)
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
            supplied: false,
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
        let result = setup::setup(
            py,
            self.call_type,
            &self.args,
            &self.kwargs,
            &self.start,
            self.asynchronous,
        )?;
        self.logger = Some(result.logger);
        self.kwargs = result.kwargs;
        self.supplied = result.supplied;
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

    pub fn dispatch_request(&self, py: Python<'_>, family: CallbackFamily) -> PyResult<()> {
        dispatch::dispatch_request(
            py,
            dispatch::RequestJob {
                logger: self.logger()?,
                family,
            },
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

    fn job(&self, py: Python<'_>, family: CallbackFamily) -> PyResult<dispatch::Job> {
        let logger = self.logger()?;
        let (targets, ids) = dispatch::family_targets(py, logger, family)?;
        Ok(dispatch::Job {
            logger: logger.clone_ref(py),
            targets,
            ids,
            family,
            response: self.response.as_ref().map(|value| value.clone_ref(py)),
            error: self.error.as_ref().map(|value| value.clone_ref(py)),
            start: self.start.clone_ref(py),
            end: self
                .end
                .as_ref()
                .map(|value| value.clone_ref(py))
                .unwrap_or_else(|| py.None()),
            stream: false,
        })
    }

    fn try_dispatch_success(&self, py: Python<'_>) -> PyResult<()> {
        let logger = self.logger()?;
        if self.supplied {
            return compat::dispatch_success(py, self, logger);
        }
        let (sync_targets, sync_ids) =
            dispatch::family_targets(py, logger, CallbackFamily::SyncSuccess)?;
        let facts = SuccessFacts {
            asynchronous: self.asynchronous,
            internal: self.internal,
            fallbacks: !self
                .kwargs
                .bind(py)
                .get_item("fallbacks")?
                .is_none_or(|value| value.is_none()),
            deferred: logger.defers_async_logging(py),
            sync_target_kinds: sync_targets.kinds(&sync_ids),
        };
        for selected in plan_success(&facts) {
            let runner = dispatch::Runner::start(py, self.job(py, selected.family)?)?;
            match (selected.delivery, selected.gate) {
                (Delivery::Worker, _) => {
                    let job = Py::new(py, dispatch::WorkerJob::new(runner))?;
                    dispatch::leaves(py)?
                        .getattr("submit_worker")?
                        .call1((job,))?;
                }
                (Delivery::Background, ReleaseGate::Immediate) => {
                    DeferredSuccess::release(py, runner)?;
                }
                (Delivery::Background, ReleaseGate::Deferred) => {
                    logger.defer_success(
                        py,
                        Py::new(
                            py,
                            DeferredSuccess {
                                runner: Some(runner),
                            },
                        )?,
                    )?;
                }
                (Delivery::Inline | Delivery::Await, _) => return Err(missing_state()),
            }
        }
        Ok(())
    }

    pub fn dispatch_failure(
        &self,
        py: Python<'_>,
        asynchronous: bool,
    ) -> PyResult<Option<Py<PyAny>>> {
        if self.logger.is_none() || self.error.is_none() {
            return Ok(None);
        }
        let phase = if asynchronous {
            HostPhase::AsyncFailure
        } else {
            HostPhase::Failure
        };
        let Some(family) = plan_failure(phase, self.asynchronous, self.internal) else {
            return Ok(None);
        };
        if self.supplied {
            return compat::dispatch_failure(py, self, family);
        }
        let mut runner = dispatch::Runner::start(py, self.job(py, family)?)?;
        match family.delivery() {
            Delivery::Inline => match runner.resume(py, None)? {
                dispatch::Step::Done => Ok(None),
                dispatch::Step::Await(_) => Err(missing_state()),
            },
            Delivery::Await => Ok(Some(dispatch::coroutine(py, runner)?)),
            Delivery::Worker | Delivery::Background => Err(missing_state()),
        }
    }

    pub fn cleanup(&mut self, py: Python<'_>) {
        if let Some(logger) = self.logger.take()
            && let Err(error) = dispatch::leaves(py).and_then(|leaves| {
                leaves
                    .getattr("restore_correlation_context")?
                    .call1((logger.object(py),))
                    .map(|_| ())
            })
        {
            error.write_unraisable(py, None);
        }
    }

    pub fn retain_error(&mut self, py: Python<'_>, error: PyErr) {
        self.error = Some(error.into_value(py));
    }

    pub fn retain_first_error(&mut self, py: Python<'_>, error: PyErr, replace: bool) {
        if self.error.is_none() || replace {
            self.retain_error(py, error);
        }
    }

    pub fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.end.is_none() {
            self.end = Some(now(py)?);
        }
        Ok(())
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

#[pyclass]
struct DeferredSuccess {
    runner: Option<dispatch::Runner>,
}

impl DeferredSuccess {
    fn release(py: Python<'_>, runner: dispatch::Runner) -> PyResult<()> {
        let coroutine = dispatch::coroutine(py, runner)?;
        let enqueue = dispatch::leaves(py)?
            .getattr("enqueue_background")?
            .call1((&coroutine,));
        if enqueue.is_err()
            && let Err(error) = coroutine.call_method0(py, "close")
        {
            error.write_unraisable(py, Some(coroutine.bind(py)));
        }
        enqueue.map(|_| ())
    }
}

#[pymethods]
impl DeferredSuccess {
    fn __call__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<()> {
        let runner = slf.borrow_mut().runner.take();
        let Some(runner) = runner else {
            return Ok(());
        };
        let logger = runner.logger(py).clone().unbind();
        match Self::release(py, runner) {
            Err(error) if error.is_instance_of::<PyException>(py) => {
                error.write_unraisable(py, Some(logger.bind(py)));
                Ok(())
            }
            result => result,
        }
    }

    fn __traverse__(&self, visit: pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        match &self.runner {
            Some(runner) => runner.traverse(&visit),
            None => Ok(()),
        }
    }

    fn close(slf: &Bound<'_, Self>) {
        let runner = slf.borrow_mut().runner.take();
        drop(runner);
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        Self::close(slf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use pyo3::types::PyDict;
    use std::sync::Mutex;

    use litellm_core::call_lifecycle::host::HostCallFuture;

    static PYTHON_GLOBALS: Mutex<()> = Mutex::new(());

    fn load_lifecycle_module(py: Python<'_>) -> Bound<'_, PyModule> {
        py.run(
            pyo3::ffi::c_str!(
                r#"
import sys
import types
for name in ("litellm", "litellm.rust_bridge"):
    sys.modules.setdefault(name, types.ModuleType(name))
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
        ) -> HostCallFuture<'_, Self::Operation, Self::Complete, Self::Error> {
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
        ) -> HostCallFuture<'_, Self::Operation, Self::Complete, Self::Error> {
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

        fn phase(_: &()) -> Option<HostPhase> {
            None
        }

        fn lifecycle_result() {}

        fn host_error(message: String) -> litellm_core::messages::Error {
            litellm_core::messages::Error::InvalidRequest(message)
        }

        fn map_error(error: litellm_core::messages::Error) -> PyErr {
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
            load_lifecycle_module(py);
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
            let module = load_lifecycle_module(py);
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
            supplied: false,
            call_type: "test",
        }
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
