use std::sync::Arc;
use std::task::Poll;

use futures_util::future::{AbortHandle, Abortable};
use tokio::sync::Mutex;

use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;

use litellm_core::call_lifecycle::host::{
    HostCall as NativeCall, HostCallStep as NativeCallStep, HostFailure, HostPhase, HostStep,
};

use super::handle::{Execution, ExecutionBody, ExecutionStep};
use super::state::{PythonCallState, missing_state, now};
use crate::execution::{poll_async_value, run_async_value, run_sync_value};

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
    fn invoke_step(
        &mut self,
        py: Python<'_>,
        operation: <Self::Call as NativeCall>::Operation,
    ) -> PyResult<HostStep<<Self::Call as NativeCall>::Result, Py<PyAny>>> {
        self.invoke(py, operation).map(HostStep::Ready)
    }
    fn accept_route(
        &mut self,
        _py: Python<'_>,
        _value: Py<PyAny>,
    ) -> PyResult<<Self::Call as NativeCall>::Result> {
        Err(missing_state())
    }
    fn cleanup(&mut self);
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

type NativeStep<C> = NativeCallStep<<C as NativeCall>::Operation, <C as NativeCall>::Complete>;
type NativeResult<C> = Result<NativeStep<C>, litellm_core::Error>;
type HostResumeStep<R> = HostStep<NativeStep<<R as PythonRoute>::Call>, Py<PyAny>>;

struct NativeCallState<C: NativeCall> {
    call: C,
    result: Option<NativeResult<C>>,
}

enum PendingOperation {
    Native,
    Host(HostPhase),
    Route,
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
    let asynchronous = route.state().mode.is_async();
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

impl<R: PythonRoute> PythonLifecycle<R> {
    fn resume_core(
        &mut self,
        py: Python<'_>,
        result: Option<Result<<R::Call as NativeCall>::Result, HostFailure>>,
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
        if self.route.state().mode.is_async() {
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
            (Some(PendingOperation::Route), Some(result)) => {
                let result = result.and_then(|value| self.route.accept_route(py, value));
                let result = result.map_err(|error| self.host_failure(py, error, None));
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
                None => match self.route.invoke_step(py, operation) {
                    Ok(HostStep::Ready(result)) => Ok(result),
                    Ok(HostStep::Suspend(awaitable)) => {
                        self.pending = Some(PendingOperation::Route);
                        return Ok(ExecutionStep::Await(awaitable));
                    }
                    Err(error) => Err(error),
                },
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
