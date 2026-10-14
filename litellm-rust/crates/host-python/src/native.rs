use std::sync::Arc;
#[cfg(not(unix))]
use std::task::Poll;

use futures_util::future::{AbortHandle, Abortable};
use litellm_host::call::HostedCompletion;
use litellm_host::machine::{HostFailure, Machine, MachineStep};
use litellm_host::protocol::Protocol;
use pyo3::exceptions::PyRuntimeError;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use tokio::sync::Mutex;

use crate::missing_state;
#[cfg(not(unix))]
use crate::runtime::poll_async_value;
use crate::runtime::{run_async_value, run_sync_value};
#[cfg(unix)]
use crate::wake::{
    NativeStep, StepCore, StepFuture, StepState, dispatcher_for, new_step, park, poll_step, reuse,
    running_loop, unpark,
};

type NativeResult<M> = Result<
    MachineStep<
        <M as Machine>::Protocol,
        HostedCompletion<<<M as Machine>::Protocol as Protocol>::Response>,
    >,
    <<M as Machine>::Protocol as Protocol>::Error,
>;

type MachineResult<M> = Result<
    MachineStep<<M as Machine>::Protocol, <M as Machine>::Complete>,
    <<M as Machine>::Protocol as Protocol>::Error,
>;

struct MachineState<M: Machine> {
    machine: M,
    result: Option<MachineResult<M>>,
}

pub(super) enum NativePoll<T> {
    Ready(T),
    Suspend(Py<PyAny>),
}

pub(super) struct NativeMachine<M: Machine> {
    state: Option<Arc<Mutex<MachineState<M>>>>,
    abort: Option<AbortHandle>,
    asynchronous: bool,
    #[cfg(unix)]
    step: Option<(Arc<StepCore>, Option<Py<NativeStep>>)>,
}

impl<M: Machine + 'static> NativeMachine<M>
where
    M::Complete: Into<HostedCompletion<<M::Protocol as Protocol>::Response>>,
{
    pub(super) fn new(asynchronous: bool) -> Self {
        Self {
            state: None,
            abort: None,
            asynchronous,
            #[cfg(unix)]
            step: None,
        }
    }

    pub(super) fn start(&mut self, machine: M) {
        self.state = Some(Arc::new(Mutex::new(MachineState {
            machine,
            result: None,
        })));
    }

    pub(super) fn resume(
        &mut self,
        py: Python<'_>,
        interruption: Option<HostFailure<<M::Protocol as Protocol>::Error>>,
    ) -> PyResult<NativePoll<NativeResult<M>>> {
        let state = Arc::clone(self.state.as_ref().ok_or_else(missing_state)?);
        let future = async move {
            let mut state = state.lock().await;
            let result = match interruption {
                Some(failure) => state
                    .machine
                    .interrupt(failure)
                    .await
                    .map(MachineStep::Complete),
                None => state.machine.resume().await,
            };
            state.result = Some(result);
            Ok(())
        };
        if self.asynchronous {
            #[cfg(unix)]
            {
                self.resume_unix(py, Box::pin(future))
            }
            #[cfg(not(unix))]
            {
                let mut future = Box::pin(future);
                if let Poll::Ready(()) = poll_async_value(py, future.as_mut())? {
                    return Ok(NativePoll::Ready(self.take_result()?));
                }
                let (abort, registration) = AbortHandle::new_pair();
                self.abort = Some(abort);
                Ok(NativePoll::Suspend(
                    run_async_value(py, async move {
                        Abortable::new(future, registration)
                            .await
                            .map_err(|_| PyRuntimeError::new_err("native execution closed"))?
                    })?
                    .unbind(),
                ))
            }
        } else {
            run_sync_value(py, future)?;
            Ok(NativePoll::Ready(self.take_result()?))
        }
    }

    #[cfg(unix)]
    fn resume_unix(
        &mut self,
        py: Python<'_>,
        future: StepFuture,
    ) -> PyResult<NativePoll<NativeResult<M>>> {
        let (mut core, existing_step) = self.step.take().unwrap_or_else(|| (StepCore::new(), None));
        core.arm(future);
        poll_step(py, &core)?;
        match core.state() {
            StepState::Ready => {
                self.step = Some((core, existing_step));
                return Ok(NativePoll::Ready(self.take_result()?));
            }
            StepState::Failed => {
                let error = core.error(py).unwrap_or_else(missing_state);
                self.step = Some((core, existing_step));
                return Err(error);
            }
            StepState::Cancelled => {
                let error = core.error(py).unwrap_or_else(missing_state);
                self.step = Some((core, existing_step));
                return Err(error);
            }
            StepState::Pending => {}
        }

        let event_loop = running_loop(py)?;
        if let Some(step) = &existing_step {
            if reuse(py, step, &event_loop)? {
                park(py, step, core.id());
                self.step = Some((core, Some(step.clone_ref(py))));
                return Ok(NativePoll::Suspend(step.clone_ref(py).into_any()));
            }
            unpark(py, step, core.id());
        }

        let Some(dispatcher) = dispatcher_for(py, &event_loop)? else {
            let future = core.take_future().ok_or_else(missing_state)?;
            self.step = None;
            return self.resume_fallback(py, future);
        };

        if core.bound() {
            let future = core.take_future().ok_or_else(missing_state)?;
            core = StepCore::new();
            core.arm(future);
            poll_step(py, &core)?;
            match core.state() {
                StepState::Ready => {
                    self.step = Some((core, None));
                    return Ok(NativePoll::Ready(self.take_result()?));
                }
                StepState::Failed => {
                    let error = core.error(py).unwrap_or_else(missing_state);
                    self.step = Some((core, None));
                    return Err(error);
                }
                StepState::Cancelled => {
                    let error = core.error(py).unwrap_or_else(missing_state);
                    self.step = Some((core, None));
                    return Err(error);
                }
                StepState::Pending => {}
            }
        }

        let step = new_step(py, &core, &dispatcher, &event_loop)?;
        park(py, &step, core.id());
        self.step = Some((core, Some(step.clone_ref(py))));
        Ok(NativePoll::Suspend(step.into_any()))
    }

    #[cfg(unix)]
    fn resume_fallback(
        &mut self,
        py: Python<'_>,
        future: StepFuture,
    ) -> PyResult<NativePoll<NativeResult<M>>> {
        let (abort, registration) = AbortHandle::new_pair();
        self.abort = Some(abort);
        Ok(NativePoll::Suspend(
            run_async_value(py, async move {
                Abortable::new(future, registration)
                    .await
                    .map_err(|_| PyRuntimeError::new_err("native execution closed"))?
            })?
            .unbind(),
        ))
    }

    pub(super) fn take_result(&self) -> PyResult<NativeResult<M>> {
        self.state
            .as_ref()
            .ok_or_else(missing_state)?
            .try_lock()
            .map_err(|_| missing_state())?
            .result
            .take()
            .ok_or_else(missing_state)
            .map(|result| {
                result.map(|step| match step {
                    MachineStep::Suspended(op) => MachineStep::Suspended(op),
                    MachineStep::Complete(response) => MachineStep::Complete(response.into()),
                })
            })
    }
}

impl<M: Machine> NativeMachine<M> {
    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        #[cfg(unix)]
        if let Some((_, Some(step))) = &self.step {
            visit.call(step)?;
        }
        Ok(())
    }

    pub(super) fn close(&mut self) {
        if let Some(abort) = self.abort.take() {
            abort.abort();
        }
        #[cfg(unix)]
        if let Some((core, _step)) = self.step.take() {
            core.close();
        }
        self.state = None;
    }
}

impl<M: Machine> Drop for NativeMachine<M> {
    fn drop(&mut self) {
        self.close();
    }
}
