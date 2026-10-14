use std::sync::Arc;
use std::task::Poll;

use futures_util::future::{AbortHandle, Abortable};
use litellm_host::call::HostedCompletion;
use litellm_host::machine::{HostFailure, Machine, MachineStep};
use litellm_host::protocol::Protocol;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::sync::Mutex;

use crate::missing_state;
use crate::runtime::{poll_async_value, run_async_value, run_sync_value};

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
        } else {
            run_sync_value(py, future)?;
            Ok(NativePoll::Ready(self.take_result()?))
        }
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
    pub(super) fn close(&mut self) {
        if let Some(abort) = self.abort.take() {
            abort.abort();
        }
        self.state = None;
    }
}

impl<M: Machine> Drop for NativeMachine<M> {
    fn drop(&mut self) {
        self.close();
    }
}
