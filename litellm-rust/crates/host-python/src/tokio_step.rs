use std::future::Future;
use std::pin::Pin;

use litellm_host::call::HostedCompletion;
use litellm_host::machine::{HostFailure, Machine, MachineStep};
use litellm_host::protocol::Protocol;
use pyo3::prelude::*;

use crate::Error;
use crate::caller::PythonCaller;
use crate::runtime::{AsyncPoll, poll_async_value_or_await, run_sync_value};

type ErrorOf<M> = <<M as Machine>::Protocol as Protocol>::Error;
type ResponseOf<M> = <<M as Machine>::Protocol as Protocol>::Response;
type MachineResult<M> =
    Result<MachineStep<<M as Machine>::Protocol, <M as Machine>::Complete>, ErrorOf<M>>;
pub(super) type StepResult<M> =
    Result<MachineStep<<M as Machine>::Protocol, HostedCompletion<ResponseOf<M>>>, ErrorOf<M>>;
type InFlight<M> = Pin<Box<dyn Future<Output = (M, MachineResult<M>)> + Send>>;

/// Why a step produced no result: polling it failed on the host side, or the caller raised
/// while waiting for it, which leaves the step in flight.
pub(super) enum StepError {
    Host(PyErr),
    Caller(PyErr),
}

enum State<M: Machine> {
    Idle(M),
    InFlight(InFlight<M>),
}

/// Runs one [`Machine`] step at a time from the Python caller's task. A synchronous call
/// blocks on tokio with the GIL released. An asynchronous call polls the step in place and,
/// while it is pending, hands the caller an awaitable that resolves when tokio wakes it.
pub(super) struct TokioStep<M: Machine> {
    state: Option<State<M>>,
    asynchronous: bool,
}

impl<M: Machine + 'static> TokioStep<M>
where
    M::Complete: Into<HostedCompletion<ResponseOf<M>>>,
{
    pub(super) fn new(asynchronous: bool) -> Self {
        Self {
            state: None,
            asynchronous,
        }
    }

    pub(super) fn start(&mut self, machine: M) {
        self.state = Some(State::Idle(machine));
    }

    /// Resumes the machine, or interrupts it with `interruption` once any step still in
    /// flight has finished, and waits for the step to complete.
    pub(super) async fn step(
        &mut self,
        caller: &PythonCaller,
        interruption: Option<HostFailure<ErrorOf<M>>>,
    ) -> Result<StepResult<M>, StepError> {
        if interruption.is_some() && matches!(self.state, Some(State::InFlight(_))) {
            self.settle(caller).await.map(drop)?;
        }
        let Some(State::Idle(mut machine)) = self.state.take() else {
            return Err(StepError::Host(Error::MissingState.into()));
        };
        self.state = Some(State::InFlight(Box::pin(async move {
            let result = match interruption {
                Some(failure) => machine.interrupt(failure).await.map(MachineStep::Complete),
                None => machine.resume().await,
            };
            (machine, result)
        })));
        self.settle(caller).await
    }

    async fn settle(&mut self, caller: &PythonCaller) -> Result<StepResult<M>, StepError> {
        loop {
            let awaitable = match Python::attach(|py| self.poll(py)).map_err(StepError::Host)? {
                AsyncPoll::Ready(result) => return Ok(result),
                AsyncPoll::Awaiting(awaitable) => awaitable,
            };
            caller
                .await_python(awaitable)
                .await
                .map_err(StepError::Caller)?;
        }
    }

    fn poll(&mut self, py: Python<'_>) -> PyResult<AsyncPoll<StepResult<M>>> {
        let Some(State::InFlight(mut future)) = self.state.take() else {
            return Err(Error::MissingState.into());
        };
        if !self.asynchronous {
            let (machine, result) = run_sync_value(py, async move { Ok(future.await) })?;
            self.state = Some(State::Idle(machine));
            return Ok(AsyncPoll::Ready(hosted::<M>(result)));
        }
        match poll_async_value_or_await(py, future.as_mut())? {
            AsyncPoll::Ready((machine, result)) => {
                self.state = Some(State::Idle(machine));
                Ok(AsyncPoll::Ready(hosted::<M>(result)))
            }
            AsyncPoll::Awaiting(awaitable) => {
                self.state = Some(State::InFlight(future));
                Ok(AsyncPoll::Awaiting(awaitable))
            }
        }
    }
}

fn hosted<M: Machine>(result: MachineResult<M>) -> StepResult<M>
where
    M::Complete: Into<HostedCompletion<ResponseOf<M>>>,
{
    result.map(|step| match step {
        MachineStep::Suspended(op) => MachineStep::Suspended(op),
        MachineStep::Complete(complete) => MachineStep::Complete(complete.into()),
    })
}
