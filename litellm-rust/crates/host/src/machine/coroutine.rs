use std::{future::Future, pin::Pin};

use litellm_coroutine::{Coroutine, CoroutineState, ResumeError};

use super::{
    context::CallContext,
    contract::{HostFailure, Interrupted, Machine, MachineStep, Step},
};
use crate::{
    failure::{Failure, Stage},
    observation::ObservationSender,
    protocol::{HostRequest, Protocol},
};

/// The machine's own failures, distinct from anything the provider call reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MachineFault {
    /// The host dropped an op's reply unanswered, or went away while the call waited.
    Abandoned,
    /// The host resumed the call out of turn.
    Protocol(ResumeError),
}

impl std::fmt::Display for MachineFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Abandoned => f.write_str("driver was abandoned"),
            Self::Protocol(error) => error.fmt(f),
        }
    }
}

pub type ExecuteFuture<R, C = <R as Protocol>::Response> =
    Pin<Box<dyn Future<Output = Result<C, Failure<<R as Protocol>::Error>>> + Send>>;

type CallCoroutine<R, C> = Coroutine<HostRequest<R>, Result<C, Failure<<R as Protocol>::Error>>>;

pub struct CallMachine<R: Protocol, C = <R as Protocol>::Response> {
    coroutine: CallCoroutine<R, C>,
    /// The stage of the op the host is answering, where a host or machine fault lands.
    pending: Stage,
}

impl<R: Protocol, C: Send + 'static> CallMachine<R, C>
where
    R::Error: From<MachineFault>,
{
    pub fn new(
        observers: Option<ObservationSender>,
        execute: impl FnOnce(CallContext<R>) -> ExecuteFuture<R, C> + Send + 'static,
    ) -> Self {
        Self {
            coroutine: Coroutine::new(move |co| execute(CallContext::new(co, observers))),
            pending: Stage::Prepare,
        }
    }
}

impl<R: Protocol, C: Send + 'static> Machine for CallMachine<R, C>
where
    R::Error: From<MachineFault>,
{
    type Protocol = R;
    type Complete = C;

    fn resume(&mut self) -> Step<'_, Self> {
        Box::pin(async move {
            match self.coroutine.resume().await.map_err(|error| {
                Failure::at(self.pending, MachineFault::Protocol(error).into())
            })? {
                CoroutineState::Yielded(op) => {
                    self.pending = op.stage();
                    Ok(MachineStep::Suspended(op))
                }
                CoroutineState::Complete(outcome) => outcome.map(MachineStep::Complete),
            }
        })
    }

    fn interrupt(&mut self, failure: HostFailure<R::Error>) -> Interrupted<'_, Self> {
        self.coroutine.cancel();
        let failure = Failure::at(self.pending, failure.into_error());
        Box::pin(async move { Err(failure) })
    }
}
