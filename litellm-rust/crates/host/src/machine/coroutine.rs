use std::{future::Future, pin::Pin};

use litellm_coroutine::{Coroutine, CoroutineState, ResumeError};

use super::{
    context::CallContext,
    contract::{HostFailure, Interrupted, Machine, MachineStep, Step},
};
use crate::protocol::{HostRequest, Protocol};

/// The machine's own failures, distinct from anything the provider call reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MachineFault {
    /// The host dropped an op's reply unanswered, or went away while the call waited.
    Abandoned,
    /// The host resumed the call out of turn.
    Protocol(ResumeError),
}

pub type ExecuteFuture<R, C = <R as Protocol>::Response> =
    Pin<Box<dyn Future<Output = Result<C, <R as Protocol>::Error>> + Send>>;

type CallCoroutine<R, C> = Coroutine<HostRequest<R>, Result<C, <R as Protocol>::Error>>;

pub struct CallMachine<R: Protocol, C = <R as Protocol>::Response> {
    coroutine: CallCoroutine<R, C>,
}

impl<R: Protocol, C: Send + 'static> CallMachine<R, C>
where
    R::Error: From<MachineFault>,
{
    pub fn new(
        execute: impl FnOnce(CallContext<R>) -> ExecuteFuture<R, C> + Send + 'static,
    ) -> Self {
        Self {
            coroutine: Coroutine::new(move |co| execute(CallContext::new(co))),
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
            match self
                .coroutine
                .resume()
                .await
                .map_err(MachineFault::Protocol)?
            {
                CoroutineState::Yielded(op) => Ok(MachineStep::Suspended(op)),
                CoroutineState::Complete(outcome) => outcome.map(MachineStep::Complete),
            }
        })
    }

    fn interrupt(&mut self, failure: HostFailure<R::Error>) -> Interrupted<'_, Self> {
        self.coroutine.cancel();
        Box::pin(async move { Err(failure.into_error()) })
    }
}
