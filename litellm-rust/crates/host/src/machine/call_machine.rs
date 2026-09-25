//! The one machine every route runs on: the route's provider future as a
//! [`Coroutine`] that yields [`HostOp`]s, each answered through its own typed reply. No
//! task is spawned; dropping the machine drops the in-flight call.

use std::{future::Future, pin::Pin};

use litellm_coroutine::{Co, Coroutine, CoroutineState, ResumeError};

use super::{HostFailure, Interrupted, Machine, MachineStep, Step};
use crate::{
    event::{MachineEvent, RequestContext, WireRequest},
    host::{Demand, HostOp, Reply},
    protocol::Protocol,
};

/// The machine's own failures, distinct from anything the provider call reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MachineFault {
    /// The host dropped an op's reply unanswered, or went away while the call waited.
    Abandoned,
    /// The host resumed the call out of turn.
    Protocol(ResumeError),
}

pub type ExecuteFuture<R> =
    Pin<Box<dyn Future<Output = Result<<R as Protocol>::Response, <R as Protocol>::Error>> + Send>>;

/// The provider side of the machine: how the in-flight call reaches its host.
pub struct HostChannel<R: Protocol> {
    co: Co<HostOp<R>>,
}

impl<R: Protocol> Clone for HostChannel<R> {
    fn clone(&self) -> Self {
        Self {
            co: self.co.clone(),
        }
    }
}

impl<R: Protocol> HostChannel<R>
where
    R::Error: From<MachineFault>,
{
    async fn yield_<A: Send>(
        &self,
        ask: impl FnOnce(Reply<A>) -> HostOp<R> + Send,
    ) -> Result<A, R::Error> {
        self.co
            .yield_(ask)
            .await
            .map_err(|_| MachineFault::Abandoned.into())
    }

    pub async fn project(&self) -> Result<R::Projection, R::Error> {
        self.yield_(HostOp::Project).await
    }

    /// Asks the host to perform the custom operation `ask` builds around its reply, as in
    /// `host.custom_op(OcrOp::AcquireAzureAdToken)`.
    pub async fn custom_op<A: Send>(
        &self,
        ask: impl FnOnce(Reply<A>) -> R::Op + Send,
    ) -> Result<A, R::Error> {
        self.yield_(|reply| HostOp::Custom(ask(reply))).await
    }

    pub async fn before_send(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, R::Error> {
        self.yield_(|reply| HostOp::BeforeSend {
            wire: Box::new(wire),
            context: Box::new(context),
            reply,
        })
        .await
    }

    pub async fn emit(&self, event: MachineEvent) -> Result<(), R::Error> {
        self.yield_(|reply| HostOp::Emit(event, reply)).await
    }

    pub async fn open(&self, head: R::StreamHead) -> Result<Demand, R::Error> {
        self.yield_(|reply| HostOp::Open(head, reply)).await
    }

    pub async fn deliver(&self, chunk: R::Chunk) -> Result<Demand, R::Error> {
        self.yield_(|reply| HostOp::Deliver(chunk, reply)).await
    }
}

type CallCoroutine<R> =
    Coroutine<HostOp<R>, Result<<R as Protocol>::Response, <R as Protocol>::Error>>;

pub struct CallMachine<R: Protocol> {
    coroutine: CallCoroutine<R>,
}

impl<R: Protocol> CallMachine<R>
where
    R::Error: From<MachineFault>,
{
    pub fn new(execute: impl FnOnce(HostChannel<R>) -> ExecuteFuture<R> + Send + 'static) -> Self {
        Self {
            coroutine: Coroutine::new(|co| execute(HostChannel { co })),
        }
    }
}

impl<R: Protocol> Machine for CallMachine<R>
where
    R::Error: From<MachineFault>,
{
    type Protocol = R;
    type Complete = R::Response;

    fn resume(&mut self) -> Step<'_, Self> {
        Box::pin(async move {
            match self
                .coroutine
                .resume()
                .await
                .map_err(MachineFault::Protocol)?
            {
                CoroutineState::Yielded(op) => Ok(MachineStep::Host(op)),
                CoroutineState::Complete(outcome) => outcome.map(MachineStep::Complete),
            }
        })
    }

    fn interrupt(&mut self, failure: HostFailure<R::Error>) -> Interrupted<'_, Self> {
        self.coroutine.cancel();
        Box::pin(async move { Err(failure.into_error()) })
    }
}
