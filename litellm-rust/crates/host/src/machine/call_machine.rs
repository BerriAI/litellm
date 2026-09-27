//! The one machine every route runs on: the route's provider future as a
//! [`Coroutine`] that yields [`Suspension`]s, each answered through its own typed reply. No
//! task is spawned; dropping the machine drops the in-flight call.

use crate::protocol::HookRequest;
use crate::protocol::StreamDelivery;
use std::{future::Future, pin::Pin};

use litellm_coroutine::{Co, Coroutine, CoroutineState, ResumeError};

use super::{HostFailure, Interrupted, Machine, MachineStep, Step};
use crate::{
    event::{MachineEvent, RequestContext, WireRequest},
    protocol::{Demand, Protocol, Reply, Suspension},
};

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

pub struct CallContext<P: Protocol> {
    pub services: HostServices<P>,
    pub hooks: ChannelHooks<P>,
    pub stream: StreamSender<P>,
}

struct Channel<P: Protocol>(Co<Suspension<P>>);

impl<P: Protocol> Clone for Channel<P> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<P: Protocol> Channel<P>
where
    P::Error: From<MachineFault>,
{
    async fn request_reply<A: Send>(
        &self,
        request: impl FnOnce(Reply<A>) -> Suspension<P> + Send,
    ) -> Result<A, P::Error> {
        self.0
            .yield_(request)
            .await
            .map_err(|_| MachineFault::Abandoned.into())
    }
}

pub struct HostServices<P: Protocol>(Channel<P>);

impl<P: Protocol> Clone for HostServices<P> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<P: Protocol> HostServices<P>
where
    P::Error: From<MachineFault>,
{
    pub async fn call<A: Send>(
        &self,
        request: impl FnOnce(Reply<A>) -> P::HostCall + Send,
    ) -> Result<A, P::Error> {
        self.0
            .request_reply(|reply| Suspension::HostCall(request(reply)))
            .await
    }
}

pub struct ChannelHooks<P: Protocol>(Channel<P>);

impl<P: Protocol> Clone for ChannelHooks<P> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<P: Protocol> crate::hooks::RouteHooks<P::Error> for ChannelHooks<P>
where
    P::Error: From<MachineFault>,
{
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, P::Error> {
        self.0
            .request_reply(|reply| {
                Suspension::Hook(HookRequest::BeforeProviderRequest {
                    wire: Box::new(wire),
                    context: Box::new(context),
                    reply,
                })
            })
            .await
    }

    async fn on_event(&self, event: MachineEvent) -> Result<(), P::Error> {
        self.0
            .request_reply(|reply| Suspension::Hook(HookRequest::Event(event, reply)))
            .await
    }
}

pub struct StreamSender<P: Protocol>(Channel<P>);

impl<P: Protocol> StreamSender<P>
where
    P::Error: From<MachineFault>,
{
    pub async fn open_stream(&self, head: P::StreamHead) -> Result<Demand, P::Error> {
        self.0
            .request_reply(|reply| Suspension::Stream(StreamDelivery::Open(head, reply)))
            .await
    }

    pub async fn send_chunk(&self, chunk: P::Chunk) -> Result<Demand, P::Error> {
        self.0
            .request_reply(|reply| Suspension::Stream(StreamDelivery::Chunk(chunk, reply)))
            .await
    }
}

type CallCoroutine<R, C> = Coroutine<Suspension<R>, Result<C, <R as Protocol>::Error>>;

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
            coroutine: Coroutine::new(|co| {
                let channel = Channel(co);
                execute(CallContext {
                    services: HostServices(channel.clone()),
                    hooks: ChannelHooks(channel.clone()),
                    stream: StreamSender(channel),
                })
            }),
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
