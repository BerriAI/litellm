use std::ops::ControlFlow;

use litellm_coroutine::Co;

use super::coroutine::MachineFault;
use crate::{
    event::{MachineEvent, RequestContext, WireRequest},
    protocol::{HookRequest, HostRequest, Protocol, Reply, StreamDelivery},
};

pub struct CallContext<P: Protocol> {
    pub services: HostServices<P>,
    pub hooks: ChannelHooks<P>,
    pub stream: StreamSender<P>,
}

impl<P: Protocol> CallContext<P> {
    pub(super) fn new(co: Co<HostRequest<P>>) -> Self {
        let channel = Channel(co);
        Self {
            services: HostServices(channel.clone()),
            hooks: ChannelHooks(channel.clone()),
            stream: StreamSender(channel),
        }
    }
}

struct Channel<P: Protocol>(Co<HostRequest<P>>);

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
        request: impl FnOnce(Reply<A>) -> HostRequest<P> + Send,
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
            .request_reply(|reply| HostRequest::HostCall(request(reply)))
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
                HostRequest::Hook(HookRequest::BeforeProviderRequest {
                    wire: Box::new(wire),
                    context: Box::new(context),
                    reply,
                })
            })
            .await
    }

    async fn on_event(&self, event: MachineEvent) -> Result<(), P::Error> {
        self.0
            .request_reply(|reply| HostRequest::Hook(HookRequest::Event(event, reply)))
            .await
    }
}

pub struct StreamSender<P: Protocol>(Channel<P>);

impl<P: Protocol> StreamSender<P>
where
    P::Error: From<MachineFault>,
{
    pub async fn open_stream(&self, head: P::StreamHead) -> Result<ControlFlow<()>, P::Error> {
        self.0
            .request_reply(|reply| HostRequest::Stream(StreamDelivery::Open(head, reply)))
            .await
    }

    pub async fn send_chunk(&self, chunk: P::Chunk) -> Result<ControlFlow<()>, P::Error> {
        self.0
            .request_reply(|reply| HostRequest::Stream(StreamDelivery::Chunk(chunk, reply)))
            .await
    }
}
