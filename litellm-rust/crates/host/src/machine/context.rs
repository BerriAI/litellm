use crate::observation::ObservationSender;
use std::ops::ControlFlow;

use litellm_coroutine::Co;

use super::coroutine::MachineFault;
use crate::protocol::{HostRequest, InterceptRequest, Protocol, Reply, StreamDelivery};

pub struct CallContext<P: Protocol> {
    pub services: HostServices<P>,
    pub interceptors: ChannelInterceptors<P>,
    pub stream: StreamSender<P>,
    pub observers: Option<ObservationSender>,
}

impl<P: Protocol> CallContext<P> {
    pub(super) fn new(co: Co<HostRequest<P>>, observers: Option<ObservationSender>) -> Self {
        let channel = Channel(co);
        Self {
            services: HostServices(channel.clone()),
            interceptors: ChannelInterceptors(channel.clone()),
            stream: StreamSender(channel),
            observers,
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

pub struct ChannelInterceptors<P: Protocol>(Channel<P>);

impl<P: Protocol> Clone for ChannelInterceptors<P> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<P: Protocol> crate::interceptors::Interceptors<P::Error> for ChannelInterceptors<P>
where
    P::Error: From<MachineFault>,
{
    async fn handle(&self, request: InterceptRequest) -> Result<(), P::Error> {
        request.relay(self).await
    }

    async fn intercept<A: Send>(
        &self,
        ask: impl FnOnce(Reply<A>) -> InterceptRequest + Send,
    ) -> Result<A, P::Error> {
        self.0
            .request_reply(|reply| HostRequest::Intercept(ask(reply)))
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
