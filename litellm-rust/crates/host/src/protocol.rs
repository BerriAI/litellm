use std::ops::ControlFlow;

pub use litellm_coroutine::{Abandoned, Answer, Reply, reply};

use crate::interceptors::{InterceptPoints, RawResponse, RequestContext, WireRequest};

pub trait Protocol: Send + Sync + 'static {
    type Request: Send + 'static;
    type Response: Send + 'static;
    type Error: Clone + Send + Sync + 'static;
    type HostCall: Send + 'static;
    type Chunk: Send + 'static;
    type StreamHead: Send + 'static;
}

pub enum HostRequest<P: Protocol> {
    HostCall(P::HostCall),
    Intercept(InterceptRequest),
    Stream(StreamDelivery<P>),
}

pub enum InterceptRequest {
    ResultReady {
        facts: crate::interceptors::ExecutionFacts,
        reply: Reply<()>,
    },
    BeforeProviderRequest {
        wire: Box<WireRequest>,
        context: Box<RequestContext>,
        reply: Reply<WireRequest>,
    },
    AfterProviderResponse {
        raw: RawResponse,
        reply: Reply<()>,
    },
}

impl InterceptRequest {
    /// Answers as if no hook ran: a transform returns its input, a notification returns.
    pub fn pass_through(self) {
        match self {
            Self::ResultReady { reply, .. } | Self::AfterProviderResponse { reply, .. } => {
                reply.send(());
            }
            Self::BeforeProviderRequest { wire, reply, .. } => reply.send(*wire),
        }
    }

    /// Raises this request again through `interceptors` and answers its reply with the result.
    pub async fn relay<E, I>(self, interceptors: &I) -> Result<(), E>
    where
        E: From<crate::machine::MachineFault>,
        I: InterceptPoints<E> + ?Sized,
    {
        match self {
            Self::ResultReady { facts, reply } => {
                reply.send(interceptors.result_ready(facts).await?)
            }
            Self::BeforeProviderRequest {
                wire,
                context,
                reply,
            } => reply.send(
                interceptors
                    .before_provider_request(*wire, *context)
                    .await?,
            ),
            Self::AfterProviderResponse { raw, reply } => {
                reply.send(interceptors.after_provider_response(raw).await?);
            }
        }
        Ok(())
    }
}

pub enum StreamDelivery<P: Protocol> {
    Open(P::StreamHead, Reply<ControlFlow<()>>),
    Chunk(P::Chunk, Reply<ControlFlow<()>>),
}
