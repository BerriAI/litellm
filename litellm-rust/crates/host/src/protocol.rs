use std::ops::ControlFlow;

pub use litellm_coroutine::{Abandoned, Answer, Reply, reply};

use crate::{
    failure::Stage,
    interceptors::{RawResponse, RequestContext, WireRequest},
};

pub trait Protocol: Send + Sync + 'static {
    type Request: Send + 'static;
    type Response: Send + 'static;
    type Error: Clone + Send + Sync + 'static;
    type HostCall: Send + 'static;
    type Chunk: Send + 'static;
    type StreamHead: Send + 'static;

    /// Where the call stands while it waits on this host op.
    fn host_call_stage(call: &Self::HostCall) -> Stage;
}

pub enum HostRequest<P: Protocol> {
    HostCall(P::HostCall),
    Intercept(InterceptRequest),
    Stream(StreamDelivery<P>),
}

impl<P: Protocol> HostRequest<P> {
    /// Where the call stands while it waits on this op, so a host that fails to answer
    /// reports the failure at that stage.
    pub fn stage(&self) -> Stage {
        match self {
            Self::HostCall(call) => P::host_call_stage(call),
            Self::Intercept(InterceptRequest::BeforeProviderRequest { .. }) => Stage::Prepare,
            Self::Intercept(
                InterceptRequest::ResultReady { .. } | InterceptRequest::AfterProviderResponse { .. },
            ) => Stage::PostCall,
            Self::Stream(_) => Stage::Receive,
        }
    }
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

pub enum StreamDelivery<P: Protocol> {
    Open(P::StreamHead, Reply<ControlFlow<()>>),
    Chunk(P::Chunk, Reply<ControlFlow<()>>),
}
