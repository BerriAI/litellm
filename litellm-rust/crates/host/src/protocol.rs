use std::ops::ControlFlow;

pub use litellm_coroutine::{Abandoned, Answer, Reply, reply};

use crate::event::{MachineEvent, RequestContext, WireRequest};

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
    Hook(HookRequest),
    Stream(StreamDelivery<P>),
}

pub enum HookRequest {
    BeforeProviderRequest {
        wire: Box<WireRequest>,
        context: Box<RequestContext>,
        reply: Reply<WireRequest>,
    },
    Event(MachineEvent, Reply<()>),
}

pub enum StreamDelivery<P: Protocol> {
    Open(P::StreamHead, Reply<ControlFlow<()>>),
    Chunk(P::Chunk, Reply<ControlFlow<()>>),
}
