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

pub enum Suspension<P: Protocol> {
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
    Open(P::StreamHead, Reply<Demand>),
    Chunk(P::Chunk, Reply<Demand>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Demand {
    More,
    Detached,
}
