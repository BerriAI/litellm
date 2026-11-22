use std::{
    ops::ControlFlow,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub use litellm_coroutine::{Abandoned, Answer, Reply, reply};

use crate::interceptors::{RawResponse, RequestContext, RetriedRequest, WireRequest};

pub trait Protocol: Send + Sync + 'static {
    type Request: Send + 'static;
    type Response: Send + 'static;
    type Error: Clone + Send + Sync + 'static;
    type HostCall: Send + 'static;
    type Chunk: Send + 'static;
    type StreamHead: Send + 'static;

    fn detached_drain(_head: &Self::StreamHead) -> Option<DetachedDrain> {
        None
    }

    fn unobserved_tail(_head: &Self::StreamHead) -> Option<UnobservedTail> {
        None
    }
}

/// Where a stream's chunks stop reaching the stream hooks: once the route begins the tail, the
/// chunks it delivers go to the caller alone, as a frame Python's relay adds after billing does.
#[derive(Clone, Debug, Default)]
pub struct UnobservedTail(Arc<AtomicBool>);

impl UnobservedTail {
    pub fn begin(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn begun(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// A stream whose upstream is read to its end after the caller detaches, so the provider's
/// terminal usage still reaches the stream hooks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetachedDrain {
    pub timeout: std::time::Duration,
    pub limit: usize,
}

pub enum HostRequest<P: Protocol> {
    HostCall(P::HostCall),
    Intercept(InterceptRequest<P::Error>),
    Stream(StreamDelivery<P>),
}

pub enum InterceptRequest<E> {
    ResultReady {
        facts: crate::interceptors::ExecutionFacts,
        reply: Reply<Result<(), E>>,
    },
    BeforeProviderRequest {
        wire: Box<WireRequest>,
        context: Box<RequestContext>,
        reply: Reply<WireRequest>,
    },
    AfterProviderResponse {
        raw: RawResponse,
        reply: Reply<Result<(), E>>,
    },
    ProviderRequestRetried {
        retry: RetriedRequest,
        reply: Reply<Result<(), E>>,
    },
}

pub enum StreamDelivery<P: Protocol> {
    Open(P::StreamHead, Reply<ControlFlow<()>>),
    Chunk(P::Chunk, Reply<ControlFlow<()>>),
}
