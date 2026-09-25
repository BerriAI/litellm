use std::future::Future;

pub use litellm_coroutine::{Abandoned, Answer, Reply, reply};

use crate::event::{CallEvent, MachineEvent, RequestContext, WireRequest};
use crate::protocol::Protocol;

/// One suspension point of a native call, performed by the host and answered through the
/// [`Reply`] it carries.
pub enum HostOp<R: Protocol> {
    /// The first op of every call: the caller's request as the host projects it.
    Project(Reply<R::Projection>),
    Custom(R::Op),
    BeforeSend {
        wire: Box<WireRequest>,
        context: Box<RequestContext>,
        reply: Reply<WireRequest>,
    },
    Emit(MachineEvent, Reply<()>),
    /// The response streams: the host hands the caller a stream and answers once the
    /// caller asks for the first chunk or goes away.
    Open(R::StreamHead, Reply<Demand>),
    /// The next chunk of an open stream, answered once the caller asks for the one after.
    Deliver(R::Chunk, Reply<Demand>),
}

/// Whether the caller of a streamed call still reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Demand {
    More,
    Detached,
}

/// A host answer that is either available now or arrives once the host's own
/// suspension (a Python awaitable, for example) resolves.
pub enum HostStep<V, S> {
    Ready(V),
    Suspend(S),
}

/// An in-process host: answers custom operations and observes the call without leaving
/// the Rust runtime. Language hosts implement their own driver instead.
pub trait Host<R: Protocol>: Send + Sync {
    fn project(&self) -> impl Future<Output = Result<R::Projection, R::Error>> + Send;

    /// Answers `op` through its reply, or fails the call.
    fn custom_op(&self, op: R::Op) -> impl Future<Output = Result<(), R::Error>> + Send;

    fn before_send(
        &self,
        wire: WireRequest,
        _context: &RequestContext,
    ) -> impl Future<Output = Result<WireRequest, R::Error>> + Send {
        async move { Ok(wire) }
    }

    fn emit(&self, _event: &CallEvent) -> impl Future<Output = Result<(), R::Error>> + Send {
        async { Ok(()) }
    }

    fn open(&self, _head: R::StreamHead) -> impl Future<Output = Result<Demand, R::Error>> + Send {
        async { Ok(Demand::More) }
    }

    fn deliver(&self, _chunk: R::Chunk) -> impl Future<Output = Result<Demand, R::Error>> + Send {
        async { Ok(Demand::More) }
    }
}
