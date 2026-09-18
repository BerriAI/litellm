use std::future::Future;

use crate::event::{CallEvent, MachineEvent, RequestContext, WireRequest};
use crate::route::Route;

/// One suspension point of a native call, performed by the host.
pub enum HostOp<R: Route> {
    Route(R::Op),
    BeforeSend {
        wire: Box<WireRequest>,
        context: Box<RequestContext>,
    },
    Emit(MachineEvent),
    /// The response streams: the host hands the caller a stream and answers once the
    /// caller asks for the first chunk or goes away.
    Open(R::StreamHead),
    /// The next chunk of an open stream, answered once the caller asks for the one after.
    Deliver(R::Chunk),
}

pub enum HostResult<R: Route> {
    Route(R::OpResult),
    BeforeSend(Box<WireRequest>),
    Emitted,
    Demand(Demand),
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

/// An in-process host: answers route operations and observes the call without leaving
/// the Rust runtime. Language hosts implement their own driver instead.
pub trait Host<R: Route>: Send + Sync {
    fn route(&self, op: R::Op) -> impl Future<Output = Result<R::OpResult, R::Error>> + Send;

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
