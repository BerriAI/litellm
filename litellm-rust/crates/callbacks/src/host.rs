use std::future::Future;

use crate::event::{CallEvent, RequestContext, WireRequest};
use crate::route::Route;

/// One suspension point of a native call, performed by the host.
pub enum HostOp<R: Route> {
    Route(R::Op),
    BeforeSend {
        wire: Box<WireRequest>,
        context: Box<RequestContext>,
    },
    Emit(CallEvent),
}

pub enum HostResult<R: Route> {
    Route(R::OpResult),
    BeforeSend(Box<WireRequest>),
    Emitted,
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
}
