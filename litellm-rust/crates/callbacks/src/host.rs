use std::future::Future;

use crate::event::{AttemptInfo, CallEvent, RequestContext, WireRequest};
use crate::failure::FailureClass;
use crate::route::Route;

/// One suspension point of a native call, performed by the host.
pub enum HostOp<R: Route> {
    Route(R::Op),
    BeforeSend {
        wire: Box<WireRequest>,
        context: Box<RequestContext>,
    },
    Emit(CallEvent),
    /// One attempt of a logical call failed. The host records it (cooldown, health and
    /// retry tables, per-attempt callbacks) and answers before the loop decides what to
    /// do next, so the decision sees the recorded state.
    AttemptFailed {
        attempt: AttemptInfo,
        class: FailureClass,
        error: R::Error,
    },
}

pub enum HostResult<R: Route> {
    Route(R::OpResult),
    BeforeSend(Box<WireRequest>),
    Emitted,
    /// The answer to [`HostOp::AttemptFailed`].
    Recorded,
    /// The answer to [`crate::machine::MachineStep::Yield`]: the consumer took the chunk.
    Consumed,
}

impl<R: Route> HostOp<R> {
    /// The same op as a route that embeds this route's ops sees it.
    pub fn map_op<T: Route<Error = R::Error>>(
        self,
        embed: impl FnOnce(R::Op) -> T::Op,
    ) -> HostOp<T> {
        match self {
            Self::Route(op) => HostOp::Route(embed(op)),
            Self::BeforeSend { wire, context } => HostOp::BeforeSend { wire, context },
            Self::Emit(event) => HostOp::Emit(event),
            Self::AttemptFailed {
                attempt,
                class,
                error,
            } => HostOp::AttemptFailed {
                attempt,
                class,
                error,
            },
        }
    }
}

impl<R: Route> HostResult<R> {
    /// The same answer as the embedded route sees it; `None` when the host answered an op
    /// that belongs to the embedding route instead.
    pub fn map_result<T: Route>(
        self,
        extract: impl FnOnce(R::OpResult) -> Option<T::OpResult>,
    ) -> Option<HostResult<T>> {
        Some(match self {
            Self::Route(result) => HostResult::Route(extract(result)?),
            Self::BeforeSend(wire) => HostResult::BeforeSend(wire),
            Self::Emitted => HostResult::Emitted,
            Self::Recorded => HostResult::Recorded,
            Self::Consumed => HostResult::Consumed,
        })
    }
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

    fn attempt_failed(
        &self,
        _attempt: &AttemptInfo,
        _class: FailureClass,
        _error: &R::Error,
    ) -> impl Future<Output = Result<(), R::Error>> + Send {
        async { Ok(()) }
    }

    fn consume(&self, _chunk: R::Chunk) -> impl Future<Output = Result<(), R::Error>> + Send {
        async { Ok(()) }
    }
}
