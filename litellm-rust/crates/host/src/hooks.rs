use std::future::Future;

use crate::{
    interceptors::{RawResponse, RequestContext, WireRequest},
    lifecycle::{CallEvent, Timing},
};

/// What a host runtime supplies to its call hooks: the context every hook receives, and
/// the value types the hooks see at each stage.
pub trait HookRuntime {
    type Context: Send + Sync;
    type Arguments: Send + Sync;
    type Response: Send + Sync;
    type Chunk: Send + Sync;
    type Error: Send + Sync;
}

pub type RuntimeCallEvent<'a, R> =
    CallEvent<&'a <R as HookRuntime>::Response, &'a <R as HookRuntime>::Error, &'a RawResponse>;

/// How the host delivers a stream to the caller. `Relay` hands chunks on as they arrive.
/// `HoldBack` keeps them until the stream ends, runs the response transformation on the
/// assembled response and delivers what came out. `Synthesize` turns a complete response
/// into a stream because the caller asked for one and a hook made the provider call
/// complete instead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StreamPolicy {
    #[default]
    Relay,
    HoldBack,
    Synthesize,
}

impl StreamPolicy {
    /// The policy a chain of hooks settles on: holding back wins over synthesizing, and
    /// either wins over relaying.
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::HoldBack, _) | (_, Self::HoldBack) => Self::HoldBack,
            (Self::Synthesize, _) | (_, Self::Synthesize) => Self::Synthesize,
            (Self::Relay, Self::Relay) => Self::Relay,
        }
    }
}

/// The stages of one call a host may act on. Every stage is awaited on the host's own
/// context, so a hook suspends the call by awaiting rather than by returning a continuation.
pub trait CallHooks<R: HookRuntime>: Send + Sized {
    fn prepare_arguments(
        &mut self,
        _runtime: &R::Context,
        arguments: R::Arguments,
        _started_at: f64,
    ) -> impl Future<Output = Result<R::Arguments, R::Error>> + Send {
        async move { Ok(arguments) }
    }

    fn prepare_request(
        &mut self,
        _runtime: &R::Context,
        arguments: R::Arguments,
    ) -> impl Future<Output = Result<R::Arguments, R::Error>> + Send {
        async move { Ok(arguments) }
    }

    fn arguments_prepared(
        &mut self,
        _runtime: &R::Context,
        _arguments: &R::Arguments,
    ) -> impl Future<Output = Result<(), R::Error>> + Send {
        async { Ok(()) }
    }

    /// A response that answers the call without a provider request, from a hook that owns
    /// the request's shape. The driver then never starts the route's machine.
    fn short_circuit(
        &mut self,
        _runtime: &R::Context,
        _arguments: &R::Arguments,
    ) -> impl Future<Output = Result<Option<R::Response>, R::Error>> + Send {
        async { Ok(None) }
    }

    fn stream_policy(
        &mut self,
        _runtime: &R::Context,
        _arguments: &R::Arguments,
    ) -> impl Future<Output = Result<StreamPolicy, R::Error>> + Send {
        async { Ok(StreamPolicy::Relay) }
    }

    fn before_provider_request(
        &mut self,
        _runtime: &R::Context,
        wire: Box<WireRequest>,
        _context: &RequestContext,
    ) -> impl Future<Output = Result<Box<WireRequest>, R::Error>> + Send {
        async move { Ok(wire) }
    }

    fn transform_response(
        &mut self,
        _runtime: &R::Context,
        response: R::Response,
        _timing: Timing,
    ) -> impl Future<Output = Result<R::Response, R::Error>> + Send {
        async move { Ok(response) }
    }

    fn on_event(
        &mut self,
        _runtime: &R::Context,
        _event: RuntimeCallEvent<'_, R>,
    ) -> impl Future<Output = Result<(), R::Error>> + Send {
        async { Ok(()) }
    }

    fn on_stream_open(
        &mut self,
        _runtime: &R::Context,
        _head: &R::Response,
    ) -> impl Future<Output = Result<(), R::Error>> + Send {
        async { Ok(()) }
    }

    fn on_stream_chunk(
        &mut self,
        _runtime: &R::Context,
        _chunk: &R::Chunk,
    ) -> impl Future<Output = Result<(), R::Error>> + Send {
        async { Ok(()) }
    }
}
