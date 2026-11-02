use crate::{
    interceptors::{RawResponse, RequestContext, WireRequest},
    lifecycle::{CallEvent, Timing},
};

pub trait HookRuntime {
    type Context<'a>;
    type Arguments;
    type Response;
    type Chunk;
    type Error;
    type Step<H, T>;

    fn ready<H, T>(value: T) -> Self::Step<H, T>;
}

pub type RuntimeCallEvent<'a, R> =
    CallEvent<&'a <R as HookRuntime>::Response, &'a <R as HookRuntime>::Error, &'a RawResponse>;

pub trait CallHooks<R: HookRuntime>: Sized {
    fn prepare_arguments(
        &mut self,
        _runtime: R::Context<'_>,
        arguments: R::Arguments,
        _started_at: f64,
    ) -> Result<R::Step<Self, R::Arguments>, R::Error> {
        Ok(R::ready(arguments))
    }

    fn arguments_prepared(
        &mut self,
        _runtime: R::Context<'_>,
        _arguments: &R::Arguments,
    ) -> Result<(), R::Error> {
        Ok(())
    }

    fn before_provider_request(
        &mut self,
        _runtime: R::Context<'_>,
        wire: Box<WireRequest>,
        _context: &RequestContext,
    ) -> Result<R::Step<Self, Box<WireRequest>>, R::Error> {
        Ok(R::ready(wire))
    }

    fn transform_response(
        &mut self,
        _runtime: R::Context<'_>,
        response: R::Response,
        _timing: Timing,
    ) -> Result<R::Step<Self, R::Response>, R::Error> {
        Ok(R::ready(response))
    }

    fn on_event(
        &mut self,
        _runtime: R::Context<'_>,
        _event: RuntimeCallEvent<'_, R>,
    ) -> Result<R::Step<Self, ()>, R::Error> {
        Ok(R::ready(()))
    }

    fn on_stream_open(
        &mut self,
        _runtime: R::Context<'_>,
        _head: &R::Response,
    ) -> Result<(), R::Error> {
        Ok(())
    }

    fn on_stream_chunk(
        &mut self,
        _runtime: R::Context<'_>,
        _chunk: &R::Chunk,
    ) -> Result<(), R::Error> {
        Ok(())
    }
}
