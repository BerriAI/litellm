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
}

pub type RuntimeCallEvent<'a, R> =
    CallEvent<&'a <R as HookRuntime>::Response, &'a <R as HookRuntime>::Error, &'a RawResponse>;

pub trait CallHooks<R: HookRuntime>: Sized {
    fn prepare_arguments(
        &mut self,
        runtime: R::Context<'_>,
        arguments: R::Arguments,
        started_at: f64,
    ) -> Result<R::Step<Self, R::Arguments>, R::Error>;

    fn arguments_prepared(
        &mut self,
        _runtime: R::Context<'_>,
        _arguments: &R::Arguments,
    ) -> Result<(), R::Error> {
        Ok(())
    }

    fn before_provider_request(
        &mut self,
        runtime: R::Context<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> Result<R::Step<Self, Box<WireRequest>>, R::Error>;

    fn transform_response(
        &mut self,
        runtime: R::Context<'_>,
        response: R::Response,
        timing: Timing,
    ) -> Result<R::Step<Self, R::Response>, R::Error>;

    fn on_event(
        &mut self,
        runtime: R::Context<'_>,
        event: RuntimeCallEvent<'_, R>,
    ) -> Result<R::Step<Self, ()>, R::Error>;

    fn on_stream_open(&mut self, runtime: R::Context<'_>) -> Result<(), R::Error>;

    fn on_stream_chunk(
        &mut self,
        runtime: R::Context<'_>,
        chunk: &R::Chunk,
    ) -> Result<(), R::Error>;
}
