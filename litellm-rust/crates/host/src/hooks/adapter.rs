use futures_util::future::BoxFuture;

use super::{CallHooks, HookRuntime, RuntimeCallEvent};
use crate::{
    interceptors::{RequestContext, WireRequest},
    lifecycle::Timing,
};

pub trait DynCallHooks<R: HookRuntime + 'static>: Sync {
    fn prepare_arguments<'a>(
        &'a self,
        runtime: R::Context<'a>,
        arguments: R::Arguments,
        started_at: f64,
    ) -> BoxFuture<'a, Result<R::Arguments, R::Error>>;
    fn arguments_prepared(&self, arguments: &R::Arguments) -> Result<(), R::Error>;
    fn before_provider_request<'a: 'b, 'b>(
        &'a self,
        runtime: R::Context<'a>,
        wire: Box<WireRequest>,
        context: &'b RequestContext,
    ) -> BoxFuture<'b, Result<Box<WireRequest>, R::Error>>;
    fn transform_response<'a>(
        &'a self,
        runtime: R::Context<'a>,
        response: R::Response,
        timing: Timing,
    ) -> BoxFuture<'a, Result<R::Response, R::Error>>;
    fn on_event<'a: 'b, 'b>(
        &'a self,
        runtime: R::Context<'a>,
        event: RuntimeCallEvent<'b, R>,
    ) -> BoxFuture<'b, Result<(), R::Error>>;
    fn on_stream_open(&self, head: &R::Response) -> Result<(), R::Error>;
    fn on_stream_chunk(&self, chunk: &R::Chunk) -> Result<(), R::Error>;
}

impl<R: HookRuntime + 'static, H: CallHooks<R>> DynCallHooks<R> for H {
    fn prepare_arguments<'a>(
        &'a self,
        runtime: R::Context<'a>,
        arguments: R::Arguments,
        started_at: f64,
    ) -> BoxFuture<'a, Result<R::Arguments, R::Error>> {
        Box::pin(CallHooks::prepare_arguments(
            self, runtime, arguments, started_at,
        ))
    }

    fn arguments_prepared(&self, arguments: &R::Arguments) -> Result<(), R::Error> {
        CallHooks::arguments_prepared(self, arguments)
    }

    fn before_provider_request<'a: 'b, 'b>(
        &'a self,
        runtime: R::Context<'a>,
        wire: Box<WireRequest>,
        context: &'b RequestContext,
    ) -> BoxFuture<'b, Result<Box<WireRequest>, R::Error>> {
        Box::pin(CallHooks::before_provider_request(
            self, runtime, wire, context,
        ))
    }

    fn transform_response<'a>(
        &'a self,
        runtime: R::Context<'a>,
        response: R::Response,
        timing: Timing,
    ) -> BoxFuture<'a, Result<R::Response, R::Error>> {
        Box::pin(CallHooks::transform_response(
            self, runtime, response, timing,
        ))
    }

    fn on_event<'a: 'b, 'b>(
        &'a self,
        runtime: R::Context<'a>,
        event: RuntimeCallEvent<'b, R>,
    ) -> BoxFuture<'b, Result<(), R::Error>> {
        Box::pin(CallHooks::on_event(self, runtime, event))
    }

    fn on_stream_open(&self, head: &R::Response) -> Result<(), R::Error> {
        CallHooks::on_stream_open(self, head)
    }

    fn on_stream_chunk(&self, chunk: &R::Chunk) -> Result<(), R::Error> {
        CallHooks::on_stream_chunk(self, chunk)
    }
}
