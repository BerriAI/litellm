use std::future::{Future, ready};

use crate::{
    interceptors::{RawResponse, RequestContext, WireRequest},
    lifecycle::{CallEvent, Timing},
};

pub trait HookRuntime {
    type Context<'a>: Send;
    type Arguments: Send;
    type Response: Send + Sync;
    type Chunk;
    type Error: Send + Sync;
    type Failure: Sync;
}

pub type RuntimeCallEvent<'a, R> =
    CallEvent<&'a <R as HookRuntime>::Response, &'a <R as HookRuntime>::Failure, &'a RawResponse>;

pub trait CallHooks<R: HookRuntime>: Sync {
    fn prepare_arguments<'a>(
        &'a self,
        _runtime: R::Context<'a>,
        arguments: R::Arguments,
        _started_at: f64,
    ) -> impl Future<Output = Result<R::Arguments, R::Error>> + Send {
        ready(Ok(arguments))
    }

    fn arguments_prepared(&self, _arguments: &R::Arguments) -> Result<(), R::Error> {
        Ok(())
    }

    fn before_provider_request<'a>(
        &'a self,
        _runtime: R::Context<'a>,
        wire: Box<WireRequest>,
        _context: &RequestContext,
    ) -> impl Future<Output = Result<Box<WireRequest>, R::Error>> + Send {
        ready(Ok(wire))
    }

    fn transform_response<'a>(
        &'a self,
        _runtime: R::Context<'a>,
        response: R::Response,
        _timing: Timing,
    ) -> impl Future<Output = Result<R::Response, R::Error>> + Send {
        ready(Ok(response))
    }

    fn on_event<'a>(
        &'a self,
        _runtime: R::Context<'a>,
        _event: RuntimeCallEvent<'_, R>,
    ) -> impl Future<Output = Result<(), R::Error>> + Send {
        ready(Ok(()))
    }

    fn on_stream_open(&self, _head: &R::Response) -> Result<(), R::Error> {
        Ok(())
    }

    fn on_stream_chunk(&self, _chunk: &R::Chunk) -> Result<(), R::Error> {
        Ok(())
    }
}
