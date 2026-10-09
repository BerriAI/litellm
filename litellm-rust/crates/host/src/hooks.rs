use std::sync::Arc;

use crate::{
    error::HookError,
    interceptors::{ExecutionFacts, RawResponse, RequestContext, WireRequest},
    lifecycle::{CallEvent, Timing},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallBoundary {
    PrepareArguments,
    PrepareRequest,
    BeforeProviderRequest,
    AfterProviderResponse,
    TransformResponse,
    Succeeded,
    Failed,
    StreamOpened,
    StreamChunk,
}

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

    fn prepare_request(
        &mut self,
        _runtime: R::Context<'_>,
        arguments: R::Arguments,
    ) -> Result<R::Step<Self, R::Arguments>, R::Error> {
        Ok(R::ready(arguments))
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

    /// The call ended before its terminal event. A runtime delivers this without awaiting
    /// and without a result, so a hook can only record it.
    fn on_cancelled(&mut self, _runtime: R::Context<'_>, _timing: Timing) {}
}

pub trait NativeHooks: Send + Sync {
    fn before_provider_request(
        &self,
        wire: Box<WireRequest>,
        _context: &RequestContext,
    ) -> Result<Box<WireRequest>, HookError> {
        Ok(wire)
    }

    fn result_ready(&self, _facts: &ExecutionFacts) -> Result<(), HookError> {
        Ok(())
    }

    fn on_event(&self, _event: &CallEvent) {}
}

impl<T: NativeHooks + ?Sized> NativeHooks for &T {
    fn before_provider_request(
        &self,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> Result<Box<WireRequest>, HookError> {
        (**self).before_provider_request(wire, context)
    }

    fn result_ready(&self, facts: &ExecutionFacts) -> Result<(), HookError> {
        (**self).result_ready(facts)
    }

    fn on_event(&self, event: &CallEvent) {
        (**self).on_event(event)
    }
}

impl<T: NativeHooks + ?Sized> NativeHooks for Arc<T> {
    fn before_provider_request(
        &self,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> Result<Box<WireRequest>, HookError> {
        (**self).before_provider_request(wire, context)
    }

    fn result_ready(&self, facts: &ExecutionFacts) -> Result<(), HookError> {
        (**self).result_ready(facts)
    }

    fn on_event(&self, event: &CallEvent) {
        (**self).on_event(event)
    }
}

impl<T: NativeHooks + ?Sized> NativeHooks for Box<T> {
    fn before_provider_request(
        &self,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> Result<Box<WireRequest>, HookError> {
        (**self).before_provider_request(wire, context)
    }

    fn result_ready(&self, facts: &ExecutionFacts) -> Result<(), HookError> {
        (**self).result_ready(facts)
    }

    fn on_event(&self, event: &CallEvent) {
        (**self).on_event(event)
    }
}

impl NativeHooks for () {}
