use futures_util::future::BoxFuture;

use super::{ArgumentHooks, Cancelled, OutcomeHooks, ProviderHooks, ResponseHooks, Surface};
use crate::{
    interceptors::{ExecutionFacts, RawResponse, RequestContext, WireRequest},
    lifecycle::{FailureOrigin, Timing},
};

/// Every facet behind one object-safe trait, so [`super::Hooks`] can store groups of
/// different types. Implemented for anything that implements the four facets.
pub trait AnyGroup<S: Surface>: Send + Sync {
    fn prepare<'a>(
        &'a self,
        task: S::Task<'a>,
        arguments: S::Arguments,
        started_at: f64,
    ) -> BoxFuture<'a, Result<S::Arguments, S::Error>>;
    fn prepared(&self, arguments: &S::Arguments) -> Result<(), S::Error>;
    fn before_provider_request<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        wire: Box<WireRequest>,
        context: &'b RequestContext,
    ) -> BoxFuture<'b, Result<Box<WireRequest>, S::Error>>;
    fn after_provider_response<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        raw: &'b RawResponse,
    ) -> BoxFuture<'b, Result<(), S::Error>>;
    fn result_ready<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        facts: &'b ExecutionFacts,
    ) -> BoxFuture<'b, Result<(), S::Error>>;
    fn transform<'a>(
        &'a self,
        task: S::Task<'a>,
        response: S::Response,
        timing: Timing,
    ) -> BoxFuture<'a, Result<S::Response, S::Error>>;
    fn stream_opened(&self, head: &S::Response) -> Result<(), S::Error>;
    fn stream_chunk(&self, chunk: &S::Chunk) -> Result<(), S::Error>;
    fn started<'a>(
        &'a self,
        task: S::Task<'a>,
        start_time: f64,
    ) -> BoxFuture<'a, Result<(), Cancelled<S::Error>>>;
    fn succeeded<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        timing: Timing,
        response: &'b S::Response,
    ) -> BoxFuture<'b, Result<(), Cancelled<S::Error>>>;
    fn failed<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        timing: Timing,
        origin: FailureOrigin,
        error: &'b S::Failure,
    ) -> BoxFuture<'b, Result<(), Cancelled<S::Error>>>;
    fn cancelled<'a>(
        &'a self,
        task: S::Task<'a>,
        timing: Timing,
    ) -> BoxFuture<'a, Result<(), Cancelled<S::Error>>>;
}

impl<S, H> AnyGroup<S> for H
where
    S: Surface,
    H: ArgumentHooks<S> + ProviderHooks<S> + ResponseHooks<S> + OutcomeHooks<S>,
{
    fn prepare<'a>(
        &'a self,
        task: S::Task<'a>,
        arguments: S::Arguments,
        started_at: f64,
    ) -> BoxFuture<'a, Result<S::Arguments, S::Error>> {
        Box::pin(ArgumentHooks::prepare(self, task, arguments, started_at))
    }

    fn prepared(&self, arguments: &S::Arguments) -> Result<(), S::Error> {
        ArgumentHooks::prepared(self, arguments)
    }

    fn before_provider_request<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        wire: Box<WireRequest>,
        context: &'b RequestContext,
    ) -> BoxFuture<'b, Result<Box<WireRequest>, S::Error>> {
        Box::pin(ProviderHooks::before_provider_request(
            self, task, wire, context,
        ))
    }

    fn after_provider_response<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        raw: &'b RawResponse,
    ) -> BoxFuture<'b, Result<(), S::Error>> {
        Box::pin(ProviderHooks::after_provider_response(self, task, raw))
    }

    fn result_ready<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        facts: &'b ExecutionFacts,
    ) -> BoxFuture<'b, Result<(), S::Error>> {
        Box::pin(ProviderHooks::result_ready(self, task, facts))
    }

    fn transform<'a>(
        &'a self,
        task: S::Task<'a>,
        response: S::Response,
        timing: Timing,
    ) -> BoxFuture<'a, Result<S::Response, S::Error>> {
        Box::pin(ResponseHooks::transform(self, task, response, timing))
    }

    fn stream_opened(&self, head: &S::Response) -> Result<(), S::Error> {
        ResponseHooks::stream_opened(self, head)
    }

    fn stream_chunk(&self, chunk: &S::Chunk) -> Result<(), S::Error> {
        ResponseHooks::stream_chunk(self, chunk)
    }

    fn started<'a>(
        &'a self,
        task: S::Task<'a>,
        start_time: f64,
    ) -> BoxFuture<'a, Result<(), Cancelled<S::Error>>> {
        Box::pin(OutcomeHooks::started(self, task, start_time))
    }

    fn succeeded<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        timing: Timing,
        response: &'b S::Response,
    ) -> BoxFuture<'b, Result<(), Cancelled<S::Error>>> {
        Box::pin(OutcomeHooks::succeeded(self, task, timing, response))
    }

    fn failed<'a: 'b, 'b>(
        &'a self,
        task: S::Task<'a>,
        timing: Timing,
        origin: FailureOrigin,
        error: &'b S::Failure,
    ) -> BoxFuture<'b, Result<(), Cancelled<S::Error>>> {
        Box::pin(OutcomeHooks::failed(self, task, timing, origin, error))
    }

    fn cancelled<'a>(
        &'a self,
        task: S::Task<'a>,
        timing: Timing,
    ) -> BoxFuture<'a, Result<(), Cancelled<S::Error>>> {
        Box::pin(OutcomeHooks::cancelled(self, task, timing))
    }
}
