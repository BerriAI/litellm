//! What a call lets its host hook into, one trait per payload. A host picks the facets
//! its public surface has and composes any number of groups through [`Hooks`].

mod chain;
mod group;

use std::future::{Future, ready};

pub use chain::Hooks;
pub use group::AnyGroup;

use crate::{
    interceptors::{ExecutionFacts, RawResponse, RequestContext, WireRequest},
    lifecycle::{FailureOrigin, Timing},
};

/// A host's public values as its hooks see them.
pub trait Surface: Sized + 'static {
    /// The caller's task a hook awaits on, or `()` for a host that never suspends.
    type Task<'a>: Copy + Send;
    type Arguments: Send;
    type Response: Send + Sync;
    type Chunk;
    type Error: Send + Sync;
    /// What a failed outcome carries: the host's retained form of its error.
    type Failure: Sync;
    /// How [`Hooks`] stores a mounted group.
    type Group: ?Sized + AnyGroup<Self> + Send + Sync;
}

/// A hook stopped the call: nothing after it runs and the error is raised as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cancelled<E>(pub E);

/// The caller's arguments before the route decodes them. Groups run in mount order and
/// each receives the previous group's output; the first error stops the chain.
pub trait ArgumentHooks<S: Surface>: Send + Sync {
    fn prepare<'a>(
        &'a self,
        _task: S::Task<'a>,
        arguments: S::Arguments,
        _started_at: f64,
    ) -> impl Future<Output = Result<S::Arguments, S::Error>> + Send {
        ready(Ok(arguments))
    }

    fn prepared(&self, _arguments: &S::Arguments) -> Result<(), S::Error> {
        Ok(())
    }
}

/// The provider exchange: the wire request about to leave, the raw response that came
/// back, and the facts of the execution. Groups run in mount order; the first error
/// interrupts the call.
pub trait ProviderHooks<S: Surface>: Send + Sync {
    fn before_provider_request<'a>(
        &'a self,
        _task: S::Task<'a>,
        wire: Box<WireRequest>,
        _context: &RequestContext,
    ) -> impl Future<Output = Result<Box<WireRequest>, S::Error>> + Send {
        ready(Ok(wire))
    }

    fn after_provider_response<'a>(
        &'a self,
        _task: S::Task<'a>,
        _raw: &RawResponse,
    ) -> impl Future<Output = Result<(), S::Error>> + Send {
        ready(Ok(()))
    }

    fn result_ready<'a>(
        &'a self,
        _task: S::Task<'a>,
        _facts: &ExecutionFacts,
    ) -> impl Future<Output = Result<(), S::Error>> + Send {
        ready(Ok(()))
    }
}

/// The public response and stream. Transforms unwind in reverse mount order, so the first
/// group mounted sees the response last; the first error stops the chain.
pub trait ResponseHooks<S: Surface>: Send + Sync {
    fn transform<'a>(
        &'a self,
        _task: S::Task<'a>,
        response: S::Response,
        _timing: Timing,
    ) -> impl Future<Output = Result<S::Response, S::Error>> + Send {
        ready(Ok(response))
    }

    fn stream_opened(&self, _head: &S::Response) -> Result<(), S::Error> {
        Ok(())
    }

    fn stream_chunk(&self, _chunk: &S::Chunk) -> Result<(), S::Error> {
        Ok(())
    }
}

/// How the call ended. Every group hears the outcome, terminal ones in reverse mount
/// order, and only a [`Cancelled`] group stops the rest. What a group does with its own
/// errors is its business.
pub trait OutcomeHooks<S: Surface>: Send + Sync {
    fn started<'a>(
        &'a self,
        _task: S::Task<'a>,
        _start_time: f64,
    ) -> impl Future<Output = Result<(), Cancelled<S::Error>>> + Send {
        ready(Ok(()))
    }

    fn succeeded<'a>(
        &'a self,
        _task: S::Task<'a>,
        _timing: Timing,
        _response: &S::Response,
    ) -> impl Future<Output = Result<(), Cancelled<S::Error>>> + Send {
        ready(Ok(()))
    }

    fn failed<'a>(
        &'a self,
        _task: S::Task<'a>,
        _timing: Timing,
        _origin: FailureOrigin,
        _error: &S::Failure,
    ) -> impl Future<Output = Result<(), Cancelled<S::Error>>> + Send {
        ready(Ok(()))
    }

    fn cancelled<'a>(
        &'a self,
        _task: S::Task<'a>,
        _timing: Timing,
    ) -> impl Future<Output = Result<(), Cancelled<S::Error>>> + Send {
        ready(Ok(()))
    }
}
