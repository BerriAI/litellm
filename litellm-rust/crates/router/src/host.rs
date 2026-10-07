//! What the router asks of its host. The host performs every attempt and owns every object
//! the router treats as opaque: responses, exceptions and the request's metadata buckets.

use std::future::Future;

use crate::failure::{Classified, Raised};

/// The `mock_testing_*` request flags. The fallback ones fail the first hop before any
/// attempt; `RateLimit` replaces the first hop's first attempt, so its retries still run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MockFailure {
    Fallbacks,
    ContextWindowFallbacks,
    ContentPolicyFallbacks,
    RateLimit,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Deployment(String),
    Mock(MockFailure),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypedFallback {
    ContextWindow,
    ContentPolicy,
}

/// The metadata a fallback hop's fresh bucket carries (`run_async_fallback`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HopStamp {
    pub original_model_group: String,
    pub model_group: String,
    pub attempted_fallbacks: u32,
    pub max_fallbacks: u32,
}

/// Writes the host applies, in order, to the request's metadata buckets and exceptions. They
/// ride on the next attempt or on the final result, so the host sees them in the order Python
/// would have made them.
#[derive(Clone, Debug)]
pub enum Op<E> {
    /// `Router.log_retry(kwargs, error)` against `bucket`, with `kwargs["model"] == model`.
    LogRetry {
        bucket: u32,
        model: String,
        error: Raised<E>,
    },
    /// A fallback hop's bucket: a shallow copy of `copy_of`, stamped.
    OpenBucket {
        id: u32,
        copy_of: u32,
        stamp: HopStamp,
    },
    /// Retries ran out: `error.max_retries` and `error.num_retries`.
    StampRetries {
        error: Raised<E>,
        max_retries: u32,
        num_retries: u32,
    },
    /// A context-window or content-policy error with no list of its own for this hop.
    MissingTypedFallbacks {
        error: Raised<E>,
        kind: TypedFallback,
        model_group: String,
    },
    /// No fallback chain matched any of `lookup_groups`.
    NoFallbackGroup {
        error: Raised<E>,
        lookup_groups: Vec<String>,
    },
    /// The outcome of the top-level fallback attempt, appended to the error the caller sees.
    FallbackOutcome {
        error: Raised<E>,
        model_group: String,
        attempted: Option<Vec<String>>,
        last: Option<Raised<E>>,
    },
}

/// The retry layer's metadata for the attempt about to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryStamp {
    pub model_group_size: usize,
    pub attempted_retries: u32,
    pub max_retries: u32,
}

#[derive(Clone, Debug)]
pub struct Attempt<E> {
    pub target: Target,
    /// The hop's `kwargs["model"]`.
    pub model_group: String,
    pub bucket: u32,
    pub fallback_depth: u32,
    pub retry: RetryStamp,
    pub ops: Vec<Op<E>>,
}

pub enum Invoked<R, E> {
    Success(R),
    Failure { error: E, classified: Classified },
}

pub trait RouterHost: Send + Sync {
    type Response: Send;
    type Error: Clone + Send + Sync;
    /// The host could not answer an op (it went away, or the caller cancelled).
    type Fault: Send;

    fn invoke(
        &self,
        attempt: Attempt<Self::Error>,
    ) -> impl Future<Output = Result<Invoked<Self::Response, Self::Error>, Self::Fault>> + Send;

    fn sleep(&self, seconds: f64) -> impl Future<Output = Result<(), Self::Fault>> + Send;
}
