use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    RateLimited,
    Timeout,
    Unavailable,
    ContextWindow,
    ContentPolicy,
    Permanent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttemptOutcome {
    Succeeded,
    Failed(Failure),
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetryDecision {
    Stop,
    Retry { after: Duration },
    Fallback { model: String },
}
