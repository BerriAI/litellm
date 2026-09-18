use std::time::Duration;

use litellm_callbacks::failure::FailureClass;

use crate::decide::Decision;
use crate::plan::DeploymentId;

/// What the proxy used to read off the shared `Logging` object, returned as a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallReport<R, E> {
    /// Spans every attempt; Python's `litellm_trace_id`.
    pub trace_id: String,
    pub outcome: Result<R, CallFailure<E>>,
    pub attempts: Vec<AttemptRecord>,
    /// The deployment that produced the terminal outcome, if any attempt ran.
    pub selected: Option<DeploymentId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptRecord {
    pub deployment: DeploymentId,
    pub group_index: u32,
    /// Offsets from logical-call start on the injected clock.
    pub started: Duration,
    pub ended: Duration,
    /// What the attempt reported and what the loop decided about it; `None` on success.
    pub failure: Option<(FailureClass, Decision)>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CallFailure<E> {
    /// Every group was tried, or the loop decided to stop: the last error is the call's.
    #[error("every candidate was tried")]
    Exhausted { last: E },
    /// The attempt budget or timeout was hit, or there was nothing to attempt.
    #[error("attempt budget or timeout exhausted")]
    Budget { last: Option<E> },
    /// The attempt failed after it had streamed `chunks`; nothing is retried mid-stream.
    #[error("the response failed after {chunks} chunks")]
    MidStream { error: E, chunks: u32 },
    /// The host failed a pending op or the caller cancelled mid-attempt.
    #[error("the logical call was interrupted")]
    Interrupted { error: E },
    /// The host answered a routing op out of turn or with the wrong variant.
    #[error("routing protocol violated: {0}")]
    Protocol(&'static str),
}
