use std::time::Duration;

use crate::attempt::AttemptDisposition;
use crate::plan::{DeploymentId, LogicalCallId};

/// What the proxy used to read off the shared `Logging` object, returned as a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallReport<R, E> {
    pub id: LogicalCallId,
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
    pub failed: Option<AttemptDisposition>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CallFailure<E> {
    /// Terminal error of the last attempt, identity preserved for the host to map.
    #[error("every candidate was tried")]
    Exhausted { last: E },
    #[error("the request cannot succeed on any deployment")]
    Fatal { error: E },
    /// The attempt budget or timeout was hit before any success or fatal error.
    #[error("attempt budget or timeout exhausted")]
    Budget { last: Option<E> },
    /// The host failed a pending op or the caller cancelled mid-attempt.
    #[error("the logical call was interrupted")]
    Interrupted { error: E },
    /// The host answered a routing op out of turn or with the wrong variant.
    #[error("routing protocol violated: {0}")]
    Protocol(&'static str),
}
