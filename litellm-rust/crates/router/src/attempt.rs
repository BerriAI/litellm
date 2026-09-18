use std::time::Duration;

use litellm_callbacks::machine::Machine;
use litellm_callbacks::route::Route;

use crate::plan::{DeploymentId, LogicalCallId};

/// What an attempt's error means for the loop. Classification is a fact the attempt
/// reports; the router never inspects provider errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttemptDisposition {
    /// The same deployment may be retried after backoff (429, 5xx, timeout).
    Retryable { retry_after: Option<Duration> },
    /// The deployment is bad for this call; try the next candidate or group.
    Reroute { cooldown: Option<Duration> },
    /// No attempt anywhere can succeed; stop now.
    Fatal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptContext {
    pub id: LogicalCallId,
    /// 0-based across the whole logical call.
    pub attempt_index: u32,
    pub deployment: DeploymentId,
    /// 0 is the primary group; n > 0 is fallback depth.
    pub group_index: u32,
}

/// A route error that knows what it means for the loop. Implemented by each route's error
/// type, so any machine over that route, layered or not, is an [`Attempt`].
pub trait AttemptError {
    fn disposition(&self) -> AttemptDisposition;
}

/// One provider attempt: a route machine whose errors carry their own disposition.
pub trait Attempt: Machine {
    fn disposition(error: &<Self::Route as Route>::Error) -> AttemptDisposition;
}

impl<M> Attempt for M
where
    M: Machine,
    <M::Route as Route>::Error: AttemptError,
{
    fn disposition(error: &<M::Route as Route>::Error) -> AttemptDisposition {
        error.disposition()
    }
}

/// Starts a fresh attempt machine for a chosen deployment. This is the seam to core via
/// the host, and the seam tests fake.
pub trait AttemptFactory: Send + Sync {
    type Attempt: Attempt + 'static;

    fn start(&self, context: &AttemptContext) -> Self::Attempt;
}
