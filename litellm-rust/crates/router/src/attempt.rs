use std::time::Duration;

use litellm_callbacks::failure::{Classified, FailureClass};
use litellm_callbacks::machine::Machine;
use litellm_callbacks::route::Route;

use crate::plan::DeploymentId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptContext {
    /// Spans every attempt of the logical call; Python's `litellm_trace_id`.
    pub trace_id: String,
    /// 0-based across the whole logical call.
    pub attempt_index: u32,
    pub deployment: DeploymentId,
    /// 0 is the primary group; n > 0 is fallback depth.
    pub group_index: u32,
}

/// One provider attempt: a route machine whose errors say what kind of failure they are.
pub trait Attempt: Machine {
    fn class(error: &<Self::Route as Route>::Error) -> FailureClass;

    fn retry_after(error: &<Self::Route as Route>::Error) -> Option<Duration>;
}

impl<M> Attempt for M
where
    M: Machine,
    <M::Route as Route>::Error: Classified,
{
    fn class(error: &<M::Route as Route>::Error) -> FailureClass {
        error.class()
    }

    fn retry_after(error: &<M::Route as Route>::Error) -> Option<Duration> {
        error.retry_after()
    }
}

/// Starts a fresh attempt machine for a chosen deployment. This is the seam to core via
/// the host, and the seam tests fake.
pub trait AttemptFactory: Send + Sync {
    type Attempt: Attempt + 'static;

    fn start(&self, context: &AttemptContext) -> Self::Attempt;
}
