use std::time::Instant;

/// Where the router reads time for the attempt records and the plan's overall timeout.
/// Waiting is not here: a backoff is a [`crate::RoutingOp::Backoff`] the host performs.
pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// The wall clock. Kept under the name in-process hosts used.
pub type TokioClock = SystemClock;
