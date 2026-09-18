use std::time::Duration;

/// Minted by the router at logical-call start; never accepted from outside. Spans every
/// retry and fallback attempt of one user call.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalCallId {
    pub call_id: String,
    pub trace_id: String,
}

/// Host-local handle for one configured deployment. The router never sees model strings,
/// credentials or host objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeploymentId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub jitter: bool,
    /// Honor a provider `Retry-After` when present; `max_backoff` still caps it.
    pub respect_retry_after: bool,
}

impl RetryPolicy {
    pub const fn none() -> Self {
        Self {
            max_retries: 0,
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
            jitter: false,
            respect_retry_after: false,
        }
    }
}

/// One configured deployment inside a plan group. Weight is the `simple-shuffle` weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deployment {
    pub id: DeploymentId,
    pub weight: u32,
}

impl Deployment {
    pub const fn new(id: DeploymentId) -> Self {
        Self { id, weight: 1 }
    }
}

/// Resolved before the first attempt, outside the loop: model-group resolution, tag and
/// budget filters, and pre-routing strategies (auto, complexity, quality) all produce this.
/// Fallback chains are groups: `[primary], [fallback_a], [fallback_b]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutePlan {
    /// The primary group is tried with retries, then each fallback group in order.
    pub groups: Vec<Vec<Deployment>>,
    pub retry: RetryPolicy,
    /// Hard ceiling on attempts across all groups; `None` lets `groups * retries` bound it.
    pub attempt_budget: Option<u32>,
    pub timeout: Option<Duration>,
}

impl RoutePlan {
    pub fn single(deployments: Vec<Deployment>, retry: RetryPolicy) -> Self {
        Self {
            groups: vec![deployments],
            retry,
            attempt_budget: None,
            timeout: None,
        }
    }
}
