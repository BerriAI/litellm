use std::time::Duration;

use litellm_callbacks::failure::FailureClass;

use crate::decide::ChainsConfigured;

/// Host-local handle for one configured deployment. The router never sees model strings,
/// credentials or host objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeploymentId(pub u64);

/// One configured deployment. `weight` is the pick weight the host computed (litellm's
/// `simple-shuffle` reads `weight`, then `rpm`, then `tpm`); `retries` is the deployment's
/// own `num_retries`, honored when the request set none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deployment {
    pub id: DeploymentId,
    pub weight: u32,
    pub retries: Option<u32>,
}

impl Deployment {
    pub const fn new(id: DeploymentId) -> Self {
        Self {
            id,
            weight: 1,
            retries: None,
        }
    }

    pub const fn with_retries(self, retries: u32) -> Self {
        Self {
            retries: Some(retries),
            ..self
        }
    }
}

/// Where a retry count came from. Python resolves the request, then the deployment, then
/// the router: a count the caller set beats the deployment's own, the router's default
/// yields to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retries {
    Request(u32),
    Default(u32),
}

/// How many times one group is retried and how long each retry waits: Python's
/// `num_retries`, `retry_after`, per-class `RetryPolicy` and `_calculate_retry_after`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    pub retries: Retries,
    /// Python's `RetryPolicy`: retries per failure class, falling back to
    /// `class_default`. When a class resolves to a count, that count is the whole
    /// decision: the failure is retried that many times whatever its status, and none of
    /// the "should this be retried" checks run. A request that set retries to zero
    /// switches this off.
    pub class_retries: Vec<(FailureClass, u32)>,
    pub class_default: Option<u32>,
    /// The floor of a computed wait, Python's `retry_after`.
    pub min_backoff: Duration,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    /// Upper bound of the uniform jitter added to every computed wait.
    pub jitter: Duration,
    /// A provider `Retry-After` is obeyed when it is positive and at most this long.
    pub max_retry_after: Duration,
}

impl RetryPolicy {
    pub const fn none() -> Self {
        Self {
            retries: Retries::Default(0),
            class_retries: Vec::new(),
            class_default: None,
            min_backoff: Duration::ZERO,
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
            jitter: Duration::ZERO,
            max_retry_after: Duration::ZERO,
        }
    }

    /// Python's timing around a retry count: half a second doubling up to eight, up to
    /// three quarters of a second of jitter, `Retry-After` obeyed up to a minute, no floor.
    pub const fn python(retries: Retries) -> Self {
        Self {
            retries,
            class_retries: Vec::new(),
            class_default: None,
            min_backoff: Duration::ZERO,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(8),
            jitter: Duration::from_millis(750),
            max_retry_after: Duration::from_secs(60),
        }
    }

    pub(crate) fn has_class_policy(&self) -> bool {
        !self.class_retries.is_empty() || self.class_default.is_some()
    }

    /// The count the class policy grants, walking to the parent class the way Python walks
    /// the exception's MRO, then the policy default.
    pub(crate) fn class_retries(&self, class: FailureClass) -> Option<u32> {
        let mut lookup = Some(class);
        while let Some(class) = lookup {
            if let Some((_, retries)) = self.class_retries.iter().find(|(c, _)| *c == class) {
                return Some(*retries);
            }
            lookup = class.parent();
        }
        self.class_default
    }
}

/// Groups tried after the primary fails for good.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fallbacks {
    /// The last error is raised; Python's `disable_fallbacks`.
    Disabled,
    /// Chains the plan already knows, walked in process.
    Chains(FallbackChains),
    /// Every next group is a [`crate::RoutingOp::NextGroup`] the host answers, which is
    /// where the Python router's per-group lookups, access and budget checks live.
    Host,
}

/// Fallback groups keyed by the failure that leaves the primary group. Python tries the
/// chain for the class when one is configured and the generic chain otherwise; a class
/// chain that runs out does not continue into the generic one. The chain is chosen once,
/// by the primary group's last failure, and walked in order whatever later groups report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FallbackChains {
    pub generic: Vec<Vec<Deployment>>,
    pub context_window: Vec<Vec<Deployment>>,
    pub content_policy: Vec<Vec<Deployment>>,
    /// Groups tried after the primary, Python's `max_fallbacks`.
    pub max_depth: u32,
}

impl FallbackChains {
    /// Python's `ROUTER_MAX_FALLBACKS`.
    pub const PYTHON_MAX_DEPTH: u32 = 5;

    pub fn generic(groups: Vec<Vec<Deployment>>) -> Self {
        Self {
            generic: groups,
            ..Self::default()
        }
    }

    pub fn chain(&self, class: FailureClass) -> &[Vec<Deployment>] {
        match class {
            FailureClass::ContextWindow if !self.context_window.is_empty() => &self.context_window,
            FailureClass::ContentPolicy if !self.content_policy.is_empty() => &self.content_policy,
            _ => &self.generic,
        }
    }

    pub fn configured(&self) -> ChainsConfigured {
        ChainsConfigured {
            generic: !self.generic.is_empty(),
            context_window: !self.context_window.is_empty(),
            content_policy: !self.content_policy.is_empty(),
        }
    }

    /// The group `depth` hops from the primary, if the chain and the depth cap allow one.
    pub fn next(&self, class: FailureClass, depth: u32) -> Option<Vec<Deployment>> {
        if depth >= self.max_depth {
            return None;
        }
        self.chain(class).get(depth as usize).cloned()
    }
}

impl Default for FallbackChains {
    fn default() -> Self {
        Self {
            generic: Vec::new(),
            context_window: Vec::new(),
            content_policy: Vec::new(),
            max_depth: Self::PYTHON_MAX_DEPTH,
        }
    }
}

/// Resolved before the first attempt, outside the loop: model-group resolution, tag and
/// budget filters, and pre-routing strategies (auto, complexity, quality) all produce this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutePlan {
    /// Tried first, with retries; then the fallback groups.
    pub primary: Vec<Deployment>,
    pub retry: RetryPolicy,
    pub fallbacks: Fallbacks,
    /// Hard ceiling on attempts across all groups; `None` lets the groups and retries bound it.
    pub attempt_budget: Option<u32>,
    pub timeout: Option<Duration>,
}

impl RoutePlan {
    pub fn single(deployments: Vec<Deployment>, retry: RetryPolicy) -> Self {
        Self {
            primary: deployments,
            retry,
            fallbacks: Fallbacks::Disabled,
            attempt_budget: None,
            timeout: None,
        }
    }
}
