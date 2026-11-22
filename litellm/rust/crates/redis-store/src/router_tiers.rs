use std::time::Duration;

use crate::TierPolicy;

/// The tier settings the router's stores share, as the host reads them: a Python process from its
/// `litellm` globals, the gateway from its environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterTiers {
    /// `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS`.
    pub cooldown_refresh: Duration,
    /// `litellm.default_redis_batch_cache_expiry`.
    pub usage_refresh: Duration,
    /// `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`.
    pub tracked_reads: usize,
    /// `litellm.default_in_memory_ttl`.
    pub default_in_memory_ttl: Option<Duration>,
}

impl RouterTiers {
    pub fn cooldowns(&self) -> TierPolicy {
        self.policy(self.cooldown_refresh)
    }

    pub fn usage(&self) -> TierPolicy {
        self.policy(self.usage_refresh)
    }

    fn policy(&self, refresh_after: Duration) -> TierPolicy {
        TierPolicy {
            refresh_after,
            tracked_reads: self.tracked_reads,
            default_in_memory_ttl: self.default_in_memory_ttl,
        }
    }
}

/// What a Python process runs with when neither the environment nor `litellm_settings` says otherwise.
impl Default for RouterTiers {
    fn default() -> Self {
        Self {
            cooldown_refresh: Duration::from_secs(1),
            usage_refresh: Duration::from_secs(10),
            tracked_reads: 1000,
            default_in_memory_ttl: None,
        }
    }
}
