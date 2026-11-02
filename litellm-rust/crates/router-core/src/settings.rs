//! [`Settings`]: one typed, frozen view of the configuration a caller may read, built from the
//! static config and the current dynamic snapshot.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use crate::{
    config::{
        AllowedFailsPolicy, DEFAULT_COOLDOWN_TIME, PluginSpec, RetryPolicy, RouterConfig,
        RouterDynamicConfig, RoutingStrategy,
    },
    error::AttemptError,
};

#[derive(Clone, Debug)]
pub struct Settings {
    pub routing_strategy: RoutingStrategy,
    pub plugins: Arc<[PluginSpec]>,
    pub cooldown_time: Duration,
    pub disable_cooldowns: bool,
    pub allowed_fails: Option<u32>,
    pub allowed_fails_policy: Option<AllowedFailsPolicy>,
    pub retry_policy: Option<RetryPolicy>,
    pub model_group_retry_policy: Arc<BTreeMap<String, RetryPolicy>>,
    pub num_retries: u32,
    pub max_fallbacks: u32,
    pub retry_after: Duration,
    pub timeout: Option<Duration>,
    pub drop_params: Option<bool>,
    pub enable_pre_call_checks: bool,
    pub routing_plugins: bool,
    pub callback_filters: bool,
    pub enable_health_check_routing: bool,
    pub health_check_staleness: Duration,
    pub background_health_check_model_groups: Option<Arc<[String]>>,
    pub enable_weighted_failover: bool,
    pub fallback_on_unsupported_params: bool,
    pub provider_default_deployment_ids: Arc<[String]>,
    pub dynamic: Arc<RouterDynamicConfig>,
}

/// `DEFAULT_HEALTH_CHECK_INTERVAL * DEFAULT_HEALTH_CHECK_STALENESS_MULTIPLIER`.
pub const DEFAULT_HEALTH_CHECK_STALENESS: Duration = Duration::from_secs(300 * 2);

fn seconds(value: f64) -> Duration {
    Duration::try_from_secs_f64(value).unwrap_or_default()
}

impl Settings {
    pub fn new(config: &RouterConfig, dynamic: Arc<RouterDynamicConfig>) -> Self {
        Self {
            routing_strategy: config.routing_strategy.clone(),
            plugins: config.plugins.clone().into(),
            cooldown_time: config.cooldown_time.map_or(DEFAULT_COOLDOWN_TIME, seconds),
            disable_cooldowns: config.disable_cooldowns,
            allowed_fails: dynamic.allowed_fails,
            allowed_fails_policy: config.allowed_fails_policy.clone(),
            retry_policy: config.retry_policy.clone(),
            model_group_retry_policy: Arc::new(config.model_group_retry_policy.clone()),
            num_retries: dynamic.num_retries.unwrap_or(0),
            max_fallbacks: dynamic.max_fallbacks,
            retry_after: seconds(config.retry_after),
            timeout: config.request_timeout.or(config.timeout).map(seconds),
            drop_params: config.drop_params,
            enable_pre_call_checks: config.enable_pre_call_checks,
            routing_plugins: config.routing_plugins,
            callback_filters: config.callback_filters,
            enable_health_check_routing: config.enable_health_check_routing,
            health_check_staleness: config
                .health_check_staleness_threshold
                .map_or(DEFAULT_HEALTH_CHECK_STALENESS, seconds),
            background_health_check_model_groups: config
                .background_health_check_model_groups
                .clone()
                .map(Into::into),
            enable_weighted_failover: config.enable_weighted_failover,
            fallback_on_unsupported_params: config.fallback_on_unsupported_params.unwrap_or(true),
            provider_default_deployment_ids: config.provider_default_deployment_ids.clone().into(),
            dynamic,
        }
    }

    /// `allowed_fails_for`: the policy's count for this failure.
    pub fn allowed_fails_for(&self, failure: &AttemptError) -> Option<u32> {
        self.allowed_fails_policy
            .as_ref()
            .and_then(|policy| policy.allowed_fails_for(failure))
    }

    /// How long an allowed-fails counter lives: `settings().cooldown_time or
    /// DEFAULT_COOLDOWN_TIME_SECONDS`, so a router cooldown of 0 still counts over the default.
    pub fn fails_window(&self) -> Duration {
        if self.cooldown_time.is_zero() {
            DEFAULT_COOLDOWN_TIME
        } else {
            self.cooldown_time
        }
    }

    /// Python's `allowed_fails` default when set without a value (`litellm.allowed_fails`).
    pub fn effective_allowed_fails(&self) -> u32 {
        self.allowed_fails.unwrap_or(3)
    }
}
