use std::{collections::BTreeMap, fmt};

use serde::Deserialize;
use serde_json::{Map, Value};

/// A proxy config's `router_settings`, as written. What each setting means, and its default when
/// unset, is the router core's.
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct RouterSettings {
    pub routing_strategy: Option<String>,
    pub routing_strategy_args: Option<Map<String, Value>>,
    pub routing_groups: Option<Vec<Map<String, Value>>>,
    pub retry_policy: Option<Map<String, Value>>,
    pub model_group_retry_policy: Option<Map<String, Value>>,
    pub allowed_fails_policy: Option<Map<String, Value>>,
    pub model_group_affinity_config: Option<Map<String, Value>>,
    pub allowed_fails: Option<u32>,
    pub cooldown_time: Option<f64>,
    pub disable_cooldowns: Option<bool>,
    pub num_retries: Option<u32>,
    pub timeout: Option<f64>,
    pub request_timeout: Option<f64>,
    pub max_retries: Option<u64>,
    pub retry_after: Option<f64>,
    pub fallbacks: Option<Vec<Value>>,
    pub context_window_fallbacks: Option<Vec<Value>>,
    pub content_policy_fallbacks: Option<Vec<Value>>,
    pub default_fallbacks: Option<Vec<String>>,
    pub max_fallbacks: Option<u32>,
    pub model_group_alias: Option<Map<String, Value>>,
    pub enable_pre_call_checks: Option<bool>,
    pub enable_weighted_failover: Option<bool>,
    pub enable_health_check_routing: Option<bool>,
    pub enable_cross_model_group_collision_check: Option<bool>,
    pub fallback_on_unsupported_params: Option<bool>,
    pub enable_tag_filtering: Option<bool>,
    pub weights: Option<Map<String, Value>>,
    pub tag_routing_prefix: Option<String>,
    pub optional_pre_call_checks: Option<Vec<String>>,
    #[serde(flatten)]
    pub additional_fields: BTreeMap<String, Value>,
}

impl fmt::Debug for RouterSettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RouterSettings")
            .field("routing_strategy", &self.routing_strategy)
            .field(
                "routing_groups",
                &self.routing_groups.as_ref().map(Vec::len),
            )
            .field("allowed_fails", &self.allowed_fails)
            .field("cooldown_time", &self.cooldown_time)
            .field("num_retries", &self.num_retries)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("retry_after", &self.retry_after)
            .field("enable_tag_filtering", &self.enable_tag_filtering)
            .field("tag_routing_prefix", &self.tag_routing_prefix)
            .field("optional_pre_call_checks", &self.optional_pre_call_checks)
            .field("additional_fields", &self.additional_fields.keys())
            .finish_non_exhaustive()
    }
}
