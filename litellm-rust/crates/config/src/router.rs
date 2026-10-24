use std::collections::BTreeMap;

use litellm_auth_types::SecretValue;

use crate::{Object, Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct RouterSettings {
    pub redis_url: Option<SecretValue>,
    pub redis_host: Option<String>,
    pub redis_port: Option<Spelled<i64>>,
    pub redis_password: Option<SecretValue>,
    pub redis_db: Option<Spelled<i64>>,
    pub cache_responses: Option<Spelled<bool>>,
    pub cache_kwargs: Option<Object>,
    pub caching_groups: Option<Raw>,
    pub client_ttl: Option<Spelled<i64>>,
    pub polling_interval: Option<Spelled<f64>>,
    pub default_priority: Option<Raw>,
    pub num_retries: Option<u64>,
    pub max_fallbacks: Option<Spelled<i64>>,
    pub timeout: Option<f64>,
    pub stream_timeout: Option<Spelled<f64>>,
    pub default_litellm_params: Option<Object>,
    pub default_max_parallel_requests: Option<Spelled<i64>>,
    pub set_verbose: Option<Spelled<bool>>,
    pub debug_level: Option<String>,
    pub default_fallbacks: Option<Box<[String]>>,
    pub fallbacks: Option<Box<[Object]>>,
    pub context_window_fallbacks: Option<Box<[Object]>>,
    pub content_policy_fallbacks: Option<Box<[Object]>>,
    pub model_group_alias: Option<Object>,
    pub enable_pre_call_checks: Option<Spelled<bool>>,
    pub enable_tag_filtering: Option<bool>,
    pub tag_filtering_match_any: Option<Spelled<bool>>,
    pub tag_routing_prefix: Option<String>,
    pub plugins: Option<Box<[String]>>,
    pub retry_after: Option<f64>,
    pub retry_policy: Option<Object>,
    pub model_group_retry_policy: Option<Object>,
    pub allowed_fails: Option<u64>,
    pub allowed_fails_policy: Option<AllowedFailsPolicy>,
    pub cooldown_time: Option<f64>,
    pub disable_cooldowns: Option<Spelled<bool>>,
    pub fallback_on_unsupported_params: Option<Spelled<bool>>,
    pub routing_strategy: Option<String>,
    pub optional_pre_call_checks: Option<Box<[String]>>,
    pub routing_strategy_args: Option<Object>,
    pub routing_groups: Option<Box<[Object]>>,
    pub provider_budget_config: Option<BTreeMap<String, ProviderBudget>>,
    pub alerting_config: Option<AlertingConfig>,
    pub deployment_affinity_ttl_seconds: Option<Spelled<i64>>,
    pub model_group_affinity_config: Option<Object>,
    pub enable_health_check_routing: Option<Spelled<bool>>,
    pub health_check_staleness_threshold: Option<Spelled<f64>>,
    pub health_check_ignore_transient_errors: Option<Spelled<bool>>,
    pub background_health_check_model_groups: Option<Box<[String]>>,
    pub enable_weighted_failover: Option<Spelled<bool>>,
    pub enable_mid_stream_fallback_continuation: Option<Spelled<bool>>,
    pub guardrail_list: Option<Box<[Object]>>,
    pub max_retries: Option<u64>,
    pub weights: Option<Object>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[macro_rules_attribute::apply(crate::config_section)]
#[serde(rename_all = "PascalCase")]
pub struct AllowedFailsPolicy {
    pub bad_request_error_allowed_fails: Option<Spelled<i64>>,
    pub authentication_error_allowed_fails: Option<Spelled<i64>>,
    pub timeout_error_allowed_fails: Option<Spelled<i64>>,
    pub rate_limit_error_allowed_fails: Option<Spelled<i64>>,
    pub content_policy_violation_error_allowed_fails: Option<Spelled<i64>>,
    pub internal_server_error_allowed_fails: Option<Spelled<i64>>,
    pub service_unavailable_error_allowed_fails: Option<Spelled<i64>>,
    pub bad_gateway_error_allowed_fails: Option<Spelled<i64>>,
    pub not_found_error_allowed_fails: Option<Spelled<i64>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct ProviderBudget {
    pub budget_limit: Option<Spelled<f64>>,
    pub time_period: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct AlertingConfig {
    pub webhook_url: Option<SecretValue>,
    pub alerting_threshold: Option<Spelled<f64>>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Config, RouterSettings, Spelled};

    fn router(yaml: &str) -> RouterSettings {
        Config::from_yaml(&format!("router_settings:\n{yaml}"))
            .unwrap()
            .router_settings
    }

    #[rstest]
    #[case::int("  redis_port: 6379", |s: &RouterSettings| s.redis_port == Some(Spelled::Value(6379)))]
    #[case::int_env(
    "  client_ttl: os.environ/CLIENT_TTL",
    |s: &RouterSettings| s.client_ttl == Some(Spelled::Text("os.environ/CLIENT_TTL".to_string()))
)]
    #[case::float_from_int("  stream_timeout: 30", |s: &RouterSettings| s.stream_timeout == Some(Spelled::Value(30.0)))]
    #[case::float(
    "  health_check_staleness_threshold: 2.5",
    |s: &RouterSettings| s.health_check_staleness_threshold == Some(Spelled::Value(2.5))
)]
    #[case::bool("  enable_weighted_failover: true", |s: &RouterSettings| s.enable_weighted_failover == Some(Spelled::Value(true)))]
    #[case::bool_word(
    "  enable_mid_stream_fallback_continuation: \"true\"",
    |s: &RouterSettings| s.enable_mid_stream_fallback_continuation == Some(Spelled::Text("true".to_string()))
)]
    #[case::string("  redis_host: cache.internal", |s: &RouterSettings| s.redis_host.as_deref() == Some("cache.internal"))]
    #[case::string_list(
    "  default_fallbacks: [gpt-small, gpt-large]",
    |s: &RouterSettings| s.default_fallbacks.as_deref() == Some(&["gpt-small".to_string(), "gpt-large".to_string()][..])
)]
    #[case::mapping_list(
    "  content_policy_fallbacks: [{gpt: [claude]}]",
    |s: &RouterSettings| s.content_policy_fallbacks.as_ref().is_some_and(|items| items[0].contains_key("gpt"))
)]
    #[case::mapping(
    "  default_litellm_params: {temperature: 0}",
    |s: &RouterSettings| s.default_litellm_params.as_ref().is_some_and(|params| params.contains_key("temperature"))
)]
    #[case::non_integer_priority("  default_priority: 1.5", |s: &RouterSettings| s.default_priority.is_some())]
    #[case::bool_priority("  default_priority: true", |s: &RouterSettings| s.default_priority.is_some())]
    #[case::caching_groups(
    "  caching_groups: [[gpt, [gpt-a, gpt-b]]]",
    |s: &RouterSettings| s.caching_groups.as_ref().is_some_and(|groups| groups.get().is_sequence())
)]
    #[case::allowed_fails_policy(
    "  allowed_fails_policy: {RateLimitErrorAllowedFails: \"3\", BadGatewayErrorAllowedFails: 2, FutureAllowedFails: 1}",
    |s: &RouterSettings| s.allowed_fails_policy.as_ref().is_some_and(|policy| {
        policy.rate_limit_error_allowed_fails == Some(Spelled::Text("3".to_string()))
            && policy.bad_gateway_error_allowed_fails == Some(Spelled::Value(2))
    })
)]
    #[case::provider_budget_config(
    "  provider_budget_config: {openai: {budget_limit: 100, time_period: 1d, rpm_limit: 5}}",
    |s: &RouterSettings| s.provider_budget_config.as_ref().is_some_and(|budgets| {
        let openai = &budgets["openai"];
        openai.budget_limit == Some(Spelled::Value(100.0))
            && openai.time_period.as_deref() == Some("1d")
    })
)]
    #[case::alerting_config(
    "  alerting_config: {webhook_url: https://hooks.example/a, alerting_threshold: 10, channel: ops}",
    |s: &RouterSettings| s.alerting_config.as_ref().is_some_and(|alerting| {
        alerting.webhook_url.as_ref().map(|url| url.expose()) == Some("https://hooks.example/a")
            && alerting.alerting_threshold == Some(Spelled::Value(10.0))
    })
)]
    fn parses_python_router_spellings(
        #[case] yaml: &str,
        #[case] check: fn(&RouterSettings) -> bool,
    ) {
        assert!(check(&router(yaml)));
    }

    #[rstest]
    #[case::unknown("future_router_setting")]
    #[case::moved_to_general_settings("health_check_interval")]
    #[case::managed_outside_config("model_list")]
    fn keeps_keys_outside_the_router_signature(#[case] key: &str) {
        let settings = router(&format!("  {key}: 60"));
        assert!(settings.additional_fields.contains_key(key));
    }

    #[rstest]
    #[case::redis_password("  redis_password: router-sentinel")]
    #[case::redis_url("  redis_url: redis://user:router-sentinel@cache:6379")]
    #[case::webhook_url("  alerting_config: {webhook_url: https://hooks.example/router-sentinel}")]
    #[case::cache_kwargs("  cache_kwargs: {password: router-sentinel}")]
    #[case::default_priority("  default_priority: router-sentinel")]
    #[case::caching_groups("  caching_groups: [[router-sentinel, [a]]]")]
    #[case::guardrail_list("  guardrail_list: [{api_key: router-sentinel}]")]
    #[case::nested_extension("  allowed_fails_policy: {FutureAllowedFails: router-sentinel}")]
    fn debug_output_does_not_expose_router_values(#[case] yaml: &str) {
        let config = Config::from_yaml(&format!("router_settings:\n{yaml}")).unwrap();
        assert!(!format!("{config:?}").contains("router-sentinel"));
    }
}
