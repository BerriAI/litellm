use crate::Spelled;

#[macro_rules_attribute::apply(crate::config_section)]
pub struct LimitsSettings {
    pub fail_closed_rate_limit_enforcement: Option<Spelled<bool>>,
    pub token_rate_limit_type: Option<String>,
    pub disable_retry_on_max_parallel_request_limit_error: Option<Spelled<bool>>,
    pub disable_fallbacks_on_per_model_rate_limits: Option<Spelled<bool>>,
    pub user_api_key_cache_ttl: Option<Spelled<f64>>,
    pub user_api_key_cache_max_size: Option<Spelled<i64>>,
}
