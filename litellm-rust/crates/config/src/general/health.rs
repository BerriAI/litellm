use litellm_auth_types::SecretValue;

use crate::Spelled;

#[macro_rules_attribute::apply(crate::config_section)]
pub struct HealthSettings {
    pub use_shared_health_check: Option<Spelled<bool>>,
    pub health_check_details: Option<Spelled<bool>>,
    pub enable_health_check_routing: Option<Spelled<bool>>,
    pub health_check_staleness_threshold: Option<Spelled<f64>>,
    pub health_check_ignore_transient_errors: Option<Spelled<bool>>,
    pub health_check_skip_disabled_background_models: Option<Spelled<bool>>,
    pub background_health_check_model_groups: Option<Box<[String]>>,
    pub model_list_healthy_only: Option<Spelled<bool>>,
    pub allow_public_health_readiness_details: Option<Spelled<bool>>,
    pub enable_drain_endpoint: Option<Spelled<bool>>,
    pub drain_endpoint_token: Option<SecretValue>,
    pub disable_model_info_refresh: Option<Spelled<bool>>,
    pub use_team_public_model_name: Option<Spelled<bool>>,
    pub image_generation_model: Option<String>,
    pub moderation_model: Option<String>,
}
