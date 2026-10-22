use crate::{Object, Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct BudgetSettings {
    pub max_budget: Option<Spelled<f64>>,
    pub max_internal_user_budget: Option<Spelled<f64>>,
    pub default_max_internal_user_budget: Option<Spelled<f64>>,
    pub default_internal_user_params: Option<Raw>,
    pub max_ui_session_budget: Option<Spelled<f64>>,
    pub default_team_settings: Option<Box<[TeamDefaultSettings]>>,
    pub upperbound_key_generate_params: Option<UpperboundKeyGenerateParams>,
    pub key_alias_pattern: Option<String>,
    pub budget_reset_time: Option<String>,
    pub disable_end_user_cost_tracking: Option<Spelled<bool>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct TeamDefaultSettings {
    pub team_id: Option<String>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct UpperboundKeyGenerateParams {
    pub max_budget: Option<Spelled<f64>>,
    pub budget_duration: Option<String>,
    pub duration: Option<String>,
    pub max_parallel_requests: Option<Spelled<i64>>,
    pub tpm_limit: Option<Spelled<i64>>,
    pub rpm_limit: Option<Spelled<i64>>,
}
