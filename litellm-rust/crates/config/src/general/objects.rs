use crate::{Object, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct ObjectsSettings {
    pub custom_key_generate: Option<String>,
    pub custom_key_update: Option<String>,
    pub custom_key_policy: Option<String>,
    pub custom_team_metadata_validate: Option<String>,
    pub team_metadata_schema: Option<Box<[TeamMetadataField]>>,
    pub team_metadata_validation_timeout: Option<Spelled<f64>>,
    pub team_metadata_validation_error_message: Option<String>,
    pub disable_auto_add_proxy_admin_to_teams: Option<Spelled<bool>>,
    pub self_serve_budget_policy: Option<String>,
    pub enforce_rpm_tpm_on_model_add: Option<Spelled<bool>>,
    pub agent_access_default_deny: Option<Spelled<bool>>,
    pub vector_store_deny_by_default: Option<Spelled<bool>>,
    pub search_tool_deny_by_default: Option<Spelled<bool>>,
    pub search_tools: Option<Box<[Object]>>,
    pub enforce_fallback_model_access: Option<Spelled<bool>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct TeamMetadataField {
    pub key: Option<String>,
    pub label: Option<String>,
}
