use serde::Deserialize;

use crate::{Object, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct UiSettings {
    pub ui_access_mode: Option<UiAccessMode>,
    pub custom_sso: Option<String>,
    pub custom_ui_sso_sign_in_handler: Option<String>,
    pub auto_redirect_ui_login_to_sso: Option<Spelled<bool>>,
    pub default_team_disabled: Option<Spelled<bool>>,
    pub allow_cli_sso_verification_uri_complete: Option<Spelled<bool>>,
    pub disable_env_credential_login: Option<Spelled<bool>>,
    pub disable_password_login_when_sso_enabled: Option<Spelled<bool>>,
    pub password_policy_min_length: Option<Spelled<f64>>,
    pub password_policy_require_uppercase: Option<Spelled<bool>>,
    pub password_policy_require_lowercase: Option<Spelled<bool>>,
    pub password_policy_require_numbers: Option<Spelled<bool>>,
    pub password_policy_require_special_characters: Option<Spelled<bool>>,
    pub password_policy_check_breached_passwords: Option<Spelled<bool>>,
    pub max_failed_login_attempts_per_source: Option<Spelled<i64>>,
    pub failed_login_window_seconds: Option<Spelled<i64>>,
    pub failed_login_block_seconds: Option<Spelled<i64>>,
    pub max_failed_login_attempts_per_source_overrides: Option<Object>,
    pub hide_default_credentials_hint: Option<Spelled<bool>>,
    pub disable_master_key_return: Option<Spelled<bool>>,
    pub disable_custom_api_keys: Option<Spelled<bool>>,
    pub disable_key_generate_for_org_admin: Option<Spelled<bool>>,
    pub team_admin_editable_team_fields: Option<Box<[String]>>,
    pub disable_model_add_for_internal_users: Option<Spelled<bool>>,
    pub disable_agents_for_internal_users: Option<Spelled<bool>>,
    pub allow_agents_for_team_admins: Option<Spelled<bool>>,
    pub disable_vector_stores_for_internal_users: Option<Spelled<bool>>,
    pub allow_vector_stores_for_team_admins: Option<Spelled<bool>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum UiAccessMode {
    Mode(String),
    SsoGroup(UiAccessSsoGroup),
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct UiAccessSsoGroup {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub restricted_sso_group: Option<String>,
    pub sso_group_jwt_field: Option<String>,
}
