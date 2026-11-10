use std::collections::BTreeSet;

use litellm_secret_value::SecretValue;
use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RequestFacts {
    pub route: String,
    pub token: Option<SecretValue>,
    pub is_llm_api_route: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanSettings {
    pub custom_auth: bool,
    pub jwt_enabled: bool,
    pub oauth2_introspection: bool,
    pub oauth2_proxy: bool,
    pub public_routes: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeySettings {
    pub master_key: Option<SecretValue>,
    pub allow_user_auth: bool,
    pub has_db: bool,
    pub master_key_only_routes: BTreeSet<String>,
    pub proxy_max_budget: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JwtSettings {
    pub audience: Option<String>,
    pub leeway_seconds: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct OAuth2Settings {
    pub user_id_header: Option<String>,
    pub team_id_header: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuthorizeSettings {
    pub admin_only_routes: BTreeSet<String>,
    pub proxy_max_budget: f64,
}
