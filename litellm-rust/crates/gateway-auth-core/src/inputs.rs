use std::collections::{BTreeMap, BTreeSet};

use ipnet::IpNet;

use litellm_secret_value::SecretValue;
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RequestFacts {
    pub route: String,
    pub token: Option<SecretValue>,
    pub is_llm_api_route: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanFacts {
    pub request: RequestFacts,
    pub is_info_route: bool,
    pub unverified_claims: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RoutingOverride {
    pub iss: Option<Vec<String>>,
    pub client_id: Option<Vec<String>>,
    pub scope: Option<Vec<String>>,
    pub aud: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanSettings {
    pub custom_auth: bool,
    pub premium_user: bool,
    pub public_routes: BTreeSet<String>,
    pub additional_public_routes: Vec<String>,
    pub metrics_require_auth: bool,
    pub oauth2_introspection: bool,
    pub oauth2_proxy: bool,
    pub jwt_enabled: bool,
    pub oauth2_routing_overrides: Vec<RoutingOverride>,
    pub unauthenticated_pass_through_routes: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeySettings {
    pub master_key: Option<SecretValue>,
    pub has_db: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JwtSettings {
    pub audience: Option<String>,
    pub leeway_seconds: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub struct OAuth2IntrospectionSettings {
    pub endpoint_configured: bool,
    pub requires_active: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OAuth2ProxySettings {
    pub trusted_networks: Vec<IpNet>,
    pub header_mappings: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuthorizeSettings {
    pub allow_user_auth: bool,
    pub has_db: bool,
    pub master_key_only_routes: BTreeSet<String>,
    pub proxy_max_budget: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckFacts {
    pub request: RequestFacts,
    pub force_virtual_key_checks: bool,
    pub sub_call: bool,
    pub token_has_end_user: bool,
    pub route_skips_budget_checks: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckSettings {
    pub plan: PlanSettings,
    pub auth_off_pass_through_routes: BTreeSet<String>,
    pub no_auth_dev_mode: bool,
    pub custom_auth_skips_checks: bool,
    pub has_db: bool,
}
