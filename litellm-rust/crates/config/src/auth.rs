use serde::Deserialize;

use crate::AdditionalFields;

/// A config value YAML may give as one string or a list of them.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Box<[String]>),
}

impl OneOrMany {
    pub fn values(&self) -> &[String] {
        match self {
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => values,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, strum::IntoStaticStr)]
pub enum RbacRole {
    #[serde(rename = "proxy_admin")]
    #[strum(serialize = "proxy_admin")]
    ProxyAdmin,
    #[serde(rename = "team")]
    #[strum(serialize = "team")]
    Team,
    #[serde(rename = "internal_user")]
    #[strum(serialize = "internal_user")]
    InternalUser,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, strum::IntoStaticStr, strum::EnumString)]
pub enum LitellmRole {
    #[serde(rename = "proxy_admin")]
    #[strum(serialize = "proxy_admin")]
    ProxyAdmin,
    #[serde(rename = "proxy_admin_viewer")]
    #[strum(serialize = "proxy_admin_viewer")]
    ProxyAdminViewOnly,
    #[serde(rename = "org_admin")]
    #[strum(serialize = "org_admin")]
    OrgAdmin,
    #[serde(rename = "internal_user")]
    #[strum(serialize = "internal_user")]
    InternalUser,
    #[serde(rename = "internal_user_viewer")]
    #[strum(serialize = "internal_user_viewer")]
    InternalUserViewOnly,
    #[serde(rename = "team")]
    #[strum(serialize = "team")]
    Team,
    #[serde(rename = "customer")]
    #[strum(serialize = "customer")]
    Customer,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, strum::IntoStaticStr)]
pub enum UnregisteredJwtClientBehavior {
    #[default]
    #[serde(rename = "fallback_team_mapping")]
    #[strum(serialize = "fallback_team_mapping")]
    FallbackTeamMapping,
    #[serde(rename = "reject")]
    #[strum(serialize = "reject")]
    Reject,
    #[serde(rename = "auto_register")]
    #[strum(serialize = "auto_register")]
    AutoRegister,
    #[serde(rename = "link_only")]
    #[strum(serialize = "link_only")]
    LinkOnly,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RoleMapping {
    pub role: String,
    pub internal_role: RbacRole,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct JwtLitellmRoleMap {
    pub jwt_role: String,
    pub litellm_role: LitellmRole,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScopeMapping {
    pub scope: String,
    #[serde(default)]
    pub models: Option<Box<[String]>>,
    #[serde(default)]
    pub routes: Option<Box<[String]>>,
}

/// One `general_settings.role_permissions` entry (`RoleBasedPermissions`).
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RolePermissions {
    pub role: RbacRole,
    #[serde(default)]
    pub models: Option<Box<[String]>>,
    #[serde(default)]
    pub routes: Option<Box<[String]>>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
pub enum RoutingPath {
    #[default]
    #[serde(rename = "oauth2")]
    OAuth2,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JwtRoutingOverride {
    pub iss: OneOrMany,
    #[serde(default)]
    pub client_id: Option<OneOrMany>,
    #[serde(default)]
    pub scope: Option<OneOrMany>,
    #[serde(default)]
    pub aud: Option<OneOrMany>,
    #[serde(default)]
    pub path: RoutingPath,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct JwtIssuerConfig {
    pub issuer: String,
    pub jwks_url: Option<String>,
    pub audience: Option<OneOrMany>,
    pub disable_audience_validation: bool,
    pub user_id_jwt_field: Option<String>,
    pub user_email_jwt_field: Option<String>,
    pub team_id_jwt_field: Option<String>,
    pub team_ids_jwt_field: Option<String>,
    pub org_id_jwt_field: Option<String>,
    pub end_user_id_jwt_field: Option<String>,
    pub virtual_key_claim_field: Option<String>,
    pub unregistered_jwt_client_behavior: Option<UnregisteredJwtClientBehavior>,
}

/// `general_settings.litellm_jwtauth`, with `LiteLLM_JWTAuth`'s defaults. `custom_validate` is
/// the operator's import path: only the host can run it, so the core only learns that it is set.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct JwtAuthSettings {
    pub admin_jwt_scope: String,
    pub admin_allowed_routes: Box<[String]>,
    pub team_id_jwt_field: Option<String>,
    pub team_id_upsert: bool,
    pub team_ids_jwt_field: Option<String>,
    pub upsert_sso_user_to_team: bool,
    pub team_allowed_routes: Box<[String]>,
    pub team_id_default: Option<String>,
    pub team_alias_jwt_field: Option<String>,
    pub org_id_jwt_field: Option<String>,
    pub org_alias_jwt_field: Option<String>,
    pub user_id_jwt_field: Option<String>,
    pub user_email_jwt_field: Option<String>,
    pub user_allowed_email_domain: Option<String>,
    pub user_roles_jwt_field: Option<String>,
    pub user_allowed_roles: Option<Box<[String]>>,
    pub user_id_upsert: bool,
    pub end_user_id_jwt_field: Option<String>,
    pub agent_id_jwt_field: Option<String>,
    pub mcp_client_id_jwt_field: Option<String>,
    pub public_key_ttl: f64,
    pub public_key_stale_ttl: f64,
    pub public_allowed_routes: Box<[String]>,
    pub enforce_rbac: bool,
    pub roles_jwt_field: Option<String>,
    pub role_mappings: Option<Box<[RoleMapping]>>,
    pub object_id_jwt_field: Option<String>,
    pub scope_mappings: Option<Box<[ScopeMapping]>>,
    pub enforce_scope_based_access: bool,
    pub enforce_team_based_model_access: bool,
    pub custom_validate: Option<String>,
    pub jwt_litellm_role_map: Option<Box<[JwtLitellmRoleMap]>>,
    pub sync_user_role_and_teams: bool,
    pub oidc_userinfo_endpoint: Option<String>,
    pub oidc_userinfo_enabled: bool,
    pub oidc_userinfo_cache_ttl: f64,
    #[serde(alias = "jwt_client_id_field")]
    pub virtual_key_claim_field: Option<String>,
    pub virtual_key_mapping_cache_ttl: f64,
    pub unregistered_jwt_client_behavior: UnregisteredJwtClientBehavior,
    pub auto_register_map_existing_key: bool,
    pub routing_overrides: Option<Box<[JwtRoutingOverride]>>,
    pub team_claim_fallback: bool,
    pub fallback_to_db_teams: bool,
    pub issuers: Option<Box<[JwtIssuerConfig]>>,
    #[serde(flatten)]
    pub additional_fields: AdditionalFields,
}

fn owned(values: &[&str]) -> Box<[String]> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

impl Default for JwtAuthSettings {
    fn default() -> Self {
        Self {
            admin_jwt_scope: "litellm_proxy_admin".to_owned(),
            admin_allowed_routes: owned(&[
                "management_routes",
                "spend_tracking_routes",
                "global_spend_tracking_routes",
                "info_routes",
            ]),
            team_id_jwt_field: None,
            team_id_upsert: false,
            team_ids_jwt_field: None,
            upsert_sso_user_to_team: false,
            team_allowed_routes: owned(&[
                "openai_routes",
                "info_routes",
                "mcp_routes",
                "/v1/messages",
                "/v1/messages/count_tokens",
                "/rag/ingest",
                "/v1/rag/ingest",
                "/rag/query",
                "/v1/rag/query",
            ]),
            team_id_default: None,
            team_alias_jwt_field: None,
            org_id_jwt_field: None,
            org_alias_jwt_field: None,
            user_id_jwt_field: None,
            user_email_jwt_field: None,
            user_allowed_email_domain: None,
            user_roles_jwt_field: None,
            user_allowed_roles: None,
            user_id_upsert: false,
            end_user_id_jwt_field: None,
            agent_id_jwt_field: None,
            mcp_client_id_jwt_field: None,
            public_key_ttl: 600.0,
            public_key_stale_ttl: 3600.0,
            public_allowed_routes: owned(&["public_routes"]),
            enforce_rbac: false,
            roles_jwt_field: None,
            role_mappings: None,
            object_id_jwt_field: None,
            scope_mappings: None,
            enforce_scope_based_access: false,
            enforce_team_based_model_access: false,
            custom_validate: None,
            jwt_litellm_role_map: None,
            sync_user_role_and_teams: false,
            oidc_userinfo_endpoint: None,
            oidc_userinfo_enabled: false,
            oidc_userinfo_cache_ttl: 300.0,
            virtual_key_claim_field: None,
            virtual_key_mapping_cache_ttl: 300.0,
            unregistered_jwt_client_behavior: UnregisteredJwtClientBehavior::FallbackTeamMapping,
            auto_register_map_existing_key: false,
            routing_overrides: None,
            team_claim_fallback: false,
            fallback_to_db_teams: false,
            issuers: None,
            additional_fields: AdditionalFields::default(),
        }
    }
}

impl JwtAuthSettings {
    pub fn issuer(&self, issuer: Option<&str>) -> Option<&JwtIssuerConfig> {
        let issuer = issuer?;
        self.issuers.as_deref()?.iter().find(|config| config.issuer == issuer)
    }

    pub fn is_virtual_key_mapping_configured(&self) -> bool {
        self.virtual_key_claim_field.is_some()
            || self
                .issuers
                .as_deref()
                .unwrap_or_default()
                .iter()
                .any(|config| config.virtual_key_claim_field.is_some())
    }

    pub fn virtual_key_claim_field(&self, issuer: Option<&str>) -> Option<&str> {
        self.issuer(issuer)
            .and_then(|config| config.virtual_key_claim_field.as_deref())
            .or(self.virtual_key_claim_field.as_deref())
    }

    pub fn is_user_identity_claim(&self, claim_field: &str, issuer: Option<&str>) -> bool {
        let (user_id, user_email) = match self.issuer(issuer) {
            None => (self.user_id_jwt_field.as_deref(), self.user_email_jwt_field.as_deref()),
            Some(config) => (
                config.user_id_jwt_field.as_deref().or(self.user_id_jwt_field.as_deref()),
                config.user_email_jwt_field.as_deref().or(self.user_email_jwt_field.as_deref()),
            ),
        };
        user_id == Some(claim_field) || user_email == Some(claim_field)
    }

    pub fn unregistered_jwt_client_behavior(&self, issuer: Option<&str>) -> UnregisteredJwtClientBehavior {
        self.issuer(issuer)
            .and_then(|config| config.unregistered_jwt_client_behavior)
            .unwrap_or(self.unregistered_jwt_client_behavior)
    }
}
