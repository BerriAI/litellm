//! Whether a JWT is genuine: its signature or the IdP's UserInfo, the operator's own check, and
//! the virtual key it maps to, which then takes the key path.

pub(crate) mod mapping;
mod operator;
mod userinfo;
mod verify;

use http::Method;
use jiff::Timestamp;
use litellm_config::{JwtAuthSettings, RolePermissions};

pub use mapping::{
    KeyMappingRow, KeyMappingWrite, KeyOrigin, MappedJwt, MappedKey, NO_MAPPING_SENTINEL,
    PROXY_ADMIN_SENTINEL, mapping_cache_key,
};
pub(crate) use operator::custom_validate;
pub(crate) use userinfo::decode;
pub use userinfo::oidc_userinfo;
pub use verify::{
    Claims, Expected, ID_JAG_TYP, is_jwt, managed_issuer_is_trusted, presents_claims,
    unverified_claims, verify, verify_assertion,
};

use crate::routes::{RouteGroups, passthrough::ForwardedRoutes};

/// `litellm_jwtauth` plus what the proxy reads from the environment for JWTs.
#[derive(Clone, Debug, Default)]
pub struct JwtSettings {
    pub config: JwtAuthSettings,
    /// `JWT_PUBLIC_KEY_URL`, comma-separated.
    pub public_key_urls: Vec<String>,
    /// `JWT_AUDIENCE`
    pub audience: Option<String>,
    /// `JWT_ISSUER`
    pub issuer: Option<String>,
    pub leeway_seconds: u64,
    /// `general_settings.role_permissions`
    pub role_permissions: Box<[RolePermissions]>,
    /// `general_settings.enforce_rbac`, which `validate_object_id` reads instead of `litellm_jwtauth.enforce_rbac`.
    pub enforce_rbac: bool,
    /// A database is connected (`prisma_client is not None`).
    pub has_db: bool,
    pub has_master_key: bool,
    /// `LiteLLMRoutes`, for `admin_allowed_routes`, `team_allowed_routes` and role routes that name a group.
    pub route_groups: RouteGroups,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct JwtRequest<'a> {
    pub route: &'a str,
    pub method: Option<&'a Method>,
    pub model: Option<&'a str>,
    /// The `x-litellm-team-id` header.
    pub team_header: Option<&'a str>,
    pub is_llm_api_route: bool,
    /// `RouteChecks.is_auth_enforced_pass_through_route(route, method)`
    pub auth_enforced_pass_through: bool,
    pub forwarded_routes: ForwardedRoutes<'a>,
    pub now: Timestamp,
}
