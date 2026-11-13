mod claims;
mod error;
mod mapping;
mod policy;
mod rbac;
mod team;
mod verify;

use http::Method;
use jiff::Timestamp;
use litellm_config::{JwtAuthSettings, RolePermissions};
use serde_json::Value;

pub use claims::{
    MappedClaims, UserInfo, agent_claim, all_team_ids, claims, end_user_id, is_admin,
    is_allowed_domain, is_allowed_user_role, is_enforced_email_domain, is_required_team_id,
    is_trusted_issuer_normalized, is_upsert_user_id, jwt_roles, litellm_role, nested_value,
    object_id, org_alias, org_id, rbac_role, scopes, team_alias, team_id, team_ids, user_email,
    user_id, user_info, user_roles,
};
pub use error::{
    AgentIdentityFailure, AssertionRejection, ClaimDenial, JwtDenial, KeyError, KeyFailure,
    ModelAccessDenied, RbacDenial, Searched, TeamDenial, TokenError, VerifyDenial,
};
pub use mapping::{
    KeyMappingRow, KeyMappingWrite, KeyOrigin, MappedJwt, MappedKey, NO_MAPPING_SENTINEL,
    PROXY_ADMIN_SENTINEL, mapping_cache_key,
};
pub use policy::{
    AgentMode, ClaimIdentity, DelegatedTeam, IdentityRead, ManagedAgent, Org, Provisioning,
    RegisteredAgent, TeamSync, User, UserLookup, UserRead, team_sync,
};
pub use rbac::{
    TeamRouteAllowed, TeamRouteListed, check_admin_route, check_rbac_role,
    check_scope_based_access, is_team_route_allowed, team_route_listed,
};
pub use team::{
    AliasRead, HeaderTeam, Joined, ModelAccess, Team, TeamClaims, TeamRead, TeamSettings,
    admin_header_team, check_provisional_header_team, db_team_fallback, header_team,
    load_header_team, passthrough_gate, select_team, single_team_fallback, specific_team,
};
pub use verify::{
    Claims, Expected, ID_JAG_TYP, is_jwt, managed_issuer_is_trusted, oidc_userinfo,
    presents_claims, unverified_claims, verify, verify_assertion,
};

use crate::{
    AuthHost, Error, Rows,
    credential::key::Token,
    host::{KeyRow, UserRow},
    routes::{RouteGroups, passthrough::ForwardedRoutes},
};

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

/// A `JWTAuthBuilderResult`: the ids the core decided and the rows it loaded them from.
#[derive(Clone, Debug, PartialEq)]
pub struct JwtIdentity<R: Rows> {
    pub claims: Value,
    pub is_proxy_admin: bool,
    pub user_id: Option<String>,
    pub user_email: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub end_user_id: Option<String>,
    pub agent_id: Option<String>,
    pub managed_agent: Option<ManagedAgent<R::Agent>>,
    pub rows: JwtRows<R>,
}

/// The rows behind a `JwtIdentity`'s ids. The membership is the user's in `team`, and none for a
/// proxy admin.
#[derive(Clone, Debug, PartialEq)]
pub struct JwtRows<R: Rows> {
    pub user: Option<R::User>,
    pub team: Option<R::Team>,
    pub membership: Option<R::Membership>,
    pub org: Option<R::Org>,
    pub end_user: Option<R::EndUser>,
}

impl<R: Rows> Default for JwtRows<R> {
    fn default() -> Self {
        Self {
            user: None,
            team: None,
            membership: None,
            org: None,
            end_user: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum JwtOutcome<R: Rows> {
    /// A proxy admin, which the host projects without finishing a token.
    Admin(JwtIdentity<R>),
    /// The token `finish_jwt` built for the JWT or the key auto-registered for it.
    Finished(R::Key),
    /// A key the JWT maps to, which still has to pass the virtual-key checks.
    MappedKey(crate::credential::key::Keyed<R::Key>, MappedJwt<R::User>),
}

/// What the host builds its token from in `finish_jwt`.
#[derive(Debug)]
pub enum FinishedJwt<'a, R: Rows> {
    Identity(&'a JwtIdentity<R>),
    /// A key auto-registered for the JWT that stands in for it without the virtual-key checks,
    /// as the host's `key_by_hash` loaded it.
    AutoRegisteredKey(&'a MappedJwt<R::User>, &'a R::Key),
}

impl<R: Rows> Clone for FinishedJwt<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Rows> Copy for FinishedJwt<'_, R> {}

/// The builder's JWT block: virtual-key mapping, then `JWTAuthManager.auth_builder`, then any
/// deferred auto-registration.
pub async fn authenticate<H: AuthHost>(
    host: &H,
    token: &Token,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
) -> Result<JwtOutcome<H::Rows>, Error> {
    policy::build(host, token, request, settings).await
}

/// `JWTAuthManager.authorize_jwt`. `auth_builder` passes the configured provisioning; the MCP
/// token flows pass `None`.
pub async fn authorize_jwt<H: AuthHost>(
    host: &H,
    token: &Token,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
    provisioning: Option<Provisioning>,
) -> Result<JwtIdentity<H::Rows>, Error> {
    let claims = policy::authenticate_token(host, token, settings, request.now).await?;
    let managed = host.managed_agent(&Value::Object(claims.0.clone())).await?;
    let authorized =
        policy::authorize(host, &claims, managed, request, settings, provisioning).await?;
    Ok(authorized.identity)
}

/// What `_resolve_jwt_to_virtual_key` found for a token, for callers that never provision a key.
#[derive(Clone, Debug, PartialEq)]
pub enum JwtMapping<K> {
    Key(Box<MappedKey<K>>),
    /// Unmapped under `auto_register` or `link_only`, which only admission may provision.
    Pending,
    Unmapped,
}

/// `JWTAuthManager.authenticate_jwt`, then `_resolve_jwt_to_virtual_key` without provisioning.
pub async fn resolve_mapping<H: AuthHost>(
    host: &H,
    token: &Token,
    now: Timestamp,
    settings: &JwtSettings,
) -> Result<JwtMapping<KeyRow<H>>, Error> {
    let claims = policy::authenticate_token(host, token, settings, now).await?;
    Ok(match mapping::resolve(host, &claims, settings).await? {
        mapping::Resolved::Key(key) => JwtMapping::Key(Box::new(key)),
        mapping::Resolved::Pending(_) => JwtMapping::Pending,
        mapping::Resolved::Unmapped => JwtMapping::Unmapped,
    })
}

/// `JWTAuthManager.resolve_identity`
pub async fn resolve_identity<H: AuthHost>(
    host: &H,
    token: &Token,
    now: Timestamp,
    settings: &JwtSettings,
) -> Result<ClaimIdentity<UserRow<H>>, Error> {
    let claims = policy::authenticate_token(host, token, settings, now).await?;
    let (identity, _) = policy::claim_identity(host, &claims, settings).await?;
    Ok(identity)
}
