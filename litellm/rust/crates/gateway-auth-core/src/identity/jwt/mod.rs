//! Who a verified JWT is acting as: the caller its claims name, then the team, user and
//! organization behind it. Provisioning writes happen here, and grants are collected for `commit`.

pub(crate) mod agent;
pub(crate) mod caller;
pub mod claims;
pub(crate) mod org;
pub(crate) mod team;
pub(crate) mod user;

use std::collections::BTreeSet;

use serde_json::Value;

pub use claims::{
    MappedClaims, UserInfo, agent_claim, all_team_ids, end_user_id, is_admin, is_allowed_domain,
    is_allowed_user_role, is_enforced_email_domain, is_required_team_id,
    is_trusted_issuer_normalized, is_upsert_user_id, jwt_roles, litellm_role, nested_value,
    object_id, org_alias, org_id, rbac_role, scopes, team_alias, team_id, team_ids, user_email,
    user_id, user_info, user_roles,
};
pub use team::{
    AliasRead, HeaderTeam, Joined, ModelAccess, Team, TeamClaims, TeamRead, TeamSettings,
};
pub use user::{TeamSync, team_sync};

use crate::{
    Affiliation, Rows,
    authn::{
        jwt::{JwtSettings, MappedJwt, MappedKey},
        key::Keyed,
    },
};

/// `_JWTProvisioning`
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Provisioning {
    pub user_id_upsert: bool,
    pub team_id_upsert: bool,
}

impl From<&JwtSettings> for Provisioning {
    fn from(settings: &JwtSettings) -> Self {
        Self {
            user_id_upsert: settings.config.user_id_upsert,
            team_id_upsert: settings.config.team_id_upsert,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentMode {
    Autonomous,
    Delegated,
}

/// The `ManagedAgentContext` `resolve_managed_agent` found for the claims, with the context row
/// the core hands back to `record_agent_authentication` and in the identity.
#[derive(Clone, Debug, PartialEq)]
pub struct ManagedAgent<R> {
    pub mode: AgentMode,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub row: R,
}

/// The agent registered under a JWT's agent claim, by id and then by name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegisteredAgent {
    pub agent_id: String,
    pub identity_managed: bool,
    pub has_identity: bool,
    /// `has_legacy_identity(litellm_params)`
    pub has_legacy_identity: bool,
}

/// The arguments of `resolve_delegated_agent_team`.
#[derive(Clone, Copy, Debug)]
pub struct DelegatedTeam<'a> {
    pub user_id: Option<&'a str>,
    pub agent_id: &'a str,
    pub team_id: Option<&'a str>,
    pub explicit_team: bool,
    pub allowed_team_ids: Option<&'a BTreeSet<String>>,
}

/// What the core reads from a `LiteLLM_UserTable`, with the row it hands back to
/// `set_user_role` and `patch_team_membership` and in the identity.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct User<R> {
    pub user_id: String,
    pub user_email: Option<String>,
    pub user_role: Option<String>,
    pub teams: Vec<String>,
    /// `metadata.scim_active is False`
    pub deactivated: bool,
    pub row: R,
}

/// `get_user_object`: `Absent` is its `UserNotFoundError`.
#[derive(Clone, Debug, PartialEq)]
pub enum UserRead<R> {
    Found(User<R>),
    Absent,
}

impl<R> UserRead<R> {
    pub fn found(self) -> Option<User<R>> {
        match self {
            Self::Found(user) => Some(user),
            Self::Absent => None,
        }
    }
}

/// `GrantResolver.resolve_identity`: the user, and its membership of the team it was asked about.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentityRead<R: Rows> {
    pub user: UserRead<R::User>,
    pub membership: Option<R::Membership>,
}

/// `UserLookup`, the arguments `get_user_object` reads a user by.
#[derive(Clone, Copy, Debug)]
pub struct UserLookup<'a> {
    pub user_id: &'a str,
    pub user_email: Option<&'a str>,
    pub sso_user_id: Option<&'a str>,
    pub upsert: bool,
    pub email_verified: Option<&'a Value>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Org<R> {
    pub organization_id: String,
    pub row: R,
}

/// `JWTIdentity`
#[derive(Clone, Debug, PartialEq)]
pub struct ClaimIdentity<R> {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub user: Option<R>,
}

/// A `JWTAuthBuilderResult`: the ids the core decided, what the caller belongs to and the end user
/// the claims name. The host still builds its token from the ids in `finish_jwt`, so they stay
/// beside `affiliation` until that build moves into the core.
#[derive(Clone, Debug, PartialEq)]
pub struct JwtIdentity<R: Rows> {
    pub claims: Value,
    pub is_proxy_admin: bool,
    // TODO: `user_id`, `team_id` and `org_id` repeat ids `affiliation` holds. They go with `Rows`
    // ("Open on the types" item 4 on the Auth High-level Picture page): Python's opaque rows cannot
    // answer an id, and a provisioned user's row id can differ from the claimed one `mapping.rs`
    // reads.
    pub user_id: Option<String>,
    pub user_email: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub end_user_id: Option<String>,
    pub agent_id: Option<String>,
    pub managed_agent: Option<ManagedAgent<R::Agent>>,
    /// A JWT has no project, and a proxy admin no membership.
    pub affiliation: Affiliation<R>,
    pub end_user: Option<R::EndUser>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum JwtOutcome<R: Rows> {
    /// A proxy admin, which the host projects without finishing a token.
    Admin(JwtIdentity<R>),
    /// The token `finish_jwt` built for the JWT or the key auto-registered for it.
    Finished(R::Key),
    /// A key the JWT maps to, which still has to pass the virtual-key checks.
    MappedKey(Keyed<R::Key>, MappedJwt<R::User>),
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

/// What `_resolve_jwt_to_virtual_key` found for a token, for callers that never provision a key.
#[derive(Clone, Debug, PartialEq)]
pub enum JwtMapping<K> {
    Key(Box<MappedKey<K>>),
    /// Unmapped under `auto_register` or `link_only`, which only admission may provision.
    Pending,
    Unmapped,
}
