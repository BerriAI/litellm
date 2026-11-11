use std::{fmt::Debug, future::Future, time::Duration};

use litellm_config::LitellmRole;
use serde_json::Value;

use crate::{
    Error,
    authorize::Membership,
    budgets::{BudgetAlert, MemberBudget, NamedBudget, RecordedBudget, SpendCounter},
    key_budget::KeyBudgetAlert,
    jwt::{
        AgentIdentityFailure, AliasRead, DelegatedTeam, FinishedJwt, IdentityRead,
        KeyMappingRow, KeyMappingWrite, ManagedAgent, MappedKey, ModelAccess, Org,
        RegisteredAgent, TeamRead, UserLookup, UserRead,
    },
    key::{CachedTeam, Keyed, Owner},
};

pub trait Row: Clone + Debug + PartialEq + Send + Sync + 'static {}

impl<T: Clone + Debug + PartialEq + Send + Sync + 'static> Row for T {}

/// The host's own records. The core decides on the facts a host call returns beside a row, never
/// reads the row, and hands it back in the identity for the host to convert.
pub trait Rows: Row {
    type Key: Row;
    type User: Row;
    type Team: Row;
    type EndUser: Row;
    type ObjectPermission: Row;
    type Project: Row;
    type Membership: Row;
    type Org: Row;
    /// A managed agent's context.
    type Agent: Row;
}

pub type KeyRow<H> = <<H as AuthHost>::Rows as Rows>::Key;
pub type UserRow<H> = <<H as AuthHost>::Rows as Rows>::User;
pub type TeamRow<H> = <<H as AuthHost>::Rows as Rows>::Team;
pub type EndUserRow<H> = <<H as AuthHost>::Rows as Rows>::EndUser;
pub type ObjectPermissionRow<H> = <<H as AuthHost>::Rows as Rows>::ObjectPermission;
pub type ProjectRow<H> = <<H as AuthHost>::Rows as Rows>::Project;
pub type MembershipRow<H> = <<H as AuthHost>::Rows as Rows>::Membership;
pub type OrgRow<H> = <<H as AuthHost>::Rows as Rows>::Org;
pub type AgentRow<H> = <<H as AuthHost>::Rows as Rows>::Agent;

/// Everything the core reads, writes or hands off, grouped by what the host is allowed to do.
/// Reads and writes come first and never decide. The last three sections are where the host still
/// runs Python: the one permanent hook, then the decisions that leave once their TODO lands.
pub trait AuthHost: Send + Sync {
    type Rows: Rows;

    // Key path reads. Each returns the row with the facts the core decides on.

    /// The key cached under `hash`.
    fn cached_key(
        &self,
        hash: &str,
    ) -> impl Future<Output = Result<Option<Keyed<KeyRow<Self>>>, Error>> + Send;
    /// The UI session key `token` encodes.
    fn ui_key(
        &self,
        token: &str,
    ) -> impl Future<Output = Result<Option<Keyed<KeyRow<Self>>>, Error>> + Send;
    /// The key's database row, stamped with the end user's limits. `None` when no key has the hash.
    fn db_key(
        &self,
        hash: &str,
        end_user: &EndUserRow<Self>,
    ) -> impl Future<Output = Result<Option<Keyed<KeyRow<Self>>>, Error>> + Send;
    /// The key's owner and whether SCIM deactivated them.
    fn key_owner(
        &self,
        user_id: &str,
    ) -> impl Future<Output = Result<Option<Owner<UserRow<Self>>>, Error>> + Send;
    /// The cached team and when it was last refreshed, for the core to compare with the key's copy.
    fn cached_team(
        &self,
        team_id: &str,
    ) -> impl Future<Output = Result<Option<CachedTeam<TeamRow<Self>>>, Error>> + Send;
    /// `get_team_object`, `None` when it reports no team.
    fn key_team(
        &self,
        team_id: &str,
    ) -> impl Future<Output = Result<Option<TeamRow<Self>>, Error>> + Send;
    fn object_permission(
        &self,
        id: &str,
    ) -> impl Future<Output = Result<Option<ObjectPermissionRow<Self>>, Error>> + Send;
    fn project(
        &self,
        project_id: &str,
    ) -> impl Future<Output = Result<Option<ProjectRow<Self>>, Error>> + Send;

    // Key path rows the host rebuilds from rows the core hands it.

    /// The session key with the grants of its user's current rows: the team's grants and the
    /// membership's, the user's role, and the user's own models when there is no team.
    fn with_session_grants(
        &self,
        key: &KeyRow<Self>,
        user: &UserRow<Self>,
        team: Option<&TeamRow<Self>>,
        membership: Option<&MembershipRow<Self>>,
    ) -> impl Future<Output = Result<Keyed<KeyRow<Self>>, Error>> + Send;
    /// The key with the `team_*` values of a fresher cached team.
    fn with_team(
        &self,
        key: &KeyRow<Self>,
        team: &TeamRow<Self>,
    ) -> impl Future<Output = Result<KeyRow<Self>, Error>> + Send;
    // TODO: rewritten by the host until the router's model aliases are readable from router-core
    /// Rewrites the request's model to the one the key's aliases and the router resolve it to, for
    /// the checks after it.
    fn normalize_model(&self, key: &KeyRow<Self>) -> impl Future<Output = Result<(), Error>> + Send;
    /// Warms the cache with the rows the key and the end user reference, which the checks after it read.
    fn prefetch_auth_objects(
        &self,
        key: &KeyRow<Self>,
        end_user: &EndUserRow<Self>,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    /// Drops the cached key, after the core found it stale or unusable.
    fn evict_key(&self, hash: &str) -> impl Future<Output = Result<(), Error>> + Send;

    // JWT path IO. The core picks the URL and the cache key, and owns the TTLs and stale windows.

    /// GET with the transient-failure retries. `None` once transport failures outlast them; a
    /// non-200 or non-JSON reply is a host error carrying the proxy's message.
    fn fetch_json(&self, url: &str) -> impl Future<Output = Result<Option<Value>, Error>> + Send;
    /// GET the OIDC UserInfo endpoint with `Authorization: Bearer {token}`. `None` is a JSON
    /// `null` body; any failure is a host error carrying `Failed to fetch OIDC UserInfo: ...`.
    fn fetch_userinfo(
        &self,
        endpoint: &str,
        token: &str,
    ) -> impl Future<Output = Result<Option<Value>, Error>> + Send;
    /// `user_api_key_cache.async_get_cache`
    fn cached_json(&self, key: &str) -> impl Future<Output = Result<Option<Value>, Error>> + Send;
    /// `user_api_key_cache.async_set_cache`
    fn cache_json(
        &self,
        key: &str,
        value: &Value,
        ttl: Duration,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    /// `user_api_key_cache.async_delete_cache`
    fn forget_json(&self, key: &str) -> impl Future<Output = Result<(), Error>> + Send;

    // JWT path reads: the teams, user, org and end user a token names.

    /// `get_team_object`, with the team's facts and its pass-through grants beside the row.
    fn team(
        &self,
        team_id: &str,
        upsert: bool,
    ) -> impl Future<Output = Result<TeamRead<TeamRow<Self>>, Error>> + Send;
    /// `get_team_object_by_alias`
    fn team_by_alias(
        &self,
        alias: &str,
    ) -> impl Future<Output = Result<AliasRead<TeamRow<Self>>, Error>> + Send;
    /// `get_team_object(team_id, check_db_only=True)` for the team a delegated agent's grant selected.
    fn granting_team(
        &self,
        team_id: &str,
    ) -> impl Future<Output = Result<TeamRead<TeamRow<Self>>, Error>> + Send;
    /// `get_team_membership`
    fn load_team_membership(
        &self,
        user_id: &str,
        team_id: &str,
    ) -> impl Future<Output = Result<Option<MembershipRow<Self>>, Error>> + Send;
    /// `get_user_object`
    fn user(
        &self,
        lookup: &UserLookup<'_>,
    ) -> impl Future<Output = Result<UserRead<UserRow<Self>>, Error>> + Send;
    /// `GrantResolver.resolve_identity`: the user and its membership of `team_id`.
    fn resolve_user_identity(
        &self,
        lookup: &UserLookup<'_>,
        team_id: Option<&str>,
    ) -> impl Future<Output = Result<IdentityRead<Self::Rows>, Error>> + Send;
    fn org(
        &self,
        org_id: &str,
    ) -> impl Future<Output = Result<Option<Org<OrgRow<Self>>>, Error>> + Send;
    fn org_by_alias(
        &self,
        alias: &str,
    ) -> impl Future<Output = Result<Option<Org<OrgRow<Self>>>, Error>> + Send;
    /// `get_end_user_object` for the JWT's end user.
    fn jwt_end_user(
        &self,
        end_user_id: &str,
    ) -> impl Future<Output = Result<EndUserRow<Self>, Error>> + Send;

    // JWT path writes the core decided on: role and team sync. Each returns the user row with
    // the change applied.

    /// `UserRepository.update` of the role, then the user re-cached with it.
    fn set_user_role(
        &self,
        user: &UserRow<Self>,
        role: LitellmRole,
    ) -> impl Future<Output = Result<UserRow<Self>, Error>> + Send;
    /// `patch_team_membership`, then the user re-cached with its new teams.
    fn patch_team_membership(
        &self,
        user: &UserRow<Self>,
        add: &[String],
        remove: &[String],
    ) -> impl Future<Output = Result<UserRow<Self>, Error>> + Send;
    /// `team_member_add` as a `user`, an already-a-member conflict included.
    fn add_team_member(
        &self,
        user_id: &str,
        team_id: &str,
    ) -> impl Future<Output = Result<(), Error>> + Send;

    // JWT to virtual key mappings. The core decides whether to map, reuse or mint.

    /// `get_jwt_key_mapping_object` for one scope: the issuer's, or the global one for `None`.
    fn key_mapping_hash(
        &self,
        claim_field: &str,
        claim_value: &str,
        issuer: Option<&str>,
    ) -> impl Future<Output = Result<Option<String>, Error>> + Send;
    /// `IdentityStore.resolve(hashed_token)`
    fn key_by_hash(
        &self,
        hash: &str,
    ) -> impl Future<Output = Result<Option<MappedKey<KeyRow<Self>>>, Error>> + Send;
    /// `_reusable_key_hash_for_user`
    fn reusable_key_hash(
        &self,
        user_id: &str,
        team_id: Option<&str>,
    ) -> impl Future<Output = Result<Option<String>, Error>> + Send;
    // TODO: written by the host until key generation and the mapping insert move to Rust
    /// Mints a key when the row names none, then writes the mapping with its race handling.
    fn create_key_mapping(
        &self,
        row: &KeyMappingRow<'_>,
    ) -> impl Future<Output = Result<KeyMappingWrite, Error>> + Send;

    // Managed and registered agents.

    /// `resolve_managed_agent`
    fn managed_agent(
        &self,
        claims: &Value,
    ) -> impl Future<Output = Result<Option<ManagedAgent<AgentRow<Self>>>, Error>> + Send;
    /// The registered agent a JWT's agent claim names.
    fn registered_agent(
        &self,
        claim: &str,
    ) -> impl Future<Output = Result<Option<RegisteredAgent>, Error>> + Send;
    /// `AgentIdentityStore.record_authentication`
    fn record_agent_authentication(
        &self,
        agent: &AgentRow<Self>,
    ) -> impl Future<Output = Result<Option<AgentIdentityFailure>, Error>> + Send;

    // Authorization reads: what `common_checks` reads from Python-owned state for the request it
    // authorizes.

    /// The request's models that have no pricing in the cost map.
    fn unpriced_models(&self) -> impl Future<Output = Result<Vec<String>, Error>> + Send;
    /// Whether the registered agent sets `require_trace_id_on_calls_by_agent`.
    fn agent_requires_trace_id(
        &self,
        agent_id: &str,
    ) -> impl Future<Output = Result<bool, Error>> + Send;
    /// Whether the request carries a trace id: in its headers, or with `headers_only` unset, also in
    /// its body or a W3C `traceparent`.
    fn has_trace_id(&self, headers_only: bool) -> impl Future<Output = Result<bool, Error>> + Send;
    /// The organization of the team a `/team/update` request names, `None` when the team does not load.
    fn team_organization(
        &self,
        team_id: &str,
    ) -> impl Future<Output = Result<Option<String>, Error>> + Send;
    /// Merges the caller's and the key's tags into the request body, where the tag budget reads them.
    fn merge_budget_tags(&self) -> impl Future<Output = Result<(), Error>> + Send;

    // Budget reads and alerts. The spend counters and alerting are host-owned, so the core reads
    // spend and limits here, compares them itself, and asks the host to alert.

    /// The live spend of `counter`.
    fn spend(&self, counter: SpendCounter<'_>) -> impl Future<Output = Result<f64, Error>> + Send;
    /// The organization's budget, `None` when it does not load or sets no ceiling. Reading it also
    /// carries the organization's budget onto the request's key.
    fn organization_budget(
        &self,
        org_id: &str,
    ) -> impl Future<Output = Result<Option<RecordedBudget>, Error>> + Send;
    /// The budgeted tags of the request body, after `merge_budget_tags`, in the body's order.
    fn tag_budgets(&self) -> impl Future<Output = Result<Vec<NamedBudget>, Error>> + Send;
    /// The budgets of `groups` that have a ceiling, in the order of `groups`.
    fn model_access_group_budgets(
        &self,
        groups: &[String],
    ) -> impl Future<Output = Result<Vec<NamedBudget>, Error>> + Send;
    /// The team member's budget, from their membership row and the team's default member budget.
    fn team_member_budget_state(
        &self,
        membership: Option<&MembershipRow<Self>>,
    ) -> impl Future<Output = Result<MemberBudget, Error>> + Send;
    fn budget_alert(&self, alert: BudgetAlert<'_>) -> impl Future<Output = Result<(), Error>> + Send;

    // The permanent host step. It runs operator code, so there is no rule to port.

    /// The operator's `custom_validate` callable. The core decides when it runs and what `false` means.
    fn custom_validate(&self, claims: &Value) -> impl Future<Output = Result<bool, Error>> + Send;

    // Budgets the host still decides, because the spend counters, alerts and budget fallback rewrite
    // are Python-owned. Each returns the host's verdict, and the core renders the denial.

    // TODO: decided by the host until end-user budgets move to Rust
    /// The request's end user and the limits it carries. `resolved` is a key the caller already
    /// holds, whose `end_user_id` wins over the request's.
    fn end_user(
        &self,
        token: &str,
        resolved: Option<&KeyRow<Self>>,
    ) -> impl Future<Output = Result<EndUserRow<Self>, Error>> + Send;
    // TODO: decided by the host until key budgets move to Rust
    /// The key with an unexpired temporary budget increase added to its budget.
    fn with_temp_budget(
        &self,
        key: &KeyRow<Self>,
    ) -> impl Future<Output = Result<KeyRow<Self>, Error>> + Send;
    // TODO: decided by the host until request pricing is readable from the router
    /// Whether the request is free, so no budget applies to the key.
    fn key_skips_budget_checks(
        &self,
        key: &KeyRow<Self>,
    ) -> impl Future<Output = Result<bool, Error>> + Send;
    // TODO: decided by the host until budget rows and their temporary increases are readable from Rust
    /// The budget the key owner's membership in the team sets, with a temporary increase applied.
    fn team_member_max_budget(
        &self,
        user_id: &str,
        team_id: &str,
    ) -> impl Future<Output = Result<Option<f64>, Error>> + Send;
    // TODO: decided by the host until the model budget fallback and per-model spend move to Rust
    /// The key's and its owner's per-model budgets, falling back to another model when the router
    /// configures one.
    fn key_model_budgets(
        &self,
        key: &KeyRow<Self>,
        owner: Option<&UserRow<Self>>,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    /// Throttles an over-budget key to `share` of its limits for this request instead of refusing it.
    fn throttle_key(
        &self,
        key: &KeyRow<Self>,
        share: f64,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    /// The proxy's global spend, `None` when it could not be read.
    fn global_proxy_spend(&self) -> impl Future<Output = Result<Option<f64>, Error>> + Send;
    /// Sends a key path budget alert for the key and its owner.
    fn key_budget_alert(
        &self,
        key: &KeyRow<Self>,
        owner: Option<&UserRow<Self>>,
        alert: KeyBudgetAlert,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: decided by the host until the user per-model budget moves to Rust and the project read
    // uses `project`
    /// Builds the request's token from the identity and its rows, then runs the JWT path's project
    /// lookup and per-model user budget check on it, and returns the token.
    fn finish_jwt(
        &self,
        finished: FinishedJwt<'_, Self::Rows>,
    ) -> impl Future<Output = Result<KeyRow<Self>, Error>> + Send;

    // `common_checks` decisions the host still makes. Each raises the proxy's own error.

    // TODO: decided by the host until access groups and model aliases are readable from router-core
    /// `can_team_access_model`, with the key's access groups as a fallback, then the member's model scope.
    fn check_team_model_access(
        &self,
        membership: Membership<'_, MembershipRow<Self>>,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: decided by the host until access groups and model aliases are readable from router-core
    /// The agent's, the personal key owner's, the end user's and the personal key's teams' model grants.
    fn check_model_grants(&self) -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: decided by the host until project budgets and model access move to Rust
    /// The project's blocked flag, model access and budgets.
    fn check_project(&self) -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: decided by the host until access groups are readable from router-core
    /// Stamps the access groups that authorized the request onto its token and returns them.
    fn stamp_access_groups(
        &self,
        membership: Membership<'_, MembershipRow<Self>>,
    ) -> impl Future<Output = Result<Vec<String>, Error>> + Send;
    // TODO: decided by the host until guardrail settings are readable from Rust
    /// `guardrail_modification_check`
    fn check_guardrail_modification(&self) -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: decided by the host until vector store and tool grants are readable from Rust
    /// Vector store access and the tool allowlist.
    fn check_object_access(&self) -> impl Future<Output = Result<(), Error>> + Send;

    // Other decisions the host still answers. Each takes the rows it decides on as arguments, and
    // leaves together with its bridge step and its `drive` case once its TODO lands.

    // TODO: answered by the host until access groups and model aliases are readable from router-core
    fn team_model_access(
        &self,
        team: &TeamRow<Self>,
        model: &str,
    ) -> impl Future<Output = Result<ModelAccess, Error>> + Send;
    // TODO: answered by the host until the verified human's agent grants are readable from Rust
    fn delegated_agent_team(
        &self,
        delegated: &DelegatedTeam<'_>,
    ) -> impl Future<Output = Result<Option<String>, Error>> + Send;
}
