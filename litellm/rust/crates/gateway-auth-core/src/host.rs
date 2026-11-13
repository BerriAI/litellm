use std::{fmt::Debug, future::Future, time::Duration};

use litellm_config::LitellmRole;
use serde_json::Value;

use crate::{
    Error,
    credential::jwt::{
        AgentIdentityFailure, AliasRead, DelegatedTeam, FinishedJwt, IdentityRead, KeyMappingRow,
        KeyMappingWrite, ManagedAgent, MappedKey, ModelAccess, Org, RegisteredAgent, TeamRead,
        UserLookup, UserRead,
    },
    credential::key::{CachedTeam, KeyHash, Keyed, Owner, Token},
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

/// Code the operator wrote, which only a host that embeds their Python can run.
pub trait OperatorHooks: Send + Sync {
    /// The operator's `custom_validate` callable. The core decides when it runs and what `false` means.
    fn custom_validate(&self, claims: &Value) -> impl Future<Output = Result<bool, Error>> + Send;
}

/// Everything the core reads, writes or hands off, grouped by why the core cannot do it itself,
/// which is also how each method leaves.
///
/// Store ports come first. Each reads or writes one row, cache entry or remote document and never
/// decides. They stay for good, since a host that answers one natively still answers the method.
/// The operator's own Python stays too, behind [`OperatorHooks`], which only a host that can run it
/// offers. Every section after it is temporary, and each method there has a TODO naming what has
/// to reach Rust first.
///
/// A method that a second host could only answer by re-implementing a rule is a decision, however
/// it is named, and belongs in those last sections.
pub trait AuthHost: Send + Sync {
    type Rows: Rows;

    // Store reads. Each returns the row with the facts the core decides on.
    //
    // TODO: one reader per row, taking the read policy (cache only, cache then database, database
    // only, upsert) as an argument. Today a key has three (`cached_key`, `db_key`, `key_by_hash`),
    // a user three (`key_owner`, `user`, `resolve_user_identity`), a team four (`cached_team`,
    // `key_team`, `team`, `granting_team`) and an end user two, each with its own facts beside the
    // row, so a native store would answer the same read several times

    /// The key cached under `hash`.
    fn cached_key(
        &self,
        hash: &KeyHash,
    ) -> impl Future<Output = Result<Option<Keyed<KeyRow<Self>>>, Error>> + Send;
    // TODO: the host stamps the end user's limits onto the row until the core returns them beside
    // the key, which leaves this a plain read by hash like `key_by_hash`
    /// The key's database row, stamped with the end user's limits. `None` when no key has the hash.
    fn db_key(
        &self,
        hash: &KeyHash,
        end_user: &EndUserRow<Self>,
    ) -> impl Future<Output = Result<Option<Keyed<KeyRow<Self>>>, Error>> + Send;
    /// `IdentityStore.resolve(hashed_token)`
    fn key_by_hash(
        &self,
        hash: &KeyHash,
    ) -> impl Future<Output = Result<Option<MappedKey<KeyRow<Self>>>, Error>> + Send;
    /// `get_jwt_key_mapping_object` for one scope: the issuer's, or the global one for `None`.
    fn key_mapping_hash(
        &self,
        claim_field: &str,
        claim_value: &str,
        issuer: Option<&str>,
    ) -> impl Future<Output = Result<Option<KeyHash>, Error>> + Send;
    /// `_reusable_key_hash_for_user`
    fn reusable_key_hash(
        &self,
        user_id: &str,
        team_id: Option<&str>,
    ) -> impl Future<Output = Result<Option<KeyHash>, Error>> + Send;
    /// The key's owner and whether SCIM deactivated them.
    fn key_owner(
        &self,
        user_id: &str,
    ) -> impl Future<Output = Result<Option<Owner<UserRow<Self>>>, Error>> + Send;
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
    /// `get_team_object`, with the team's facts and its pass-through grants beside the row.
    fn team(
        &self,
        team_id: &str,
        upsert: bool,
    ) -> impl Future<Output = Result<TeamRead<TeamRow<Self>>, Error>> + Send;
    /// `get_team_object(team_id, check_db_only=True)` for the team a delegated agent's grant selected.
    fn granting_team(
        &self,
        team_id: &str,
    ) -> impl Future<Output = Result<TeamRead<TeamRow<Self>>, Error>> + Send;
    /// `get_team_object_by_alias`
    fn team_by_alias(
        &self,
        alias: &str,
    ) -> impl Future<Output = Result<AliasRead<TeamRow<Self>>, Error>> + Send;
    /// `get_team_membership`
    fn load_team_membership(
        &self,
        user_id: &str,
        team_id: &str,
    ) -> impl Future<Output = Result<Option<MembershipRow<Self>>, Error>> + Send;
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
    fn object_permission(
        &self,
        id: &str,
    ) -> impl Future<Output = Result<Option<ObjectPermissionRow<Self>>, Error>> + Send;
    fn project(
        &self,
        project_id: &str,
    ) -> impl Future<Output = Result<Option<ProjectRow<Self>>, Error>> + Send;
    // TODO: the host reads the agent's claims, checks the binding and classifies the subject until
    // the binding, the agent and the subject are reads of their own and the core decides on them
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

    // Store writes the core decided on. A user write returns the user row with the change applied.

    /// Drops the cached key, after the core found it stale or unusable.
    fn evict_key(&self, hash: &KeyHash) -> impl Future<Output = Result<(), Error>> + Send;
    /// Warms the cache with the rows auth reads before it knows the key's owners: the end user's,
    /// and the key under the hash of `unread`, the token whose key the caller does not hold yet.
    fn prefetch_identity(
        &self,
        unread: Option<&Token>,
        end_user_id: Option<&str>,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    /// Warms the cache with the rows the key and the end user reference, which the checks after it read.
    fn prefetch_auth_objects(
        &self,
        key: &KeyRow<Self>,
        end_user: &EndUserRow<Self>,
    ) -> impl Future<Output = Result<(), Error>> + Send;
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
    // TODO: the host runs the whole `team_member_add` management endpoint as a proxy admin until
    // the membership insert is a store write of its own
    /// `team_member_add` as a `user`, an already-a-member conflict included.
    fn add_team_member(
        &self,
        user_id: &str,
        team_id: &str,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: written by the host until key generation and the mapping insert move to Rust
    /// Mints a key when the row names none, then writes the mapping with its race handling. The
    /// core has already decided whether to map, reuse or mint.
    fn create_key_mapping(
        &self,
        row: &KeyMappingRow<'_>,
    ) -> impl Future<Output = Result<KeyMappingWrite, Error>> + Send;
    /// `AgentIdentityStore.record_authentication`
    fn record_agent_authentication(
        &self,
        agent: &AgentRow<Self>,
    ) -> impl Future<Output = Result<Option<AgentIdentityFailure>, Error>> + Send;

    // JWT documents and keyspaces. The core picks the URL and the cache key, and owns the TTLs and
    // stale windows.
    //
    // TODO: `fetch_json` and the three cache methods are generic until each document (JWKS, OIDC
    // discovery, key mapping entry) has a named method, since a host can only tell them apart by
    // matching on the URL or the key

    /// GET with the transient-failure retries. `None` once transport failures outlast them; a
    /// non-200 or non-JSON reply is a host error carrying the proxy's message.
    fn fetch_json(&self, url: &str) -> impl Future<Output = Result<Option<Value>, Error>> + Send;
    /// GET the OIDC UserInfo endpoint with `Authorization: Bearer {token}`. `None` is a JSON
    /// `null` body; any failure is a host error carrying `Failed to fetch OIDC UserInfo: ...`.
    fn fetch_userinfo(
        &self,
        endpoint: &str,
        token: &Token,
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

    // The permanent host step. It runs operator code, so there is no rule to port.

    /// The operator's own code, or `None` on a host that cannot run it. Such a host hands a request
    /// planned as [`Plan::JwtWithOperatorCode`](crate::plan::Plan::JwtWithOperatorCode) to one
    /// that can, and the core refuses the request if it is asked to run the code anyway.
    fn operator_hooks(&self) -> Option<&impl OperatorHooks>;

    // Decisions the host still answers from state only the Python process holds. Each takes the
    // rows it decides on as arguments, and leaves together with its bridge step and its `drive`
    // case once its TODO lands.

    // TODO: rewritten by the host until the router's model aliases are readable from router-core,
    // when the core resolves the model and returns the rewrite in its outcome. Until then the host
    // rewrites the request in place, where the budget steps and the handlers after auth read it
    /// Rewrites the request's model to the one the key's aliases and the router resolve it to, for
    /// the checks after it.
    fn normalize_model(&self, key: &KeyRow<Self>)
    -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: decided by the host until `validate_end_user_id_in_db` and the default end-user budget
    // are settings the core reads, which leaves a read of whether the id is known
    /// The end-user id the request keeps. That is `end_user_id` itself, or `None` when the proxy
    /// validates end-user ids, this one is unknown and no default end-user budget covers it.
    fn validated_end_user_id(
        &self,
        end_user_id: &str,
    ) -> impl Future<Output = Result<Option<String>, Error>> + Send;
    // TODO: decided by the host until request pricing is readable from the router
    /// Whether the request is free, so no budget applies to the key.
    fn key_skips_budget_checks(
        &self,
        key: &KeyRow<Self>,
    ) -> impl Future<Output = Result<bool, Error>> + Send;
    // TODO: answered by the host until access groups and model aliases are readable from router-core
    fn team_model_access(
        &self,
        team: &TeamRow<Self>,
        model: &str,
    ) -> impl Future<Output = Result<ModelAccess, Error>> + Send;
    // TODO: answered by the host until the verified human's agent grants are readable from Rust,
    // which leaves the choice among the granting teams to the core
    fn delegated_agent_team(
        &self,
        delegated: &DelegatedTeam<'_>,
    ) -> impl Future<Output = Result<Option<String>, Error>> + Send;

    // Budgets the host still decides, because the spend counters, alerts and budget fallback rewrite
    // are Python-owned. Each returns the host's verdict, and the core renders the denial.

    // TODO: decided by the host until end-user budgets move to Rust. It also reads the default
    // end-user budget and merges the budget's limits into the row, which become a read of their
    // own and facts the core returns beside the key
    /// The end user `end_user_id` names with the limits it carries, refusing one over budget. The
    /// row is empty for `None`, a request that names no end user.
    fn end_user(
        &self,
        end_user_id: Option<&str>,
    ) -> impl Future<Output = Result<EndUserRow<Self>, Error>> + Send;
    // TODO: decided by the host until team member budgets and their spend counters move to Rust
    /// The key owner's spend in the key's team against the membership's budget, alerting on the
    /// request it refuses.
    fn team_member_budget(
        &self,
        key: &KeyRow<Self>,
        owner: Option<&UserRow<Self>>,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: decided by the host until key budgets, their alerts and the model budget fallback move to Rust
    /// The key's max budget alert, max budget on LLM routes, soft budget, then its and its owner's
    /// per-model budgets. An over-budget key that opted into throttling is throttled instead.
    fn key_budgets(
        &self,
        key: &KeyRow<Self>,
        owner: Option<&UserRow<Self>>,
        is_llm_api_route: bool,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: the host reads the global spend and skips the alert without one, until that is a read
    // of the spend and an alert the core asks for
    /// Sends the proxy budget alert with the proxy's global spend.
    fn proxy_budget_alert(
        &self,
        key: &KeyRow<Self>,
    ) -> impl Future<Output = Result<(), Error>> + Send;
    // TODO: decided by the host until the user per-model budget moves to Rust and the project read
    // uses `project`
    /// Builds the request's token from the identity and its rows, then runs the JWT path's project
    /// lookup and per-model user budget check on it, and returns the token.
    fn finish_jwt(
        &self,
        finished: FinishedJwt<'_, Self::Rows>,
    ) -> impl Future<Output = Result<KeyRow<Self>, Error>> + Send;

    // Pure rules the host still runs. No Python-owned state blocks them. The core cannot write a
    // row, so each leaves once the core returns its result as facts beside the row and the host
    // applies them when it converts the identity.

    // TODO: decoded and refreshed by the host until session tokens are their own credential path
    // in Rust
    /// The UI or `lite login` session key `token` encodes. A `lite login` key carries the grants of
    /// its user's and team's current rows, and the host refuses it when either is gone.
    fn ui_key(
        &self,
        token: &Token,
    ) -> impl Future<Output = Result<Option<Keyed<KeyRow<Self>>>, Error>> + Send;
    // TODO: rebuilt by the host until the core returns the fresher team beside the key
    /// The key with the `team_*` values of a fresher cached team.
    fn with_team(
        &self,
        key: &KeyRow<Self>,
        team: &TeamRow<Self>,
    ) -> impl Future<Output = Result<KeyRow<Self>, Error>> + Send;
    // TODO: added up by the host until the key's budget and its temporary increase are facts the
    // core reads, with the key budgets
    /// The key with an unexpired temporary budget increase added to its budget, in the row and in
    /// the facts the budget checks decide on.
    fn with_temp_budget(
        &self,
        key: Keyed<KeyRow<Self>>,
    ) -> impl Future<Output = Result<Keyed<KeyRow<Self>>, Error>> + Send;
}
