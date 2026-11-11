use std::{
    fmt,
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use litellm_config::LitellmRole;
use litellm_coroutine::{Co, Reply};
use litellm_gateway_auth_core::{
    AuthHost, BudgetExceeded, Error, Rows,
    authorize::Membership,
    key_budget::KeyBudgetAlert,
    budgets::{BudgetAlert, MemberBudget, NamedBudget, RecordedBudget, SpendCounter},
    jwt::{
        AgentIdentityFailure, AliasRead, DelegatedTeam, FinishedJwt, IdentityRead, JwtIdentity, KeyMappingRow,
        KeyMappingWrite, ManagedAgent, MappedJwt, MappedKey, ModelAccess, Org, RegisteredAgent, TeamRead, UserLookup,
        UserRead,
    },
    key::{CachedTeam, Keyed, Owner},
};
use pyo3::{Py, PyAny, Python};
use serde_json::Value;

/// A Python record the core carries without reading it: a `UserAPIKeyAuth`, a team, a user.
#[derive(Clone)]
pub(crate) struct PyRow(Arc<Py<PyAny>>);

impl PyRow {
    pub(crate) fn new(row: Py<PyAny>) -> Self {
        Self(Arc::new(row))
    }

    pub(crate) fn object(&self, py: Python<'_>) -> Py<PyAny> {
        self.0.clone_ref(py)
    }
}

impl fmt::Debug for PyRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PyRow")
    }
}

impl PartialEq for PyRow {
    fn eq(&self, other: &Self) -> bool {
        self.0.is(&*other.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PyRows;

impl Rows for PyRows {
    type Key = PyRow;
    type User = PyRow;
    type Team = PyRow;
    type EndUser = PyRow;
    type ObjectPermission = PyRow;
    type Project = PyRow;
    type Membership = PyRow;
    type Org = PyRow;
    type Agent = PyRow;
}

/// Where Python leaves how a hostcall failed before it drops the question. Unset is a plain
/// failure.
#[derive(Clone, Default)]
pub(crate) struct Failure(Arc<Mutex<Option<Error>>>);

impl Failure {
    pub(crate) fn set(&self, error: Error) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(error);
    }

    fn take(&self) -> Error {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).take().unwrap_or(Error::HostFailed)
    }
}

/// `UserLookup`, owned so it outlives the core's borrow.
pub(crate) struct Lookup {
    pub(crate) user_id: String,
    pub(crate) user_email: Option<String>,
    pub(crate) sso_user_id: Option<String>,
    pub(crate) upsert: bool,
}

impl From<&UserLookup<'_>> for Lookup {
    fn from(lookup: &UserLookup<'_>) -> Self {
        Self {
            user_id: lookup.user_id.to_owned(),
            user_email: lookup.user_email.map(str::to_owned),
            sso_user_id: lookup.sso_user_id.map(str::to_owned),
            upsert: lookup.upsert,
        }
    }
}

/// `KeyMappingRow`, owned.
pub(crate) struct MappingRow {
    pub(crate) claim_field: String,
    pub(crate) claim_value: String,
    pub(crate) issuer: Option<String>,
    pub(crate) behavior: &'static str,
    pub(crate) existing_hash: Option<String>,
    pub(crate) team_id: Option<String>,
    pub(crate) user_id: Option<String>,
    pub(crate) org_id: Option<String>,
    pub(crate) agent_id: Option<String>,
}

/// `DelegatedTeam`, owned.
pub(crate) struct Delegation {
    pub(crate) user_id: Option<String>,
    pub(crate) agent_id: String,
    pub(crate) team_id: Option<String>,
    pub(crate) explicit_team: bool,
    pub(crate) allowed_team_ids: Option<Vec<String>>,
}

/// `FinishedJwt`, owned.
pub(crate) enum Finished {
    Identity(Box<JwtIdentity<PyRows>>),
    AutoRegisteredKey(Box<MappedJwt<PyRow>>, PyRow),
}

fn owned(text: Option<&str>) -> Option<String> {
    text.map(str::to_owned)
}

/// One host question. The core never sees Python: each method of `AuthHost` becomes one of these,
/// Python answers it, and the reply wakes the core.
pub(crate) enum Ask {
    EndUser { token: String, resolved: Option<PyRow>, reply: Reply<Result<PyRow, BudgetExceeded>> },
    CachedKey { hash: String, reply: Reply<Option<Keyed<PyRow>>> },
    UiKey { token: String, reply: Reply<Option<Keyed<PyRow>>> },
    WithSessionGrants {
        key: PyRow,
        user: PyRow,
        team: Option<PyRow>,
        membership: Option<PyRow>,
        reply: Reply<Keyed<PyRow>>,
    },
    EvictKey { hash: String, reply: Reply<()> },
    CachedTeam { team_id: String, reply: Reply<Option<CachedTeam<PyRow>>> },
    WithTeam { key: PyRow, team: PyRow, reply: Reply<PyRow> },
    DbKey { hash: String, end_user: PyRow, reply: Reply<Option<Keyed<PyRow>>> },
    WithTempBudget { key: PyRow, reply: Reply<PyRow> },
    NormalizeModel { key: PyRow, reply: Reply<()> },
    PrefetchAuthObjects { key: PyRow, end_user: PyRow, reply: Reply<()> },
    KeyOwner { user_id: String, reply: Reply<Option<Owner<PyRow>>> },
    KeySkipsBudgetChecks { key: PyRow, reply: Reply<bool> },
    TeamMemberMaxBudget { user_id: String, team_id: String, reply: Reply<Option<f64>> },
    KeyModelBudgets { key: PyRow, owner: Option<PyRow>, reply: Reply<Result<(), BudgetExceeded>> },
    ThrottleKey { key: PyRow, share: f64, reply: Reply<()> },
    KeyTeam { team_id: String, reply: Reply<Option<PyRow>> },
    ObjectPermission { id: String, reply: Reply<Option<PyRow>> },
    Project { project_id: String, reply: Reply<Option<PyRow>> },
    GlobalProxySpend(Reply<Option<f64>>),
    KeyBudgetAlert { key: PyRow, owner: Option<PyRow>, alert: KeyBudgetAlert, reply: Reply<()> },

    FetchJson { url: String, reply: Reply<Option<Value>> },
    FetchUserinfo { endpoint: String, token: String, reply: Reply<Option<Value>> },
    CachedJson { key: String, reply: Reply<Option<Value>> },
    CacheJson { key: String, value: Box<Value>, ttl: Duration, reply: Reply<()> },
    ForgetJson { key: String, reply: Reply<()> },

    Team { team_id: String, upsert: bool, reply: Reply<TeamRead<PyRow>> },
    TeamByAlias { alias: String, reply: Reply<AliasRead<PyRow>> },
    GrantingTeam { team_id: String, reply: Reply<TeamRead<PyRow>> },
    LoadTeamMembership { user_id: String, team_id: String, reply: Reply<Option<PyRow>> },
    TeamModelAccess { team: PyRow, model: String, reply: Reply<ModelAccess> },

    User { lookup: Lookup, reply: Reply<UserRead<PyRow>> },
    ResolveUserIdentity { lookup: Lookup, team_id: Option<String>, reply: Reply<IdentityRead<PyRows>> },
    SetUserRole { user: PyRow, role: LitellmRole, reply: Reply<PyRow> },
    PatchTeamMembership { user: PyRow, add: Vec<String>, remove: Vec<String>, reply: Reply<PyRow> },
    AddTeamMember { user_id: String, team_id: String, reply: Reply<()> },
    Org { org_id: String, reply: Reply<Option<Org<PyRow>>> },
    OrgByAlias { alias: String, reply: Reply<Option<Org<PyRow>>> },
    JwtEndUser { end_user_id: String, reply: Reply<PyRow> },
    FinishJwt { finished: Finished, reply: Reply<Result<PyRow, BudgetExceeded>> },

    KeyMappingHash { claim_field: String, claim_value: String, issuer: Option<String>, reply: Reply<Option<String>> },
    KeyByHash { hash: String, reply: Reply<Option<MappedKey<PyRow>>> },
    ReusableKeyHash { user_id: String, team_id: Option<String>, reply: Reply<Option<String>> },
    CreateKeyMapping { row: Box<MappingRow>, reply: Reply<KeyMappingWrite> },

    ManagedAgent { claims: Value, reply: Reply<Option<ManagedAgent<PyRow>>> },
    RecordAgentAuthentication { agent: PyRow, reply: Reply<Option<AgentIdentityFailure>> },
    RegisteredAgent { claim: String, reply: Reply<Option<RegisteredAgent>> },
    DelegatedAgentTeam { delegation: Box<Delegation>, reply: Reply<Option<String>> },
    CustomValidate { claims: Value, reply: Reply<bool> },

    UnpricedModels(Reply<Vec<String>>),
    TeamOrganization { team_id: String, reply: Reply<Option<String>> },
    AgentRequiresTraceId { agent_id: String, reply: Reply<bool> },
    HasTraceId { headers_only: bool, reply: Reply<bool> },
    MergeBudgetTags(Reply<()>),
    Spend { counter: SpendArg, reply: Reply<f64> },
    OrganizationBudget { org_id: String, reply: Reply<Option<RecordedBudget>> },
    TagBudgets(Reply<Vec<NamedBudget>>),
    ModelAccessGroupBudgets { groups: Vec<String>, reply: Reply<Vec<NamedBudget>> },
    TeamMemberBudgetState { membership: Option<PyRow>, reply: Reply<MemberBudget> },
    BudgetAlert { alert: AlertArg, reply: Reply<()> },
    CheckTeamModelAccess { membership: MembershipArg, reply: Reply<()> },
    CheckModelGrants(Reply<()>),
    CheckProject(Reply<()>),
    StampAccessGroups { membership: MembershipArg, reply: Reply<Vec<String>> },
    CheckGuardrailModification(Reply<()>),
    CheckObjectAccess(Reply<()>),
}

/// `SpendCounter`, owned: the counter's scope and entity, and what Python passes `get_current_spend`.
pub(crate) struct SpendArg {
    pub(crate) scope: &'static str,
    pub(crate) entity_id: String,
    pub(crate) team_id: Option<String>,
    pub(crate) window: Option<String>,
    pub(crate) recorded: f64,
    pub(crate) max_budget: f64,
}

impl From<SpendCounter<'_>> for SpendArg {
    fn from(counter: SpendCounter<'_>) -> Self {
        let scope = counter.into();
        let arg = |entity_id: &str, recorded: f64, max_budget: f64| Self {
            scope,
            entity_id: entity_id.to_owned(),
            team_id: None,
            window: None,
            recorded,
            max_budget,
        };
        match counter {
            SpendCounter::Team { team_id: id, recorded, max_budget }
            | SpendCounter::Organization { org_id: id, recorded, max_budget }
            | SpendCounter::Tag { tag: id, recorded, max_budget }
            | SpendCounter::ModelAccessGroup { group: id, recorded, max_budget }
            | SpendCounter::User { user_id: id, recorded, max_budget }
            | SpendCounter::EndUser { end_user_id: id, recorded, max_budget }
            | SpendCounter::Key { token: id, recorded, max_budget } => arg(id, recorded, max_budget),
            SpendCounter::TeamWindow { team_id: id, window } | SpendCounter::KeyWindow { token: id, window } => Self {
                window: Some(window.duration.clone()),
                ..arg(id, 0.0, window.max_budget)
            },
            SpendCounter::TeamMember { user_id, team_id, recorded, max_budget } => Self {
                team_id: Some(team_id.to_owned()),
                ..arg(user_id, recorded, max_budget)
            },
        }
    }
}

/// `BudgetAlert`, owned.
pub(crate) struct AlertArg {
    pub(crate) kind: &'static str,
    pub(crate) org_id: Option<String>,
    pub(crate) spend: Option<f64>,
    pub(crate) max_budget: Option<f64>,
}

impl From<BudgetAlert<'_>> for AlertArg {
    fn from(alert: BudgetAlert<'_>) -> Self {
        let kind = alert.into();
        let (org_id, spend, max_budget) = match alert {
            BudgetAlert::TeamSoftBudget => (None, None, None),
            BudgetAlert::Organization { org_id, spend, max_budget } => (Some(org_id.to_owned()), Some(spend), Some(max_budget)),
            BudgetAlert::Team { spend, max_budget }
            | BudgetAlert::User { spend, max_budget }
            | BudgetAlert::TeamMember { spend, max_budget } => (None, Some(spend), Some(max_budget)),
        };
        Self { kind, org_id, spend, max_budget }
    }
}

/// `Membership`, owned.
pub(crate) enum MembershipArg {
    Unread,
    Read(Option<PyRow>),
}

impl From<Membership<'_, PyRow>> for MembershipArg {
    fn from(membership: Membership<'_, PyRow>) -> Self {
        match membership {
            Membership::Unread => Self::Unread,
            Membership::Read(row) => Self::Read(row.cloned()),
        }
    }
}

pub(crate) struct PythonAuthHost {
    pub(crate) co: Co<Ask>,
    pub(crate) failure: Failure,
}

impl PythonAuthHost {
    async fn ask<A>(&self, ask: impl FnOnce(Reply<A>) -> Ask) -> Result<A, Error> {
        self.co.yield_(ask).await.map_err(|_| self.failure.take())
    }
}

impl AuthHost for PythonAuthHost {
    type Rows = PyRows;

    async fn end_user(&self, token: &str, resolved: Option<&PyRow>) -> Result<PyRow, Error> {
        let (token, resolved) = (token.to_owned(), resolved.cloned());
        self.ask(|reply| Ask::EndUser { token, resolved, reply }).await?.map_err(Error::from)
    }
    async fn cached_key(&self, hash: &str) -> Result<Option<Keyed<PyRow>>, Error> {
        let hash = hash.to_owned();
        self.ask(|reply| Ask::CachedKey { hash, reply }).await
    }
    async fn ui_key(&self, token: &str) -> Result<Option<Keyed<PyRow>>, Error> {
        let token = token.to_owned();
        self.ask(|reply| Ask::UiKey { token, reply }).await
    }
    async fn with_session_grants(
        &self,
        key: &PyRow,
        user: &PyRow,
        team: Option<&PyRow>,
        membership: Option<&PyRow>,
    ) -> Result<Keyed<PyRow>, Error> {
        let (key, user, team, membership) = (key.clone(), user.clone(), team.cloned(), membership.cloned());
        self.ask(|reply| Ask::WithSessionGrants {
            key,
            user,
            team,
            membership,
            reply,
        })
        .await
    }
    async fn evict_key(&self, hash: &str) -> Result<(), Error> {
        let hash = hash.to_owned();
        self.ask(|reply| Ask::EvictKey { hash, reply }).await
    }
    async fn cached_team(&self, team_id: &str) -> Result<Option<CachedTeam<PyRow>>, Error> {
        let team_id = team_id.to_owned();
        self.ask(|reply| Ask::CachedTeam { team_id, reply }).await
    }
    async fn with_team(&self, key: &PyRow, team: &PyRow) -> Result<PyRow, Error> {
        let (key, team) = (key.clone(), team.clone());
        self.ask(|reply| Ask::WithTeam { key, team, reply }).await
    }
    async fn db_key(&self, hash: &str, end_user: &PyRow) -> Result<Option<Keyed<PyRow>>, Error> {
        let (hash, end_user) = (hash.to_owned(), end_user.clone());
        self.ask(|reply| Ask::DbKey { hash, end_user, reply }).await
    }
    async fn with_temp_budget(&self, key: &PyRow) -> Result<PyRow, Error> {
        let key = key.clone();
        self.ask(|reply| Ask::WithTempBudget { key, reply }).await
    }
    async fn normalize_model(&self, key: &PyRow) -> Result<(), Error> {
        let key = key.clone();
        self.ask(|reply| Ask::NormalizeModel { key, reply }).await
    }
    async fn prefetch_auth_objects(&self, key: &PyRow, end_user: &PyRow) -> Result<(), Error> {
        let (key, end_user) = (key.clone(), end_user.clone());
        self.ask(|reply| Ask::PrefetchAuthObjects { key, end_user, reply }).await
    }
    async fn key_owner(&self, user_id: &str) -> Result<Option<Owner<PyRow>>, Error> {
        let user_id = user_id.to_owned();
        self.ask(|reply| Ask::KeyOwner { user_id, reply }).await
    }
    async fn key_skips_budget_checks(&self, key: &PyRow) -> Result<bool, Error> {
        let key = key.clone();
        self.ask(|reply| Ask::KeySkipsBudgetChecks { key, reply }).await
    }
    async fn team_member_max_budget(&self, user_id: &str, team_id: &str) -> Result<Option<f64>, Error> {
        let (user_id, team_id) = (user_id.to_owned(), team_id.to_owned());
        self.ask(|reply| Ask::TeamMemberMaxBudget { user_id, team_id, reply }).await
    }
    async fn key_model_budgets(&self, key: &PyRow, owner: Option<&PyRow>) -> Result<(), Error> {
        let (key, owner) = (key.clone(), owner.cloned());
        self.ask(|reply| Ask::KeyModelBudgets { key, owner, reply }).await?.map_err(Error::from)
    }
    async fn throttle_key(&self, key: &PyRow, share: f64) -> Result<(), Error> {
        let key = key.clone();
        self.ask(|reply| Ask::ThrottleKey { key, share, reply }).await
    }
    async fn key_team(&self, team_id: &str) -> Result<Option<PyRow>, Error> {
        let team_id = team_id.to_owned();
        self.ask(|reply| Ask::KeyTeam { team_id, reply }).await
    }
    async fn object_permission(&self, id: &str) -> Result<Option<PyRow>, Error> {
        let id = id.to_owned();
        self.ask(|reply| Ask::ObjectPermission { id, reply }).await
    }
    async fn project(&self, project_id: &str) -> Result<Option<PyRow>, Error> {
        let project_id = project_id.to_owned();
        self.ask(|reply| Ask::Project { project_id, reply }).await
    }
    async fn global_proxy_spend(&self) -> Result<Option<f64>, Error> {
        self.ask(Ask::GlobalProxySpend).await
    }
    async fn key_budget_alert(&self, key: &PyRow, owner: Option<&PyRow>, alert: KeyBudgetAlert) -> Result<(), Error> {
        let (key, owner) = (key.clone(), owner.cloned());
        self.ask(|reply| Ask::KeyBudgetAlert { key, owner, alert, reply }).await
    }

    async fn fetch_json(&self, url: &str) -> Result<Option<Value>, Error> {
        let url = url.to_owned();
        self.ask(|reply| Ask::FetchJson { url, reply }).await
    }
    async fn fetch_userinfo(&self, endpoint: &str, token: &str) -> Result<Option<Value>, Error> {
        let (endpoint, token) = (endpoint.to_owned(), token.to_owned());
        self.ask(|reply| Ask::FetchUserinfo { endpoint, token, reply }).await
    }
    async fn cached_json(&self, key: &str) -> Result<Option<Value>, Error> {
        let key = key.to_owned();
        self.ask(|reply| Ask::CachedJson { key, reply }).await
    }
    async fn cache_json(&self, key: &str, value: &Value, ttl: Duration) -> Result<(), Error> {
        let (key, value) = (key.to_owned(), Box::new(value.clone()));
        self.ask(|reply| Ask::CacheJson { key, value, ttl, reply }).await
    }
    async fn forget_json(&self, key: &str) -> Result<(), Error> {
        let key = key.to_owned();
        self.ask(|reply| Ask::ForgetJson { key, reply }).await
    }

    async fn team(&self, team_id: &str, upsert: bool) -> Result<TeamRead<PyRow>, Error> {
        let team_id = team_id.to_owned();
        self.ask(|reply| Ask::Team { team_id, upsert, reply }).await
    }
    async fn team_by_alias(&self, alias: &str) -> Result<AliasRead<PyRow>, Error> {
        let alias = alias.to_owned();
        self.ask(|reply| Ask::TeamByAlias { alias, reply }).await
    }
    async fn granting_team(&self, team_id: &str) -> Result<TeamRead<PyRow>, Error> {
        let team_id = team_id.to_owned();
        self.ask(|reply| Ask::GrantingTeam { team_id, reply }).await
    }
    async fn load_team_membership(&self, user_id: &str, team_id: &str) -> Result<Option<PyRow>, Error> {
        let (user_id, team_id) = (user_id.to_owned(), team_id.to_owned());
        self.ask(|reply| Ask::LoadTeamMembership { user_id, team_id, reply }).await
    }

    async fn user(&self, lookup: &UserLookup<'_>) -> Result<UserRead<PyRow>, Error> {
        let lookup = Lookup::from(lookup);
        self.ask(|reply| Ask::User { lookup, reply }).await
    }
    async fn resolve_user_identity(
        &self,
        lookup: &UserLookup<'_>,
        team_id: Option<&str>,
    ) -> Result<IdentityRead<PyRows>, Error> {
        let (lookup, team_id) = (Lookup::from(lookup), owned(team_id));
        self.ask(|reply| Ask::ResolveUserIdentity { lookup, team_id, reply }).await
    }
    async fn set_user_role(&self, user: &PyRow, role: LitellmRole) -> Result<PyRow, Error> {
        let user = user.clone();
        self.ask(|reply| Ask::SetUserRole { user, role, reply }).await
    }
    async fn patch_team_membership(&self, user: &PyRow, add: &[String], remove: &[String]) -> Result<PyRow, Error> {
        let (user, add, remove) = (user.clone(), add.to_vec(), remove.to_vec());
        self.ask(|reply| Ask::PatchTeamMembership {
            user,
            add,
            remove,
            reply,
        })
        .await
    }
    async fn add_team_member(&self, user_id: &str, team_id: &str) -> Result<(), Error> {
        let (user_id, team_id) = (user_id.to_owned(), team_id.to_owned());
        self.ask(|reply| Ask::AddTeamMember { user_id, team_id, reply }).await
    }
    async fn org(&self, org_id: &str) -> Result<Option<Org<PyRow>>, Error> {
        let org_id = org_id.to_owned();
        self.ask(|reply| Ask::Org { org_id, reply }).await
    }
    async fn org_by_alias(&self, alias: &str) -> Result<Option<Org<PyRow>>, Error> {
        let alias = alias.to_owned();
        self.ask(|reply| Ask::OrgByAlias { alias, reply }).await
    }
    async fn jwt_end_user(&self, end_user_id: &str) -> Result<PyRow, Error> {
        let end_user_id = end_user_id.to_owned();
        self.ask(|reply| Ask::JwtEndUser { end_user_id, reply }).await
    }
    async fn finish_jwt(&self, finished: FinishedJwt<'_, PyRows>) -> Result<PyRow, Error> {
        let finished = match finished {
            FinishedJwt::Identity(identity) => Finished::Identity(Box::new(identity.clone())),
            FinishedJwt::AutoRegisteredKey(jwt, key) => Finished::AutoRegisteredKey(Box::new(jwt.clone()), key.clone()),
        };
        self.ask(|reply| Ask::FinishJwt { finished, reply }).await?.map_err(Error::from)
    }

    async fn key_mapping_hash(
        &self,
        claim_field: &str,
        claim_value: &str,
        issuer: Option<&str>,
    ) -> Result<Option<String>, Error> {
        let (claim_field, claim_value, issuer) = (claim_field.to_owned(), claim_value.to_owned(), owned(issuer));
        self.ask(|reply| Ask::KeyMappingHash {
            claim_field,
            claim_value,
            issuer,
            reply,
        })
        .await
    }
    async fn key_by_hash(&self, hash: &str) -> Result<Option<MappedKey<PyRow>>, Error> {
        let hash = hash.to_owned();
        self.ask(|reply| Ask::KeyByHash { hash, reply }).await
    }
    async fn reusable_key_hash(&self, user_id: &str, team_id: Option<&str>) -> Result<Option<String>, Error> {
        let (user_id, team_id) = (user_id.to_owned(), owned(team_id));
        self.ask(|reply| Ask::ReusableKeyHash { user_id, team_id, reply }).await
    }
    async fn create_key_mapping(&self, row: &KeyMappingRow<'_>) -> Result<KeyMappingWrite, Error> {
        let row = Box::new(MappingRow {
            claim_field: row.claim_field.to_owned(),
            claim_value: row.claim_value.to_owned(),
            issuer: owned(row.issuer),
            behavior: row.behavior.into(),
            existing_hash: owned(row.existing_hash),
            team_id: owned(row.team_id),
            user_id: owned(row.user_id),
            org_id: owned(row.org_id),
            agent_id: owned(row.agent_id),
        });
        self.ask(|reply| Ask::CreateKeyMapping { row, reply }).await
    }

    async fn managed_agent(&self, claims: &Value) -> Result<Option<ManagedAgent<PyRow>>, Error> {
        let claims = claims.clone();
        self.ask(|reply| Ask::ManagedAgent { claims, reply }).await
    }
    async fn record_agent_authentication(&self, agent: &PyRow) -> Result<Option<AgentIdentityFailure>, Error> {
        let agent = agent.clone();
        self.ask(|reply| Ask::RecordAgentAuthentication { agent, reply }).await
    }
    async fn registered_agent(&self, claim: &str) -> Result<Option<RegisteredAgent>, Error> {
        let claim = claim.to_owned();
        self.ask(|reply| Ask::RegisteredAgent { claim, reply }).await
    }
    async fn delegated_agent_team(&self, delegated: &DelegatedTeam<'_>) -> Result<Option<String>, Error> {
        let delegation = Box::new(Delegation {
            user_id: owned(delegated.user_id),
            agent_id: delegated.agent_id.to_owned(),
            team_id: owned(delegated.team_id),
            explicit_team: delegated.explicit_team,
            allowed_team_ids: delegated.allowed_team_ids.map(|ids| ids.iter().cloned().collect()),
        });
        self.ask(|reply| Ask::DelegatedAgentTeam { delegation, reply }).await
    }
    async fn custom_validate(&self, claims: &Value) -> Result<bool, Error> {
        let claims = claims.clone();
        self.ask(|reply| Ask::CustomValidate { claims, reply }).await
    }
    async fn unpriced_models(&self) -> Result<Vec<String>, Error> {
        self.ask(Ask::UnpricedModels).await
    }
    async fn agent_requires_trace_id(&self, agent_id: &str) -> Result<bool, Error> {
        let agent_id = agent_id.to_owned();
        self.ask(|reply| Ask::AgentRequiresTraceId { agent_id, reply }).await
    }
    async fn has_trace_id(&self, headers_only: bool) -> Result<bool, Error> {
        self.ask(|reply| Ask::HasTraceId { headers_only, reply }).await
    }
    async fn team_organization(&self, team_id: &str) -> Result<Option<String>, Error> {
        let team_id = team_id.to_owned();
        self.ask(|reply| Ask::TeamOrganization { team_id, reply }).await
    }
    async fn merge_budget_tags(&self) -> Result<(), Error> {
        self.ask(Ask::MergeBudgetTags).await
    }
    async fn spend(&self, counter: SpendCounter<'_>) -> Result<f64, Error> {
        let counter = SpendArg::from(counter);
        self.ask(|reply| Ask::Spend { counter, reply }).await
    }
    async fn organization_budget(&self, org_id: &str) -> Result<Option<RecordedBudget>, Error> {
        let org_id = org_id.to_owned();
        self.ask(|reply| Ask::OrganizationBudget { org_id, reply }).await
    }
    async fn tag_budgets(&self) -> Result<Vec<NamedBudget>, Error> {
        self.ask(Ask::TagBudgets).await
    }
    async fn model_access_group_budgets(&self, groups: &[String]) -> Result<Vec<NamedBudget>, Error> {
        let groups = groups.to_vec();
        self.ask(|reply| Ask::ModelAccessGroupBudgets { groups, reply }).await
    }
    async fn team_member_budget_state(&self, membership: Option<&PyRow>) -> Result<MemberBudget, Error> {
        let membership = membership.cloned();
        self.ask(|reply| Ask::TeamMemberBudgetState { membership, reply }).await
    }
    async fn budget_alert(&self, alert: BudgetAlert<'_>) -> Result<(), Error> {
        let alert = AlertArg::from(alert);
        self.ask(|reply| Ask::BudgetAlert { alert, reply }).await
    }
    async fn check_team_model_access(&self, membership: Membership<'_, PyRow>) -> Result<(), Error> {
        let membership = MembershipArg::from(membership);
        self.ask(|reply| Ask::CheckTeamModelAccess { membership, reply }).await
    }
    async fn check_model_grants(&self) -> Result<(), Error> {
        self.ask(Ask::CheckModelGrants).await
    }
    async fn check_project(&self) -> Result<(), Error> {
        self.ask(Ask::CheckProject).await
    }
    async fn stamp_access_groups(&self, membership: Membership<'_, PyRow>) -> Result<Vec<String>, Error> {
        let membership = MembershipArg::from(membership);
        self.ask(|reply| Ask::StampAccessGroups { membership, reply }).await
    }
    async fn check_guardrail_modification(&self) -> Result<(), Error> {
        self.ask(Ask::CheckGuardrailModification).await
    }
    async fn check_object_access(&self) -> Result<(), Error> {
        self.ask(Ask::CheckObjectAccess).await
    }
    async fn team_model_access(&self, team: &PyRow, model: &str) -> Result<ModelAccess, Error> {
        let (team, model) = (team.clone(), model.to_owned());
        self.ask(|reply| Ask::TeamModelAccess { team, model, reply }).await
    }
}
