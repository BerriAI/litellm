use std::{fmt, sync::Arc, time::Duration};

use litellm_config::LitellmRole;
use litellm_coroutine::{Co, Reply};
use litellm_gateway_auth_core::{
    AgentIdentityFailure, AuthHost, BudgetExceeded, Error, KeyHash, OperatorHooks, Rows, Token,
    authn::{
        jwt::{KeyMappingRow, KeyMappingWrite, MappedJwt, MappedKey},
        key::{CachedTeam, Keyed},
    },
    identity::{
        jwt::{
            AliasRead, DelegatedTeam, FinishedJwt, IdentityRead, JwtIdentity, ManagedAgent,
            ModelAccess, Org, RegisteredAgent, TeamRead, UserLookup, UserRead,
        },
        key::Owner,
    },
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
    pub(crate) existing_hash: Option<KeyHash>,
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
    ValidatedEndUserId {
        end_user_id: String,
        reply: Reply<Option<String>>,
    },
    PrefetchIdentity {
        token: Option<Token>,
        end_user_id: Option<String>,
        reply: Reply<()>,
    },
    EndUser {
        end_user_id: Option<String>,
        reply: Reply<Result<PyRow, BudgetExceeded>>,
    },
    CachedKey {
        hash: KeyHash,
        reply: Reply<Option<Keyed<PyRow>>>,
    },
    UiKey {
        token: Token,
        reply: Reply<Option<Keyed<PyRow>>>,
    },
    EvictKey {
        hash: KeyHash,
        reply: Reply<()>,
    },
    CachedTeam {
        team_id: String,
        reply: Reply<Option<CachedTeam<PyRow>>>,
    },
    WithTeam {
        key: PyRow,
        team: PyRow,
        reply: Reply<PyRow>,
    },
    DbKey {
        hash: KeyHash,
        end_user: PyRow,
        reply: Reply<Option<Keyed<PyRow>>>,
    },
    WithTempBudget {
        key: PyRow,
        reply: Reply<Keyed<PyRow>>,
    },
    NormalizeModel {
        key: PyRow,
        reply: Reply<()>,
    },
    PrefetchAuthObjects {
        key: PyRow,
        end_user: PyRow,
        reply: Reply<()>,
    },
    KeyOwner {
        user_id: String,
        reply: Reply<Option<Owner<PyRow>>>,
    },
    KeySkipsBudgetChecks {
        key: PyRow,
        reply: Reply<bool>,
    },
    TeamMemberBudget {
        key: PyRow,
        owner: Option<PyRow>,
        reply: Reply<Result<(), BudgetExceeded>>,
    },
    KeyBudgets {
        key: PyRow,
        owner: Option<PyRow>,
        is_llm_api_route: bool,
        reply: Reply<Result<(), BudgetExceeded>>,
    },
    KeyTeam {
        team_id: String,
        reply: Reply<Option<PyRow>>,
    },
    ObjectPermission {
        id: String,
        reply: Reply<Option<PyRow>>,
    },
    Project {
        project_id: String,
        reply: Reply<Option<PyRow>>,
    },
    ProxyBudgetAlert {
        key: PyRow,
        reply: Reply<()>,
    },

    FetchJson {
        url: String,
        reply: Reply<Option<Value>>,
    },
    FetchUserinfo {
        endpoint: String,
        token: Token,
        reply: Reply<Option<Value>>,
    },
    CachedJson {
        key: String,
        reply: Reply<Option<Value>>,
    },
    CacheJson {
        key: String,
        value: Box<Value>,
        ttl: Duration,
        reply: Reply<()>,
    },
    ForgetJson {
        key: String,
        reply: Reply<()>,
    },

    Team {
        team_id: String,
        upsert: bool,
        reply: Reply<TeamRead<PyRow>>,
    },
    TeamByAlias {
        alias: String,
        reply: Reply<AliasRead<PyRow>>,
    },
    GrantingTeam {
        team_id: String,
        reply: Reply<TeamRead<PyRow>>,
    },
    LoadTeamMembership {
        user_id: String,
        team_id: String,
        reply: Reply<Option<PyRow>>,
    },
    TeamModelAccess {
        team: PyRow,
        model: String,
        reply: Reply<ModelAccess>,
    },

    User {
        lookup: Lookup,
        reply: Reply<UserRead<PyRow>>,
    },
    ResolveUserIdentity {
        lookup: Lookup,
        team_id: Option<String>,
        reply: Reply<IdentityRead<PyRows>>,
    },
    SetUserRole {
        user: PyRow,
        role: LitellmRole,
        reply: Reply<PyRow>,
    },
    PatchTeamMembership {
        user: PyRow,
        add: Vec<String>,
        remove: Vec<String>,
        reply: Reply<PyRow>,
    },
    AddTeamMember {
        user_id: String,
        team_id: String,
        reply: Reply<()>,
    },
    Org {
        org_id: String,
        reply: Reply<Option<Org<PyRow>>>,
    },
    OrgByAlias {
        alias: String,
        reply: Reply<Option<Org<PyRow>>>,
    },
    JwtEndUser {
        end_user_id: String,
        reply: Reply<PyRow>,
    },
    FinishJwt {
        finished: Finished,
        reply: Reply<Result<PyRow, BudgetExceeded>>,
    },

    KeyMappingHash {
        claim_field: String,
        claim_value: String,
        issuer: Option<String>,
        reply: Reply<Option<KeyHash>>,
    },
    KeyByHash {
        hash: KeyHash,
        reply: Reply<Option<MappedKey<PyRow>>>,
    },
    ReusableKeyHash {
        user_id: String,
        team_id: Option<String>,
        reply: Reply<Option<KeyHash>>,
    },
    CreateKeyMapping {
        row: Box<MappingRow>,
        reply: Reply<KeyMappingWrite>,
    },

    ManagedAgent {
        claims: Value,
        reply: Reply<Option<ManagedAgent<PyRow>>>,
    },
    RecordAgentAuthentication {
        agent: PyRow,
        reply: Reply<Option<AgentIdentityFailure>>,
    },
    RegisteredAgent {
        claim: String,
        reply: Reply<Option<RegisteredAgent>>,
    },
    DelegatedAgentTeam {
        delegation: Box<Delegation>,
        reply: Reply<Option<String>>,
    },
    CustomValidate {
        claims: Value,
        reply: Reply<bool>,
    },
}

pub(crate) struct PythonAuthHost {
    pub(crate) co: Co<Ask>,
}

impl PythonAuthHost {
    async fn ask<A>(&self, ask: impl FnOnce(Reply<A>) -> Ask) -> Result<A, Error> {
        self.co.yield_(ask).await.map_err(|_| Error::HostFailed)
    }
}

impl OperatorHooks for PythonAuthHost {
    async fn custom_validate(&self, claims: &Value) -> Result<bool, Error> {
        let claims = claims.clone();
        self.ask(|reply| Ask::CustomValidate { claims, reply })
            .await
    }
}

impl AuthHost for PythonAuthHost {
    type Rows = PyRows;

    async fn validated_end_user_id(&self, end_user_id: &str) -> Result<Option<String>, Error> {
        let end_user_id = end_user_id.to_owned();
        self.ask(|reply| Ask::ValidatedEndUserId { end_user_id, reply })
            .await
    }
    async fn prefetch_identity(
        &self,
        unread: Option<&Token>,
        end_user_id: Option<&str>,
    ) -> Result<(), Error> {
        let (token, end_user_id) = (unread.cloned(), owned(end_user_id));
        self.ask(|reply| Ask::PrefetchIdentity {
            token,
            end_user_id,
            reply,
        })
        .await
    }
    async fn end_user(&self, end_user_id: Option<&str>) -> Result<PyRow, Error> {
        let end_user_id = owned(end_user_id);
        self.ask(|reply| Ask::EndUser { end_user_id, reply })
            .await?
            .map_err(Error::from)
    }
    async fn cached_key(&self, hash: &KeyHash) -> Result<Option<Keyed<PyRow>>, Error> {
        let hash = hash.clone();
        self.ask(|reply| Ask::CachedKey { hash, reply }).await
    }
    async fn ui_key(&self, token: &Token) -> Result<Option<Keyed<PyRow>>, Error> {
        let token = token.clone();
        self.ask(|reply| Ask::UiKey { token, reply }).await
    }
    async fn evict_key(&self, hash: &KeyHash) -> Result<(), Error> {
        let hash = hash.clone();
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
    async fn db_key(
        &self,
        hash: &KeyHash,
        end_user: &PyRow,
    ) -> Result<Option<Keyed<PyRow>>, Error> {
        let (hash, end_user) = (hash.clone(), end_user.clone());
        self.ask(|reply| Ask::DbKey {
            hash,
            end_user,
            reply,
        })
        .await
    }
    async fn with_temp_budget(&self, key: Keyed<PyRow>) -> Result<Keyed<PyRow>, Error> {
        let key = key.row;
        self.ask(|reply| Ask::WithTempBudget { key, reply }).await
    }
    async fn normalize_model(&self, key: &PyRow) -> Result<(), Error> {
        let key = key.clone();
        self.ask(|reply| Ask::NormalizeModel { key, reply }).await
    }
    async fn prefetch_auth_objects(&self, key: &PyRow, end_user: &PyRow) -> Result<(), Error> {
        let (key, end_user) = (key.clone(), end_user.clone());
        self.ask(|reply| Ask::PrefetchAuthObjects {
            key,
            end_user,
            reply,
        })
        .await
    }
    async fn key_owner(&self, user_id: &str) -> Result<Option<Owner<PyRow>>, Error> {
        let user_id = user_id.to_owned();
        self.ask(|reply| Ask::KeyOwner { user_id, reply }).await
    }
    async fn key_skips_budget_checks(&self, key: &PyRow) -> Result<bool, Error> {
        let key = key.clone();
        self.ask(|reply| Ask::KeySkipsBudgetChecks { key, reply })
            .await
    }
    async fn team_member_budget(&self, key: &PyRow, owner: Option<&PyRow>) -> Result<(), Error> {
        let (key, owner) = (key.clone(), owner.cloned());
        self.ask(|reply| Ask::TeamMemberBudget { key, owner, reply })
            .await?
            .map_err(Error::from)
    }
    async fn key_budgets(
        &self,
        key: &PyRow,
        owner: Option<&PyRow>,
        is_llm_api_route: bool,
    ) -> Result<(), Error> {
        let (key, owner) = (key.clone(), owner.cloned());
        self.ask(|reply| Ask::KeyBudgets {
            key,
            owner,
            is_llm_api_route,
            reply,
        })
        .await?
        .map_err(Error::from)
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
    async fn proxy_budget_alert(&self, key: &PyRow) -> Result<(), Error> {
        let key = key.clone();
        self.ask(|reply| Ask::ProxyBudgetAlert { key, reply }).await
    }

    async fn fetch_json(&self, url: &str) -> Result<Option<Value>, Error> {
        let url = url.to_owned();
        self.ask(|reply| Ask::FetchJson { url, reply }).await
    }
    async fn fetch_userinfo(&self, endpoint: &str, token: &Token) -> Result<Option<Value>, Error> {
        let (endpoint, token) = (endpoint.to_owned(), token.clone());
        self.ask(|reply| Ask::FetchUserinfo {
            endpoint,
            token,
            reply,
        })
        .await
    }
    async fn cached_json(&self, key: &str) -> Result<Option<Value>, Error> {
        let key = key.to_owned();
        self.ask(|reply| Ask::CachedJson { key, reply }).await
    }
    async fn cache_json(&self, key: &str, value: &Value, ttl: Duration) -> Result<(), Error> {
        let (key, value) = (key.to_owned(), Box::new(value.clone()));
        self.ask(|reply| Ask::CacheJson {
            key,
            value,
            ttl,
            reply,
        })
        .await
    }
    async fn forget_json(&self, key: &str) -> Result<(), Error> {
        let key = key.to_owned();
        self.ask(|reply| Ask::ForgetJson { key, reply }).await
    }

    async fn team(&self, team_id: &str, upsert: bool) -> Result<TeamRead<PyRow>, Error> {
        let team_id = team_id.to_owned();
        self.ask(|reply| Ask::Team {
            team_id,
            upsert,
            reply,
        })
        .await
    }
    async fn team_by_alias(&self, alias: &str) -> Result<AliasRead<PyRow>, Error> {
        let alias = alias.to_owned();
        self.ask(|reply| Ask::TeamByAlias { alias, reply }).await
    }
    async fn granting_team(&self, team_id: &str) -> Result<TeamRead<PyRow>, Error> {
        let team_id = team_id.to_owned();
        self.ask(|reply| Ask::GrantingTeam { team_id, reply }).await
    }
    async fn load_team_membership(
        &self,
        user_id: &str,
        team_id: &str,
    ) -> Result<Option<PyRow>, Error> {
        let (user_id, team_id) = (user_id.to_owned(), team_id.to_owned());
        self.ask(|reply| Ask::LoadTeamMembership {
            user_id,
            team_id,
            reply,
        })
        .await
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
        self.ask(|reply| Ask::ResolveUserIdentity {
            lookup,
            team_id,
            reply,
        })
        .await
    }
    async fn set_user_role(&self, user: &PyRow, role: LitellmRole) -> Result<PyRow, Error> {
        let user = user.clone();
        self.ask(|reply| Ask::SetUserRole { user, role, reply })
            .await
    }
    async fn patch_team_membership(
        &self,
        user: &PyRow,
        add: &[String],
        remove: &[String],
    ) -> Result<PyRow, Error> {
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
        self.ask(|reply| Ask::AddTeamMember {
            user_id,
            team_id,
            reply,
        })
        .await
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
        self.ask(|reply| Ask::JwtEndUser { end_user_id, reply })
            .await
    }
    async fn finish_jwt(&self, finished: FinishedJwt<'_, PyRows>) -> Result<PyRow, Error> {
        let finished = match finished {
            FinishedJwt::Identity(identity) => Finished::Identity(Box::new(identity.clone())),
            FinishedJwt::AutoRegisteredKey(jwt, key) => {
                Finished::AutoRegisteredKey(Box::new(jwt.clone()), key.clone())
            }
        };
        self.ask(|reply| Ask::FinishJwt { finished, reply })
            .await?
            .map_err(Error::from)
    }

    async fn key_mapping_hash(
        &self,
        claim_field: &str,
        claim_value: &str,
        issuer: Option<&str>,
    ) -> Result<Option<KeyHash>, Error> {
        let (claim_field, claim_value, issuer) = (
            claim_field.to_owned(),
            claim_value.to_owned(),
            owned(issuer),
        );
        self.ask(|reply| Ask::KeyMappingHash {
            claim_field,
            claim_value,
            issuer,
            reply,
        })
        .await
    }
    async fn key_by_hash(&self, hash: &KeyHash) -> Result<Option<MappedKey<PyRow>>, Error> {
        let hash = hash.clone();
        self.ask(|reply| Ask::KeyByHash { hash, reply }).await
    }
    async fn reusable_key_hash(
        &self,
        user_id: &str,
        team_id: Option<&str>,
    ) -> Result<Option<KeyHash>, Error> {
        let (user_id, team_id) = (user_id.to_owned(), owned(team_id));
        self.ask(|reply| Ask::ReusableKeyHash {
            user_id,
            team_id,
            reply,
        })
        .await
    }
    async fn create_key_mapping(&self, row: &KeyMappingRow<'_>) -> Result<KeyMappingWrite, Error> {
        let row = Box::new(MappingRow {
            claim_field: row.claim_field.to_owned(),
            claim_value: row.claim_value.to_owned(),
            issuer: owned(row.issuer),
            behavior: row.behavior.into(),
            existing_hash: row.existing_hash.cloned(),
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
    async fn record_agent_authentication(
        &self,
        agent: &PyRow,
    ) -> Result<Option<AgentIdentityFailure>, Error> {
        let agent = agent.clone();
        self.ask(|reply| Ask::RecordAgentAuthentication { agent, reply })
            .await
    }
    async fn registered_agent(&self, claim: &str) -> Result<Option<RegisteredAgent>, Error> {
        let claim = claim.to_owned();
        self.ask(|reply| Ask::RegisteredAgent { claim, reply })
            .await
    }
    async fn delegated_agent_team(
        &self,
        delegated: &DelegatedTeam<'_>,
    ) -> Result<Option<String>, Error> {
        let delegation = Box::new(Delegation {
            user_id: owned(delegated.user_id),
            agent_id: delegated.agent_id.to_owned(),
            team_id: owned(delegated.team_id),
            explicit_team: delegated.explicit_team,
            allowed_team_ids: delegated
                .allowed_team_ids
                .map(|ids| ids.iter().cloned().collect()),
        });
        self.ask(|reply| Ask::DelegatedAgentTeam { delegation, reply })
            .await
    }
    fn operator_hooks(&self) -> Option<&impl OperatorHooks> {
        Some(self)
    }
    async fn team_model_access(&self, team: &PyRow, model: &str) -> Result<ModelAccess, Error> {
        let (team, model) = (team.clone(), model.to_owned());
        self.ask(|reply| Ask::TeamModelAccess { team, model, reply })
            .await
    }
}
