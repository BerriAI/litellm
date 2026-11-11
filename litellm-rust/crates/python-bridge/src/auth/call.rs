use litellm_gateway_auth_core::{
    self as core, BudgetExceeded, Dependency, Error, Identity, Outcome, PublicDenial, Request, Unavailable,
    jwt::{
        self, AgentIdentityFailure, AgentMode, AliasRead, ClaimIdentity, Expected, IdentityRead, JwtDenial,
        JwtIdentity, JwtMapping, KeyMappingWrite, KeyOrigin, ManagedAgent, MappedJwt, MappedKey, ModelAccess, Org,
        Provisioning, RegisteredAgent, Team, TeamRead, User, UserRead,
    },
    budgets::{BudgetRow, MemberBudget, NamedBudget, RecordedBudget},
    key_budget::{self, KeyBudgetAlert, KeyBudgetRecord, MaxBudgetWarning},
    passthrough::PassthroughGrants,
    plan::Plan,
    key::{
        CachedTeam, KeyContext, KeyIdentity, KeyRecord, Keyed, Owner, TeamGrant,
    },
};
use litellm_host_python::{from_py, to_py};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use serde_json::Value;

use super::{
    authorize::AuthorizeFacts,
    host::{Ask, Failure, Finished, MembershipArg, PyRow, PyRows, PythonAuthHost},
    settings::AuthSettings,
    stepper::Stepper,
};

/// The request facts Python passes in, read from a frozen dataclass (`core.AuthRequest`).
#[derive(FromPyObject)]
pub(crate) struct RequestFacts {
    #[pyo3(attribute)]
    route: String,
    #[pyo3(attribute)]
    method: Option<String>,
    #[pyo3(attribute)]
    token: Option<String>,
    #[pyo3(attribute)]
    model: Option<String>,
    #[pyo3(attribute)]
    team_header: Option<String>,
    #[pyo3(attribute)]
    is_llm_api_route: bool,
    #[pyo3(attribute)]
    auth_enforced_pass_through: bool,
    #[pyo3(attribute)]
    forwarded_routes: Vec<String>,
    #[pyo3(attribute)]
    is_info_route: bool,
    #[pyo3(attribute)]
    now: f64,
    #[pyo3(attribute)]
    pass_through_open: bool,
    #[pyo3(attribute)]
    from_custom_auth: bool,
}

impl RequestFacts {
    fn request(&self) -> Request<'_> {
        Request {
            route: &self.route,
            method: self.method.as_deref(),
            token: self.token.as_deref(),
            model: self.model.as_deref(),
            team_header: self.team_header.as_deref(),
            is_llm_api_route: self.is_llm_api_route,
            auth_enforced_pass_through: self.auth_enforced_pass_through,
            forwarded_routes: &self.forwarded_routes,
            is_info_route: self.is_info_route,
            pass_through_open: self.pass_through_open,
            from_custom_auth: self.from_custom_auth,
            now: self.now,
        }
    }
}

/// The key fields the core decides on, read from `facts.KeyFacts`.
#[derive(FromPyObject)]
pub(crate) struct KeyFacts {
    #[pyo3(attribute)]
    token: Option<String>,
    #[pyo3(attribute)]
    blocked: bool,
    #[pyo3(attribute)]
    expires: Option<f64>,
    #[pyo3(attribute)]
    is_proxy_admin: bool,
    #[pyo3(attribute)]
    is_session_token: bool,
    #[pyo3(attribute)]
    user_id: Option<String>,
    #[pyo3(attribute)]
    team_id: Option<String>,
    #[pyo3(attribute)]
    team_object_permission_id: Option<String>,
    #[pyo3(attribute)]
    project_id: Option<String>,
    #[pyo3(attribute)]
    last_refreshed_at: Option<f64>,
    #[pyo3(attribute)]
    team_member_spend: Option<f64>,
    #[pyo3(attribute)]
    max_budget: Option<f64>,
    #[pyo3(attribute)]
    spend: Option<f64>,
    #[pyo3(attribute)]
    soft_budget: Option<f64>,
    #[pyo3(attribute)]
    key_alias: Option<String>,
    #[pyo3(attribute)]
    key_name: Option<String>,
    /// The lowest configured alert email percentage, unset when emails name no percentage.
    #[pyo3(attribute)]
    alert_email_lowest_percent: Option<f64>,
    /// Alert emails are configured, so the default warning share does not apply.
    #[pyo3(attribute)]
    alert_emails_configured: bool,
    /// `EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE`
    #[pyo3(attribute)]
    default_warning_share: f64,
    #[pyo3(attribute)]
    throttle_share: Option<f64>,
}

impl From<KeyFacts> for KeyRecord {
    fn from(facts: KeyFacts) -> Self {
        Self {
            token: facts.token,
            blocked: facts.blocked,
            expires: facts.expires,
            is_proxy_admin: facts.is_proxy_admin,
            is_session_token: facts.is_session_token,
            user_id: facts.user_id,
            team_id: facts.team_id,
            team_object_permission_id: facts.team_object_permission_id,
            project_id: facts.project_id,
            last_refreshed_at: facts.last_refreshed_at,
            team_member_spend: facts.team_member_spend,
            budget: KeyBudgetRecord {
                max_budget: facts.max_budget,
                spend: facts.spend,
                soft_budget: facts.soft_budget,
                key_alias: facts.key_alias,
                key_name: facts.key_name,
                warning: match facts.alert_emails_configured {
                    true => MaxBudgetWarning::Emails {
                        lowest_percent: facts.alert_email_lowest_percent,
                    },
                    false => MaxBudgetWarning::Default {
                        share: facts.default_warning_share,
                    },
                },
                throttle_share: facts.throttle_share,
            },
        }
    }
}

/// `litellm.BudgetExceededError`, read as is.
#[derive(FromPyObject)]
pub(crate) struct BudgetFacts {
    #[pyo3(attribute)]
    current_cost: f64,
    #[pyo3(attribute)]
    max_budget: f64,
    #[pyo3(attribute)]
    message: String,
    #[pyo3(attribute)]
    entity_type: Option<String>,
    #[pyo3(attribute)]
    entity_id: Option<String>,
}

impl From<BudgetFacts> for BudgetExceeded {
    fn from(budget: BudgetFacts) -> Self {
        Self {
            current_cost: budget.current_cost,
            max_budget: budget.max_budget,
            message: budget.message,
            entity_type: budget.entity_type,
            entity_id: budget.entity_id,
        }
    }
}

fn keyed(facts: KeyFacts, row: Py<PyAny>) -> Keyed<PyRow> {
    Keyed {
        record: facts.into(),
        row: PyRow::new(row),
    }
}

fn row(py: Python<'_>, row: &PyRow) -> Py<PyAny> {
    row.object(py)
}

fn rows(py: Python<'_>, row: Option<&PyRow>) -> Option<Py<PyAny>> {
    row.map(|row| row.object(py))
}

/// `facts.TeamFacts`
#[derive(FromPyObject)]
pub(crate) struct TeamFacts {
    #[pyo3(attribute)]
    team_id: String,
    #[pyo3(attribute)]
    models: Option<Vec<String>>,
    #[pyo3(attribute)]
    allowed_passthrough_routes: Option<Vec<String>>,
    #[pyo3(attribute)]
    denied_passthrough_routes: Vec<String>,
    #[pyo3(attribute)]
    members: Vec<String>,
}

fn team(facts: TeamFacts, row: Py<PyAny>) -> Team<PyRow> {
    Team {
        team_id: facts.team_id,
        models: facts.models,
        passthrough: PassthroughGrants {
            allowed: facts.allowed_passthrough_routes,
            denied: facts.denied_passthrough_routes,
        },
        members: facts.members,
        row: PyRow::new(row),
    }
}

/// A team read: the team, or `facts.TeamMissing` / `facts.AliasMissing` naming why there is none.
#[derive(FromPyObject)]
pub(crate) enum TeamAnswer {
    Found(TeamFacts),
    Missing(String),
}

/// `facts.UserFacts`
#[derive(FromPyObject)]
pub(crate) struct UserFacts {
    #[pyo3(attribute)]
    user_id: String,
    #[pyo3(attribute)]
    user_email: Option<String>,
    #[pyo3(attribute)]
    user_role: Option<String>,
    #[pyo3(attribute)]
    teams: Vec<String>,
    #[pyo3(attribute)]
    deactivated: bool,
}

/// `ManagedAgentContext`, read as is. The context itself is the row.
#[derive(FromPyObject)]
pub(crate) struct ManagedAgentFacts {
    #[pyo3(attribute)]
    mode: String,
    #[pyo3(attribute)]
    agent_id: String,
    #[pyo3(attribute)]
    user_id: Option<String>,
}

/// `AgentIdentityFailure`, read as is.
#[derive(FromPyObject)]
pub(crate) struct AgentFailureFacts {
    #[pyo3(attribute)]
    code: String,
    #[pyo3(attribute)]
    message: String,
}

/// `facts.RegisteredAgentFacts`
#[derive(FromPyObject)]
pub(crate) struct RegisteredAgentFacts {
    #[pyo3(attribute)]
    agent_id: String,
    #[pyo3(attribute)]
    identity_managed: bool,
    #[pyo3(attribute)]
    has_identity: bool,
    #[pyo3(attribute)]
    has_legacy_identity: bool,
}

/// The question Python answers next, or the call's result.
#[pyclass(frozen, name = "RustAuthStep", module = "litellm.rust_bridge._native")]
pub(crate) enum AuthStep {
    EndUser { token: String, resolved: Option<Py<PyAny>> },
    CachedKey { hash: String },
    UiKey { token: String },
    WithSessionGrants {
        key: Py<PyAny>,
        user: Py<PyAny>,
        team: Option<Py<PyAny>>,
        membership: Option<Py<PyAny>>,
    },
    EvictKey { hash: String },
    CachedTeam { team_id: String },
    WithTeam { key: Py<PyAny>, team: Py<PyAny> },
    DbKey { hash: String, end_user: Py<PyAny> },
    WithTempBudget { key: Py<PyAny> },
    NormalizeModel { key: Py<PyAny> },
    PrefetchAuthObjects { key: Py<PyAny>, end_user: Py<PyAny> },
    KeyOwner { user_id: String },
    KeySkipsBudgetChecks { key: Py<PyAny> },
    TeamMemberMaxBudget { user_id: String, team_id: String },
    KeyModelBudgets { key: Py<PyAny>, owner: Option<Py<PyAny>> },
    ThrottleKey { key: Py<PyAny>, share: f64 },
    KeyTeam { team_id: String },
    ObjectPermission { id: String },
    Project { project_id: String },
    GlobalProxySpend(),
    /// `kind` names a `KeyBudgetAlert`; `spend` and `max_budget` are set when it carries them.
    KeyBudgetAlert {
        key: Py<PyAny>,
        owner: Option<Py<PyAny>>,
        kind: String,
        spend: Option<f64>,
        max_budget: Option<f64>,
    },
    FetchJson { url: String },
    FetchUserinfo { endpoint: String, token: String },
    CachedJson { key: String },
    CacheJson { key: String, value: Py<PyAny>, ttl_seconds: f64 },
    ForgetJson { key: String },
    Team { team_id: String, upsert: bool },
    TeamByAlias { alias: String },
    GrantingTeam { team_id: String },
    LoadTeamMembership { user_id: String, team_id: String },
    TeamModelAccess { team: Py<PyAny>, model: String },
    User {
        user_id: String,
        user_email: Option<String>,
        sso_user_id: Option<String>,
        upsert: bool,
    },
    ResolveUserIdentity {
        user_id: String,
        user_email: Option<String>,
        sso_user_id: Option<String>,
        upsert: bool,
        team_id: Option<String>,
    },
    SetUserRole { user: Py<PyAny>, role: String },
    PatchTeamMembership { user: Py<PyAny>, add: Vec<String>, remove: Vec<String> },
    AddTeamMember { user_id: String, team_id: String },
    Org { org_id: String },
    OrgByAlias { alias: String },
    JwtEndUser { end_user_id: String },
    FinishJwt { identity: Py<PyAny> },
    KeyMappingHash { claim_field: String, claim_value: String, issuer: Option<String> },
    KeyByHash { hash: String },
    ReusableKeyHash { user_id: String, team_id: Option<String> },
    CreateKeyMapping {
        claim_field: String,
        claim_value: String,
        issuer: Option<String>,
        behavior: String,
        existing_hash: Option<String>,
        team_id: Option<String>,
        user_id: Option<String>,
        org_id: Option<String>,
        agent_id: Option<String>,
    },
    ManagedAgent { claims: Py<PyAny> },
    RecordAgentAuthentication { agent: Py<PyAny> },
    RegisteredAgent { claim: String },
    DelegatedAgentTeam {
        user_id: Option<String>,
        agent_id: String,
        team_id: Option<String>,
        explicit_team: bool,
        allowed_team_ids: Option<Vec<String>>,
    },
    CustomValidate { claims: Py<PyAny> },
    UnpricedModels(),
    TeamOrganization { team_id: String },
    AgentRequiresTraceId { agent_id: String },
    HasTraceId { headers_only: bool },
    MergeBudgetTags(),
    /// `membership_read` is unset when no check needed the team membership.
    CheckTeamModelAccess { membership_read: bool, membership: Option<Py<PyAny>> },
    CheckModelGrants(),
    CheckProject(),
    StampAccessGroups { membership_read: bool, membership: Option<Py<PyAny>> },
    /// `scope` is a `SpendCounter` name. `team_id` is set for a team member, `window` for a key or team window.
    Spend {
        scope: String,
        entity_id: String,
        team_id: Option<String>,
        window: Option<String>,
        recorded: f64,
        max_budget: f64,
    },
    OrganizationBudget { org_id: String },
    TagBudgets(),
    ModelAccessGroupBudgets { groups: Vec<String> },
    TeamMemberBudgetState { membership: Option<Py<PyAny>> },
    /// `kind` is a `BudgetAlert` name.
    BudgetAlert {
        kind: String,
        org_id: Option<String>,
        spend: Option<f64>,
        max_budget: Option<f64>,
    },
    CheckGuardrailModification(),
    CheckObjectAccess(),
    Done { outcome: Py<AuthOutcome> },
}

/// How a call ended, by entry point.
enum Ended {
    Outcome(Outcome<PyRows>),
    Verified(Result<jwt::Claims, Error>),
    Authorized(Result<JwtIdentity<PyRows>, Error>),
    Resolved(Result<ClaimIdentity<PyRow>, Error>),
    Mapped(Result<JwtMapping<PyRow>, Error>),
    Checked(Result<(), Error>),
}

/// What a call ended with. Python projects an identity into `UserAPIKeyAuth` and maps a denial to
/// its public error, each in one place (`core.project`, `core.raise_denial`).
#[pyclass(frozen, name = "RustAuthOutcome", module = "litellm.rust_bridge._native")]
pub(crate) struct AuthOutcome(Ended);

/// Which path authenticates the request, for the host's dispatch.
#[pyclass(frozen, eq, skip_from_py_object, name = "RustPlan", module = "litellm.rust_bridge._native")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanView {
    Public,
    OAuth2Introspection,
    OAuth2Proxy,
    Jwt,
    PassThroughOpen,
    Key,
}

impl From<Plan> for PlanView {
    fn from(plan: Plan) -> Self {
        match plan {
            Plan::Public => Self::Public,
            Plan::OAuth2Introspection => Self::OAuth2Introspection,
            Plan::OAuth2Proxy => Self::OAuth2Proxy,
            Plan::Jwt => Self::Jwt,
            Plan::PassThroughOpen => Self::PassThroughOpen,
            Plan::Key => Self::Key,
        }
    }
}

/// The path to run, or the license error for a path the proxy may not run.
#[derive(IntoPyObject)]
enum Planned {
    Plan(PlanView),
    Denial(DenialView),
}

/// The team values a virtual key carries, for `core.project`.
#[pyclass(frozen, name = "RustTeamGrant", module = "litellm.rust_bridge._native")]
pub(crate) enum TeamGrantView {
    NoTeam(),
    Loaded { team: Py<PyAny> },
    /// The key's own copy, with the object permission it names.
    KeyCopy { object_permission: Option<Py<PyAny>> },
}

/// The rows a virtual key passed its checks with, which `core.project` converts.
#[pyclass(frozen, get_all, name = "RustKeyContext", module = "litellm.rust_bridge._native")]
pub(crate) struct KeyContextView {
    key: Py<PyAny>,
    end_user: Py<PyAny>,
    owner: Option<Py<PyAny>>,
    /// A `RustTeamGrant` variant.
    team: Py<PyAny>,
    project: Option<Py<PyAny>>,
}

#[pymethods]
impl KeyContextView {
    #[new]
    #[pyo3(signature = (key, end_user, *, owner = None, team, project = None))]
    fn py_new(
        key: Py<PyAny>,
        end_user: Py<PyAny>,
        owner: Option<Py<PyAny>>,
        team: Bound<'_, TeamGrantView>,
        project: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            key,
            end_user,
            owner,
            team: team.into_any().unbind(),
            project,
        }
    }
}

impl KeyContextView {
    fn new(py: Python<'_>, context: &KeyContext<PyRows>) -> PyResult<Py<Self>> {
        let team = match &context.team {
            TeamGrant::NoTeam => TeamGrantView::NoTeam(),
            TeamGrant::Loaded(team) => TeamGrantView::Loaded { team: row(py, team) },
            TeamGrant::KeyCopy(permission) => TeamGrantView::KeyCopy {
                object_permission: rows(py, permission.as_ref()),
            },
        };
        Py::new(
            py,
            Self {
                key: row(py, &context.key),
                end_user: row(py, &context.end_user),
                owner: rows(py, context.owner.as_ref()),
                team: team.into_pyobject(py)?.into_any().unbind(),
                project: rows(py, context.project.as_ref()),
            },
        )
    }
}

/// Who the request authenticated as, for `core.project`.
#[pyclass(frozen, name = "RustIdentity", module = "litellm.rust_bridge._native")]
pub(crate) enum IdentityView {
    NoMasterKey { token: Option<String> },
    UserAuthRoute(),
    CachedAdmin { key: Py<PyAny>, end_user: Py<PyAny> },
    MasterKey { hash: String, end_user: Py<PyAny> },
    VirtualKey { context: Py<KeyContextView> },
    /// A JWT mapped onto a virtual key that passed the key checks. `auto_registered` keys were
    /// written for this JWT; `minted` ones were created for it and take its `org_id`. `user` is
    /// the user the policy loaded before an auto-registration.
    MappedJwtKey {
        claims: Py<PyAny>,
        end_user_id: Option<String>,
        user_email: Option<String>,
        auto_registered: bool,
        minted: bool,
        org_id: Option<String>,
        user: Option<Py<PyAny>>,
        context: Py<KeyContextView>,
    },
    /// Only in `RustAuthStep.FinishJwt`: the key auto-registered for a JWT, to finish a token from.
    AutoRegisteredJwtKey {
        claims: Py<PyAny>,
        end_user_id: Option<String>,
        user_email: Option<String>,
        minted: bool,
        org_id: Option<String>,
        user: Option<Py<PyAny>>,
        key: Py<PyAny>,
    },
    /// The token the host's `finish_jwt` built.
    FinishedJwt { token: Py<PyAny> },
    /// The database was unreachable and `allow_requests_on_db_unavailable` let the request through.
    DbUnavailable { message: String },
    /// A JWT's ids and the rows behind them: a proxy admin's outcome, `authorize_jwt`'s result, or
    /// the identity `RustAuthStep.FinishJwt` finishes.
    Jwt {
        claims: Py<PyAny>,
        is_proxy_admin: bool,
        user_id: Option<String>,
        user_email: Option<String>,
        team_id: Option<String>,
        org_id: Option<String>,
        end_user_id: Option<String>,
        agent_id: Option<String>,
        managed_agent: Option<Py<PyAny>>,
        user: Option<Py<PyAny>>,
        team: Option<Py<PyAny>>,
        membership: Option<Py<PyAny>>,
        org: Option<Py<PyAny>>,
        end_user: Option<Py<PyAny>>,
    },
}

fn origin(origin: &KeyOrigin) -> (bool, bool, Option<String>) {
    match origin {
        KeyOrigin::Mapping => (false, false, None),
        KeyOrigin::AutoRegistered { minted, org_id } => (true, *minted, org_id.clone()),
    }
}

impl IdentityView {
    fn jwt(py: Python<'_>, jwt: &JwtIdentity<PyRows>) -> PyResult<Self> {
        Ok(Self::Jwt {
            claims: to_py(py, &jwt.claims)?,
            is_proxy_admin: jwt.is_proxy_admin,
            user_id: jwt.user_id.clone(),
            user_email: jwt.user_email.clone(),
            team_id: jwt.team_id.clone(),
            org_id: jwt.org_id.clone(),
            end_user_id: jwt.end_user_id.clone(),
            agent_id: jwt.agent_id.clone(),
            managed_agent: jwt.managed_agent.as_ref().map(|agent| row(py, &agent.row)),
            user: rows(py, jwt.rows.user.as_ref()),
            team: rows(py, jwt.rows.team.as_ref()),
            membership: rows(py, jwt.rows.membership.as_ref()),
            org: rows(py, jwt.rows.org.as_ref()),
            end_user: rows(py, jwt.rows.end_user.as_ref()),
        })
    }

    fn mapped(py: Python<'_>, mapped: &MappedJwt<PyRow>, context: &KeyContext<PyRows>) -> PyResult<Self> {
        let (auto_registered, minted, org_id) = origin(&mapped.origin);
        Ok(Self::MappedJwtKey {
            claims: to_py(py, &mapped.claims)?,
            end_user_id: mapped.end_user_id.clone(),
            user_email: mapped.user_email.clone(),
            auto_registered,
            minted,
            org_id,
            user: rows(py, mapped.user.as_ref()),
            context: KeyContextView::new(py, context)?,
        })
    }

    fn auto_registered(py: Python<'_>, mapped: &MappedJwt<PyRow>, key: &PyRow) -> PyResult<Self> {
        let (_, minted, org_id) = origin(&mapped.origin);
        Ok(Self::AutoRegisteredJwtKey {
            claims: to_py(py, &mapped.claims)?,
            end_user_id: mapped.end_user_id.clone(),
            user_email: mapped.user_email.clone(),
            minted,
            org_id,
            user: rows(py, mapped.user.as_ref()),
            key: row(py, key),
        })
    }

    fn new(py: Python<'_>, identity: &Identity<PyRows>) -> PyResult<Self> {
        Ok(match identity {
            Identity::Key(KeyIdentity::NoMasterKey { token }) => Self::NoMasterKey { token: token.clone() },
            Identity::Key(KeyIdentity::UserAuthRoute) => Self::UserAuthRoute(),
            Identity::Key(KeyIdentity::CachedAdmin { key, end_user }) => Self::CachedAdmin {
                key: row(py, key),
                end_user: row(py, end_user),
            },
            Identity::Key(KeyIdentity::MasterKey { hash, end_user }) => Self::MasterKey {
                hash: hash.clone(),
                end_user: row(py, end_user),
            },
            Identity::Key(KeyIdentity::VirtualKey(context)) => Self::VirtualKey {
                context: KeyContextView::new(py, context)?,
            },
            Identity::MappedJwtKey(mapped, context) => Self::mapped(py, mapped, context)?,
            Identity::AdminJwt(jwt) => Self::jwt(py, jwt)?,
            Identity::FinishedJwt(token) => Self::FinishedJwt { token: row(py, token) },
            Identity::DbUnavailable(unavailable) => Self::DbUnavailable {
                message: unavailable.message.clone(),
            },
        })
    }
}

/// `JWTIdentity`
#[pyclass(frozen, get_all, name = "RustClaimIdentity", module = "litellm.rust_bridge._native")]
pub(crate) struct ClaimIdentityView {
    user_id: Option<String>,
    agent_id: Option<String>,
    user: Option<Py<PyAny>>,
}

/// What `_resolve_jwt_to_virtual_key` found. A `Key` is the one the host kept from `key_by_hash`.
#[pyclass(frozen, eq, skip_from_py_object, name = "RustJwtMapping", module = "litellm.rust_bridge._native")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MappingView {
    Key,
    Pending,
    Unmapped,
}

/// A denial as the exception the proxy raises for it; `core.raise_denial` only constructs it.
#[pyclass(frozen, name = "RustDenial", module = "litellm.rust_bridge._native")]
pub(crate) enum DenialView {
    Proxy {
        status: u16,
        message: String,
        error_type: String,
        param: Option<String>,
    },
    Http {
        status: u16,
        detail: String,
    },
    InvalidVirtualKey {
        status: u16,
        detail: String,
    },
    ModelAccess {
        status: u16,
        detail: String,
        internal_message: String,
    },
    Plain {
        message: String,
    },
    BudgetExceeded {
        current_cost: f64,
        max_budget: f64,
        message: String,
        entity_type: Option<String>,
        entity_id: Option<String>,
    },
    TeamNotFound {
        team_id: String,
    },
}

impl From<PublicDenial> for DenialView {
    fn from(denial: PublicDenial) -> Self {
        match denial {
            PublicDenial::Proxy {
                status,
                message,
                error_type,
                param,
            } => Self::Proxy {
                status,
                message,
                error_type: <&'static str>::from(error_type).to_owned(),
                param,
            },
            PublicDenial::Http { status, detail } => Self::Http { status, detail },
            PublicDenial::InvalidVirtualKey { status, detail } => Self::InvalidVirtualKey { status, detail },
            PublicDenial::ModelAccess {
                status,
                detail,
                internal_message,
            } => Self::ModelAccess {
                status,
                detail,
                internal_message,
            },
            PublicDenial::Plain { message } => Self::Plain { message },
            PublicDenial::BudgetExceeded(budget) => Self::BudgetExceeded {
                current_cost: budget.current_cost,
                max_budget: budget.max_budget,
                message: budget.message,
                entity_type: budget.entity_type,
                entity_id: budget.entity_id,
            },
            PublicDenial::TeamNotFound { team_id } => Self::TeamNotFound { team_id },
        }
    }
}

#[pymethods]
impl AuthOutcome {
    #[getter]
    fn denied(&self) -> bool {
        self.public_denial().is_some()
    }

    #[getter]
    fn identity(&self, py: Python<'_>) -> PyResult<Option<IdentityView>> {
        match &self.0 {
            Ended::Outcome(Outcome::Authenticated(identity)) => IdentityView::new(py, identity).map(Some),
            Ended::Authorized(Ok(identity)) => IdentityView::jwt(py, identity).map(Some),
            _ => Ok(None),
        }
    }

    #[getter]
    fn denial(&self) -> Option<DenialView> {
        self.public_denial().map(DenialView::from)
    }

    /// The key the request had resolved before it was denied, which labels the failure.
    #[getter]
    fn denied_key(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        match &self.0 {
            Ended::Outcome(Outcome::Denied(_, key)) => rows(py, key.as_ref()),
            _ => None,
        }
    }

    /// The verified claims, for `verify_jwt` and `verify_jwt_assertion`.
    #[getter]
    fn claims(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.0 {
            Ended::Verified(Ok(claims)) => to_py(py, &claims.0).map(Some),
            _ => Ok(None),
        }
    }

    /// Whether an assertion was refused for what it is rather than for an unreachable JWKS.
    #[getter]
    fn assertion_rejected(&self) -> bool {
        matches!(
            &self.0,
            Ended::Verified(Err(Error::Jwt(JwtDenial::Verify(jwt::VerifyDenial::Assertion(_)))))
        )
    }

    #[getter]
    fn claim_identity(&self, py: Python<'_>) -> Option<ClaimIdentityView> {
        match &self.0 {
            Ended::Resolved(Ok(identity)) => Some(ClaimIdentityView {
                user_id: identity.user_id.clone(),
                agent_id: identity.agent_id.clone(),
                user: rows(py, identity.user.as_ref()),
            }),
            _ => None,
        }
    }

    #[getter]
    fn mapping(&self) -> Option<MappingView> {
        match &self.0 {
            Ended::Mapped(Ok(JwtMapping::Key(_))) => Some(MappingView::Key),
            Ended::Mapped(Ok(JwtMapping::Pending)) => Some(MappingView::Pending),
            Ended::Mapped(Ok(JwtMapping::Unmapped)) => Some(MappingView::Unmapped),
            _ => None,
        }
    }

    /// The key a `Key` mapping names.
    #[getter]
    fn mapped_key(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        match &self.0 {
            Ended::Mapped(Ok(JwtMapping::Key(mapped))) => Some(row(py, &mapped.key.row)),
            _ => None,
        }
    }
}

impl AuthOutcome {
    fn public_denial(&self) -> Option<PublicDenial> {
        match &self.0 {
            Ended::Outcome(Outcome::Denied(error, _))
            | Ended::Verified(Err(error))
            | Ended::Authorized(Err(error))
            | Ended::Resolved(Err(error))
            | Ended::Mapped(Err(error))
            | Ended::Checked(Err(error)) => error.public(),
            _ => None,
        }
    }
}

fn role(role: litellm_config::LitellmRole) -> String {
    <&str>::from(role).to_owned()
}

impl AuthStep {
    fn asked(py: Python<'_>, ask: &Ask) -> PyResult<Self> {
        Ok(match ask {
            Ask::EndUser { token, resolved, .. } => Self::EndUser {
                token: token.clone(),
                resolved: rows(py, resolved.as_ref()),
            },
            Ask::CachedKey { hash, .. } => Self::CachedKey { hash: hash.clone() },
            Ask::UiKey { token, .. } => Self::UiKey { token: token.clone() },
            Ask::WithSessionGrants {
                key, user, team, membership, ..
            } => Self::WithSessionGrants {
                key: row(py, key),
                user: row(py, user),
                team: rows(py, team.as_ref()),
                membership: rows(py, membership.as_ref()),
            },
            Ask::EvictKey { hash, .. } => Self::EvictKey { hash: hash.clone() },
            Ask::CachedTeam { team_id, .. } => Self::CachedTeam {
                team_id: team_id.clone(),
            },
            Ask::WithTeam { key, team, .. } => Self::WithTeam {
                key: row(py, key),
                team: row(py, team),
            },
            Ask::DbKey { hash, end_user, .. } => Self::DbKey {
                hash: hash.clone(),
                end_user: row(py, end_user),
            },
            Ask::WithTempBudget { key, .. } => Self::WithTempBudget { key: row(py, key) },
            Ask::NormalizeModel { key, .. } => Self::NormalizeModel { key: row(py, key) },
            Ask::PrefetchAuthObjects { key, end_user, .. } => Self::PrefetchAuthObjects {
                key: row(py, key),
                end_user: row(py, end_user),
            },
            Ask::KeyOwner { user_id, .. } => Self::KeyOwner {
                user_id: user_id.clone(),
            },
            Ask::KeySkipsBudgetChecks { key, .. } => Self::KeySkipsBudgetChecks { key: row(py, key) },
            Ask::TeamMemberMaxBudget { user_id, team_id, .. } => Self::TeamMemberMaxBudget {
                user_id: user_id.clone(),
                team_id: team_id.clone(),
            },
            Ask::ThrottleKey { key, share, .. } => Self::ThrottleKey {
                key: row(py, key),
                share: *share,
            },
            Ask::KeyModelBudgets { key, owner, .. } => Self::KeyModelBudgets {
                key: row(py, key),
                owner: rows(py, owner.as_ref()),
            },
            Ask::KeyTeam { team_id, .. } => Self::KeyTeam {
                team_id: team_id.clone(),
            },
            Ask::ObjectPermission { id, .. } => Self::ObjectPermission { id: id.clone() },
            Ask::Project { project_id, .. } => Self::Project {
                project_id: project_id.clone(),
            },
            Ask::GlobalProxySpend(_) => Self::GlobalProxySpend(),
            Ask::KeyBudgetAlert { key, owner, alert, .. } => {
                let (spend, max_budget) = match *alert {
                    KeyBudgetAlert::TeamMember { spend, max_budget } => (Some(spend), Some(max_budget)),
                    KeyBudgetAlert::Proxy { spend } | KeyBudgetAlert::TokenBudget { spend } => (Some(spend), None),
                    KeyBudgetAlert::MaxBudgetWarning | KeyBudgetAlert::SoftBudget => (None, None),
                };
                Self::KeyBudgetAlert {
                    key: row(py, key),
                    owner: rows(py, owner.as_ref()),
                    kind: <&str>::from(*alert).to_owned(),
                    spend,
                    max_budget,
                }
            }
            Ask::FetchJson { url, .. } => Self::FetchJson { url: url.clone() },
            Ask::FetchUserinfo { endpoint, token, .. } => Self::FetchUserinfo {
                endpoint: endpoint.clone(),
                token: token.clone(),
            },
            Ask::CachedJson { key, .. } => Self::CachedJson { key: key.clone() },
            Ask::CacheJson { key, value, ttl, .. } => Self::CacheJson {
                key: key.clone(),
                value: to_py(py, &**value)?,
                ttl_seconds: ttl.as_secs_f64(),
            },
            Ask::ForgetJson { key, .. } => Self::ForgetJson { key: key.clone() },
            Ask::Team { team_id, upsert, .. } => Self::Team {
                team_id: team_id.clone(),
                upsert: *upsert,
            },
            Ask::TeamByAlias { alias, .. } => Self::TeamByAlias { alias: alias.clone() },
            Ask::GrantingTeam { team_id, .. } => Self::GrantingTeam {
                team_id: team_id.clone(),
            },
            Ask::LoadTeamMembership { user_id, team_id, .. } => Self::LoadTeamMembership {
                user_id: user_id.clone(),
                team_id: team_id.clone(),
            },
            Ask::TeamModelAccess { team, model, .. } => Self::TeamModelAccess {
                team: row(py, team),
                model: model.clone(),
            },
            Ask::User { lookup, .. } => Self::User {
                user_id: lookup.user_id.clone(),
                user_email: lookup.user_email.clone(),
                sso_user_id: lookup.sso_user_id.clone(),
                upsert: lookup.upsert,
            },
            Ask::ResolveUserIdentity { lookup, team_id, .. } => Self::ResolveUserIdentity {
                user_id: lookup.user_id.clone(),
                user_email: lookup.user_email.clone(),
                sso_user_id: lookup.sso_user_id.clone(),
                upsert: lookup.upsert,
                team_id: team_id.clone(),
            },
            Ask::SetUserRole { user, role: new_role, .. } => Self::SetUserRole {
                user: row(py, user),
                role: role(*new_role),
            },
            Ask::PatchTeamMembership { user, add, remove, .. } => Self::PatchTeamMembership {
                user: row(py, user),
                add: add.clone(),
                remove: remove.clone(),
            },
            Ask::AddTeamMember { user_id, team_id, .. } => Self::AddTeamMember {
                user_id: user_id.clone(),
                team_id: team_id.clone(),
            },
            Ask::Org { org_id, .. } => Self::Org { org_id: org_id.clone() },
            Ask::OrgByAlias { alias, .. } => Self::OrgByAlias { alias: alias.clone() },
            Ask::JwtEndUser { end_user_id, .. } => Self::JwtEndUser {
                end_user_id: end_user_id.clone(),
            },
            Ask::FinishJwt { finished, .. } => Self::FinishJwt {
                identity: match finished {
                    Finished::Identity(identity) => IdentityView::jwt(py, identity)?,
                    Finished::AutoRegisteredKey(jwt, key) => IdentityView::auto_registered(py, jwt, key)?,
                }
                .into_pyobject(py)?
                .into_any()
                .unbind(),
            },
            Ask::KeyMappingHash {
                claim_field,
                claim_value,
                issuer,
                ..
            } => Self::KeyMappingHash {
                claim_field: claim_field.clone(),
                claim_value: claim_value.clone(),
                issuer: issuer.clone(),
            },
            Ask::KeyByHash { hash, .. } => Self::KeyByHash { hash: hash.clone() },
            Ask::ReusableKeyHash { user_id, team_id, .. } => Self::ReusableKeyHash {
                user_id: user_id.clone(),
                team_id: team_id.clone(),
            },
            Ask::CreateKeyMapping { row, .. } => Self::CreateKeyMapping {
                claim_field: row.claim_field.clone(),
                claim_value: row.claim_value.clone(),
                issuer: row.issuer.clone(),
                behavior: row.behavior.to_owned(),
                existing_hash: row.existing_hash.clone(),
                team_id: row.team_id.clone(),
                user_id: row.user_id.clone(),
                org_id: row.org_id.clone(),
                agent_id: row.agent_id.clone(),
            },
            Ask::ManagedAgent { claims, .. } => Self::ManagedAgent {
                claims: to_py(py, claims)?,
            },
            Ask::RecordAgentAuthentication { agent, .. } => Self::RecordAgentAuthentication { agent: row(py, agent) },
            Ask::RegisteredAgent { claim, .. } => Self::RegisteredAgent { claim: claim.clone() },
            Ask::DelegatedAgentTeam { delegation, .. } => Self::DelegatedAgentTeam {
                user_id: delegation.user_id.clone(),
                agent_id: delegation.agent_id.clone(),
                team_id: delegation.team_id.clone(),
                explicit_team: delegation.explicit_team,
                allowed_team_ids: delegation.allowed_team_ids.clone(),
            },
            Ask::CustomValidate { claims, .. } => Self::CustomValidate {
                claims: to_py(py, claims)?,
            },
            Ask::UnpricedModels(_) => Self::UnpricedModels(),
            Ask::AgentRequiresTraceId { agent_id, .. } => Self::AgentRequiresTraceId {
                agent_id: agent_id.clone(),
            },
            Ask::HasTraceId { headers_only, .. } => Self::HasTraceId {
                headers_only: *headers_only,
            },
            Ask::MergeBudgetTags(_) => Self::MergeBudgetTags(),
            Ask::CheckTeamModelAccess { membership, .. } => {
                let (membership_read, membership) = membership_view(py, membership);
                Self::CheckTeamModelAccess {
                    membership_read,
                    membership,
                }
            }
            Ask::CheckModelGrants(_) => Self::CheckModelGrants(),
            Ask::CheckProject(_) => Self::CheckProject(),
            Ask::StampAccessGroups { membership, .. } => {
                let (membership_read, membership) = membership_view(py, membership);
                Self::StampAccessGroups {
                    membership_read,
                    membership,
                }
            }
            Ask::Spend { counter, .. } => Self::Spend {
                scope: counter.scope.to_owned(),
                entity_id: counter.entity_id.clone(),
                team_id: counter.team_id.clone(),
                window: counter.window.clone(),
                recorded: counter.recorded,
                max_budget: counter.max_budget,
            },
            Ask::OrganizationBudget { org_id, .. } => Self::OrganizationBudget { org_id: org_id.clone() },
            Ask::TagBudgets(_) => Self::TagBudgets(),
            Ask::ModelAccessGroupBudgets { groups, .. } => Self::ModelAccessGroupBudgets { groups: groups.clone() },
            Ask::TeamMemberBudgetState { membership, .. } => Self::TeamMemberBudgetState {
                membership: rows(py, membership.as_ref()),
            },
            Ask::BudgetAlert { alert, .. } => Self::BudgetAlert {
                kind: alert.kind.to_owned(),
                org_id: alert.org_id.clone(),
                spend: alert.spend,
                max_budget: alert.max_budget,
            },
            Ask::CheckGuardrailModification(_) => Self::CheckGuardrailModification(),
            Ask::TeamOrganization { team_id, .. } => Self::TeamOrganization {
                team_id: team_id.clone(),
            },
            Ask::CheckObjectAccess(_) => Self::CheckObjectAccess(),
        })
    }
}

fn membership_view(py: Python<'_>, membership: &MembershipArg) -> (bool, Option<Py<PyAny>>) {
    match membership {
        MembershipArg::Unread => (false, None),
        MembershipArg::Read(row) => (true, rows(py, row.as_ref())),
    }
}

type Called = Result<AuthOutcome, Error>;

/// A denial ends the call with an outcome. A host failure makes `resume` raise instead, so
/// `core.drive` re-raises the host's own exception.
fn ended<T>(result: Result<T, Error>, end: impl FnOnce(Result<T, Error>) -> Ended) -> Called {
    match result {
        Err(error) if error.is_host_failure() => Err(error),
        result => Ok(AuthOutcome(end(result))),
    }
}

/// One authentication call, driven from Python by `core.drive`: `resume()` returns the next
/// `RustAuthStep`, Python answers it with the method for its reply type, and resumes again until
/// `Done`. A step Python could not answer is reported with `unavailable` or `fail`.
#[pyclass(frozen, name = "RustAuth", module = "litellm.rust_bridge._native")]
pub(crate) struct AuthCall {
    stepper: Stepper<Ask, Called>,
    failure: Failure,
}

fn call<F>(body: impl FnOnce(PythonAuthHost) -> F) -> AuthCall
where
    F: Future<Output = Called> + Send + 'static,
{
    let failure = Failure::default();
    let host_failure = failure.clone();
    AuthCall {
        stepper: Stepper::new(move |co| {
            body(PythonAuthHost {
                co,
                failure: host_failure,
            })
        }),
        failure,
    }
}

fn missing(kind: &str, value: &str) -> PyErr {
    PyValueError::new_err(format!("auth host answered {kind} with an unknown reason: {value}"))
}

fn found_row(row: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    row.ok_or_else(|| PyValueError::new_err("auth host answered a found team without its row"))
}

fn team_read(answer: TeamAnswer, row: Option<Py<PyAny>>) -> PyResult<TeamRead<PyRow>> {
    match answer {
        TeamAnswer::Found(facts) => Ok(TeamRead::Found(team(facts, found_row(row)?))),
        TeamAnswer::Missing(reason) => match reason.as_str() {
            "absent" => Ok(TeamRead::Absent),
            "unreadable" => Ok(TeamRead::Unreadable),
            other => Err(missing("a team read", other)),
        },
    }
}

fn alias_read(answer: TeamAnswer, row: Option<Py<PyAny>>) -> PyResult<AliasRead<PyRow>> {
    match answer {
        TeamAnswer::Found(facts) => Ok(AliasRead::Found(team(facts, found_row(row)?))),
        TeamAnswer::Missing(reason) => match reason.as_str() {
            "absent" => Ok(AliasRead::Absent),
            "ambiguous" => Ok(AliasRead::Ambiguous),
            "no_db" => Ok(AliasRead::NoDb),
            other => Err(missing("a team alias read", other)),
        },
    }
}

fn agent_mode(mode: &str) -> PyResult<AgentMode> {
    match mode {
        "autonomous" => Ok(AgentMode::Autonomous),
        "delegated" => Ok(AgentMode::Delegated),
        other => Err(PyValueError::new_err(format!("unknown managed agent mode: {other}"))),
    }
}

#[pymethods]
impl AuthCall {
    #[staticmethod]
    fn plan(request: RequestFacts, settings: &AuthSettings) -> Planned {
        match core::plan(&request.request(), &settings.0) {
            Ok(plan) => Planned::Plan(plan.into()),
            Err(premium) => Planned::Denial(premium.public().into()),
        }
    }

    #[staticmethod]
    /// `key` is the credential the key path authenticates: the token the plan was made on, a key custom
    /// auth handed back, or one a pass-through endpoint read from its own header.
    fn authenticate_key(request: RequestFacts, key: Option<String>, settings: &AuthSettings) -> AuthCall {
        let settings = settings.0.clone();
        call(move |host| async move {
            core::authenticate_key(&host, &request.request(), key.as_deref(), &settings)
                .await
                .map(|o| AuthOutcome(Ended::Outcome(o)))
        })
    }

    #[staticmethod]
    fn oauth2_introspection_applies(request: RequestFacts, settings: &AuthSettings) -> bool {
        core::oauth2_introspection_applies(&request.request(), &settings.0)
    }

    #[staticmethod]
    fn authenticate_jwt(request: RequestFacts, settings: &AuthSettings) -> AuthCall {
        let settings = settings.0.clone();
        call(move |host| async move {
            core::authenticate_jwt(&host, &request.request(), &settings).await.map(|o| AuthOutcome(Ended::Outcome(o)))
        })
    }

    /// `JWTAuthManager.authorize_jwt`: `provisioning` applies the configured user and team upserts
    /// the way `auth_builder` does; the MCP token flows pass none.
    #[staticmethod]
    #[pyo3(signature = (request, settings, *, provisioning))]
    fn authorize_jwt(request: RequestFacts, settings: &AuthSettings, provisioning: bool) -> AuthCall {
        let settings = settings.0.clone();
        let provisioning = provisioning.then(|| Provisioning::from(&settings.jwt));
        call(move |host| async move {
            ended(core::authorize_jwt(&host, &request.request(), &settings, provisioning).await, Ended::Authorized)
        })
    }

    #[staticmethod]
    fn resolve_jwt_identity(request: RequestFacts, settings: &AuthSettings) -> AuthCall {
        let settings = settings.0.clone();
        call(move |host| async move {
            ended(core::resolve_jwt_identity(&host, &request.request(), &settings).await, Ended::Resolved)
        })
    }

    #[staticmethod]
    fn resolve_jwt_mapping(request: RequestFacts, settings: &AuthSettings) -> AuthCall {
        let settings = settings.0.clone();
        call(move |host| async move {
            ended(core::resolve_jwt_mapping(&host, &request.request(), &settings).await, Ended::Mapped)
        })
    }

    /// `common_checks` for a request the host already authenticated.
    #[staticmethod]
    fn authorize(facts: AuthorizeFacts) -> AuthCall {
        call(move |host| async move {
            let memberships = facts.memberships();
            ended(core::authorize(&host, &facts.request(memberships.as_deref())).await, Ended::Checked)
        })
    }

    /// `virtual_key_max_budget_check` for a key the caller already authenticated.
    #[staticmethod]
    #[pyo3(signature = (facts, key, owner = None))]
    fn key_max_budget(facts: KeyFacts, key: Py<PyAny>, owner: Option<Py<PyAny>>) -> AuthCall {
        let key = keyed(facts, key);
        let owner = owner.map(PyRow::new);
        call(move |host| async move {
            ended(
                key_budget::key_max_budget(&host, &key.record, &key.row, owner.as_ref()).await,
                Ended::Checked,
            )
        })
    }

    #[staticmethod]
    fn verify_jwt(token: String, now: f64, settings: &AuthSettings) -> AuthCall {
        let jwt = settings.0.jwt.clone();
        call(move |host| async move {
            ended(jwt::verify(&host, &token, &jwt, now).await, Ended::Verified)
        })
    }

    #[staticmethod]
    #[pyo3(signature = (token, now, settings, *, audience, required_claims))]
    fn verify_jwt_assertion(
        token: String,
        now: f64,
        settings: &AuthSettings,
        audience: Vec<String>,
        required_claims: Vec<String>,
    ) -> AuthCall {
        let jwt = settings.0.jwt.clone();
        let expected = Expected {
            audience,
            required_claims,
        };
        call(move |host| async move {
            ended(jwt::verify_assertion(&host, &token, &expected, &jwt, now).await, Ended::Verified)
        })
    }

    fn resume(&self, py: Python<'_>) -> PyResult<AuthStep> {
        match self.stepper.resume(|ask| AuthStep::asked(py, ask))? {
            Ok(step) => step,
            Err(Ok(outcome)) => Ok(AuthStep::Done {
                outcome: Py::new(py, outcome)?,
            }),
            Err(Err(_)) => Err(PyRuntimeError::new_err("auth call lost its host")),
        }
    }

    /// The host step raised. The core sees a host error: it skips the steps whose failure the proxy
    /// tolerated and otherwise ends the call, and `core.drive` re-raises the exception.
    fn fail(&self) -> PyResult<()> {
        self.stepper.answer("a failure", |ask| {
            drop(ask);
            Ok(())
        })
    }

    /// The host step could not reach `dependency`. `transient` outages are the ones
    /// `allow_requests_on_db_unavailable` serves through; `message` is the 503 the caller sees.
    #[pyo3(signature = (*, dependency, transient, message))]
    fn unavailable(&self, dependency: &str, transient: bool, message: String) -> PyResult<()> {
        let dependency: Dependency = dependency
            .parse()
            .map_err(|_| PyValueError::new_err(format!("unknown dependency: {dependency}")))?;
        self.stepper.answer("an unavailable dependency", |ask| {
            self.failure.set(Error::Unavailable(Unavailable {
                dependency,
                transient,
                message,
            }));
            drop(ask);
            Ok(())
        })
    }

    fn done(&self) -> PyResult<()> {
        self.stepper.answer("a completed step", |ask| match ask {
            Ask::EvictKey { reply, .. }
            | Ask::KeyBudgetAlert { reply, .. }
            | Ask::ThrottleKey { reply, .. }
            | Ask::CacheJson { reply, .. }
            | Ask::ForgetJson { reply, .. }
            | Ask::NormalizeModel { reply, .. }
            | Ask::PrefetchAuthObjects { reply, .. }
            | Ask::AddTeamMember { reply, .. }
            | Ask::MergeBudgetTags(reply)
            | Ask::CheckTeamModelAccess { reply, .. }
            | Ask::CheckModelGrants(reply)
            | Ask::CheckProject(reply)
            | Ask::BudgetAlert { reply, .. }
            | Ask::CheckGuardrailModification(reply)
            | Ask::CheckObjectAccess(reply) => {
                reply.send(());
                Ok(())
            }
            Ask::KeyModelBudgets { reply, .. } => {
                reply.send(Ok(()));
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A budget step, or the end user's budget, refused the request with this
    /// `litellm.BudgetExceededError`.
    fn budget_exceeded(&self, error: BudgetFacts) -> PyResult<()> {
        let exceeded = BudgetExceeded::from(error);
        self.stepper.answer("an exceeded budget", |ask| match ask {
            Ask::KeyModelBudgets { reply, .. } => {
                reply.send(Err(exceeded));
                Ok(())
            }
            Ask::EndUser { reply, .. } | Ask::FinishJwt { reply, .. } => {
                reply.send(Err(exceeded));
                Ok(())
            }
            other => Err(other),
        })
    }

    fn found(&self, found: bool) -> PyResult<()> {
        self.stepper.answer("a yes or no", |ask| match ask {
            Ask::CustomValidate { reply, .. }
            | Ask::KeySkipsBudgetChecks { reply, .. }
            | Ask::AgentRequiresTraceId { reply, .. }
            | Ask::HasTraceId { reply, .. } => {
                reply.send(found);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A row the host built: the end user, a key with team values or prepared for the checks, a
    /// user with its role or teams synced, or a finished JWT token.
    fn row(&self, row: Py<PyAny>) -> PyResult<()> {
        let row = PyRow::new(row);
        self.stepper.answer("a row", |ask| match ask {
            Ask::EndUser { reply, .. } | Ask::FinishJwt { reply, .. } => {
                reply.send(Ok(row));
                Ok(())
            }
            Ask::WithTeam { reply, .. }
            | Ask::WithTempBudget { reply, .. }
            | Ask::SetUserRole { reply, .. }
            | Ask::PatchTeamMembership { reply, .. }
            | Ask::JwtEndUser { reply, .. } => {
                reply.send(row);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A row read by id, or nothing when there is none.
    #[pyo3(signature = (row = None))]
    fn found_row(&self, row: Option<Py<PyAny>>) -> PyResult<()> {
        let row = row.map(PyRow::new);
        self.stepper.answer("a row or nothing", |ask| match ask {
            Ask::KeyTeam { reply, .. }
            | Ask::ObjectPermission { reply, .. }
            | Ask::Project { reply, .. }
            | Ask::LoadTeamMembership { reply, .. } => {
                reply.send(row);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A key and the fields the core decides on, or no key.
    #[pyo3(signature = (facts = None, row = None))]
    fn key(&self, facts: Option<KeyFacts>, row: Option<Py<PyAny>>) -> PyResult<()> {
        let key = match (facts, row) {
            (Some(facts), Some(row)) => Some(keyed(facts, row)),
            (None, None) => None,
            _ => return Err(PyValueError::new_err("a key is answered with both its facts and its row")),
        };
        self.stepper.answer("a key", |ask| match (ask, key) {
            (Ask::CachedKey { reply, .. } | Ask::UiKey { reply, .. } | Ask::DbKey { reply, .. }, key) => {
                reply.send(key);
                Ok(())
            }
            (Ask::WithSessionGrants { reply, .. }, Some(key)) => {
                reply.send(key);
                Ok(())
            }
            (other, _) => Err(other),
        })
    }

    #[pyo3(signature = (row = None, *, deactivated = false))]
    fn key_owner(&self, row: Option<Py<PyAny>>, deactivated: bool) -> PyResult<()> {
        let owner = row.map(|row| Owner {
            deactivated,
            row: PyRow::new(row),
        });
        self.stepper.answer("a key owner", |ask| match ask {
            Ask::KeyOwner { reply, .. } => {
                reply.send(owner);
                Ok(())
            }
            other => Err(other),
        })
    }

    #[pyo3(signature = (row = None, *, refreshed_at = None))]
    fn cached_team(&self, row: Option<Py<PyAny>>, refreshed_at: Option<f64>) -> PyResult<()> {
        let team = row.map(|row| CachedTeam {
            refreshed_at,
            row: PyRow::new(row),
        });
        self.stepper.answer("a cached team", |ask| match ask {
            Ask::CachedTeam { reply, .. } => {
                reply.send(team);
                Ok(())
            }
            other => Err(other),
        })
    }

    #[pyo3(signature = (value = None))]
    fn json(&self, value: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let value: Option<Value> = value.filter(|value| !value.is_none()).map(|value| from_py(&value)).transpose()?;
        self.stepper.answer("a JSON document", |ask| match ask {
            Ask::FetchJson { reply, .. } | Ask::FetchUserinfo { reply, .. } | Ask::CachedJson { reply, .. } => {
                reply.send(value);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A `get_team_object` read: the team and its row, or a `facts.TeamMissing`.
    #[pyo3(signature = (read, row = None))]
    fn team(&self, read: TeamAnswer, row: Option<Py<PyAny>>) -> PyResult<()> {
        let read = team_read(read, row)?;
        self.stepper.answer("a team", |ask| match ask {
            Ask::Team { reply, .. } | Ask::GrantingTeam { reply, .. } => {
                reply.send(read);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A `get_team_object_by_alias` read: the team and its row, or a `facts.AliasMissing`.
    #[pyo3(signature = (read, row = None))]
    fn team_alias(&self, read: TeamAnswer, row: Option<Py<PyAny>>) -> PyResult<()> {
        let read = alias_read(read, row)?;
        self.stepper.answer("a team alias", |ask| match ask {
            Ask::TeamByAlias { reply, .. } => {
                reply.send(read);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// `can_team_access_model`, which the host still decides.
    #[pyo3(signature = (*, allowed))]
    fn model_access(&self, allowed: bool) -> PyResult<()> {
        let access = if allowed { ModelAccess::Allowed } else { ModelAccess::Denied };
        self.stepper.answer("a team model access", |ask| match ask {
            Ask::TeamModelAccess { reply, .. } => {
                reply.send(access);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A `get_user_object` read with its row; `None` is its `UserNotFoundError`. A
    /// `resolve_identity` read also names the user's membership of the team it was asked about.
    #[pyo3(signature = (user = None, row = None, *, membership = None))]
    fn user(&self, user: Option<UserFacts>, row: Option<Py<PyAny>>, membership: Option<Py<PyAny>>) -> PyResult<()> {
        let read = match (user, row) {
            (Some(user), Some(row)) => UserRead::Found(User {
                user_id: user.user_id,
                user_email: user.user_email,
                user_role: user.user_role,
                teams: user.teams,
                deactivated: user.deactivated,
                row: PyRow::new(row),
            }),
            (None, None) => UserRead::Absent,
            _ => return Err(PyValueError::new_err("a user is answered with both its facts and its row")),
        };
        let membership = membership.map(PyRow::new);
        self.stepper.answer("a user", |ask| match (ask, membership) {
            (Ask::User { reply, .. }, None) => {
                reply.send(read);
                Ok(())
            }
            (Ask::ResolveUserIdentity { reply, .. }, membership) => {
                reply.send(IdentityRead { user: read, membership });
                Ok(())
            }
            (other, _) => Err(other),
        })
    }

    #[pyo3(signature = (org_id = None, row = None))]
    fn org(&self, org_id: Option<String>, row: Option<Py<PyAny>>) -> PyResult<()> {
        let org = match (org_id, row) {
            (Some(organization_id), Some(row)) => Some(Org {
                organization_id,
                row: PyRow::new(row),
            }),
            (None, None) => None,
            _ => return Err(PyValueError::new_err("an org is answered with both its id and its row")),
        };
        self.stepper.answer("an org", |ask| match ask {
            Ask::Org { reply, .. } | Ask::OrgByAlias { reply, .. } => {
                reply.send(org);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A counter's live spend.
    fn spend(&self, spend: f64) -> PyResult<()> {
        self.stepper.answer("a spend", |ask| match ask {
            Ask::Spend { reply, .. } => {
                reply.send(spend);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// The organization's ceiling and recorded spend, or nothing when it sets none.
    #[pyo3(signature = (max_budget = None, spend = 0.0))]
    fn organization_budget(&self, max_budget: Option<f64>, spend: f64) -> PyResult<()> {
        let budget = max_budget.map(|max_budget| RecordedBudget { max_budget, spend });
        self.stepper.answer("an organization budget", |ask| match ask {
            Ask::OrganizationBudget { reply, .. } => {
                reply.send(budget);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// Tag or model access group budgets, as `(name, max_budget, spend)`.
    fn named_budgets(&self, budgets: Vec<(String, f64, f64)>) -> PyResult<()> {
        let budgets = budgets
            .into_iter()
            .map(|(name, max_budget, spend)| NamedBudget { name, max_budget, spend })
            .collect();
        self.stepper.answer("named budgets", |ask| match ask {
            Ask::TagBudgets(reply) | Ask::ModelAccessGroupBudgets { reply, .. } => {
                reply.send(budgets);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// The team member's recorded spend, their own budget row as `(max_budget, temp_budget_increase,
    /// temp_budget_expiry)` with the expiry in epoch seconds, and the team's default member ceiling.
    #[pyo3(signature = (*, spend, budget = None, team_default = None))]
    fn member_budget(
        &self,
        spend: f64,
        budget: Option<(Option<f64>, Option<f64>, Option<f64>)>,
        team_default: Option<f64>,
    ) -> PyResult<()> {
        let member = MemberBudget {
            spend,
            budget: budget.map(|(max_budget, temp_budget_increase, temp_budget_expiry)| BudgetRow {
                max_budget,
                temp_budget_increase,
                temp_budget_expiry,
            }),
            team_default,
        };
        self.stepper.answer("a team member budget", |ask| match ask {
            Ask::TeamMemberBudgetState { reply, .. } => {
                reply.send(member);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A spend or a budget, or nothing when there is none.
    #[pyo3(signature = (value = None))]
    fn amount(&self, value: Option<f64>) -> PyResult<()> {
        self.stepper.answer("an amount", |ask| match ask {
            Ask::TeamMemberMaxBudget { reply, .. } | Ask::GlobalProxySpend(reply) => {
                reply.send(value);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// Model or access group names.
    fn names(&self, names: Vec<String>) -> PyResult<()> {
        self.stepper.answer("names", |ask| match ask {
            Ask::UnpricedModels(reply) | Ask::StampAccessGroups { reply, .. } => {
                reply.send(names);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A key hash or team id, or nothing.
    #[pyo3(signature = (value = None))]
    fn text(&self, value: Option<String>) -> PyResult<()> {
        self.stepper.answer("a key hash or team id", |ask| match ask {
            Ask::KeyMappingHash { reply, .. }
            | Ask::ReusableKeyHash { reply, .. }
            | Ask::DelegatedAgentTeam { reply, .. }
            | Ask::TeamOrganization { reply, .. } => {
                reply.send(value);
                Ok(())
            }
            other => Err(other),
        })
    }

    #[pyo3(signature = (facts = None, row = None, *, user_email = None))]
    fn mapped_key(&self, facts: Option<KeyFacts>, row: Option<Py<PyAny>>, user_email: Option<String>) -> PyResult<()> {
        let mapped = match (facts, row) {
            (Some(facts), Some(row)) => Some(MappedKey {
                key: keyed(facts, row),
                user_email,
            }),
            (None, None) => None,
            _ => return Err(PyValueError::new_err("a mapped key is answered with both its facts and its row")),
        };
        self.stepper.answer("a mapped key", |ask| match ask {
            Ask::KeyByHash { reply, .. } => {
                reply.send(mapped);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// The mapping write: the hash it maps to, `conflict` when a concurrent request won the
    /// unique race, and no hash when the winner's mapping was removed before it could be read.
    #[pyo3(signature = (*, token_hash, conflict))]
    fn key_mapping(&self, token_hash: Option<String>, conflict: bool) -> PyResult<()> {
        let write = match (token_hash, conflict) {
            (Some(hash), false) => KeyMappingWrite::Created(hash),
            (Some(hash), true) => KeyMappingWrite::Conflict(hash),
            (None, true) => KeyMappingWrite::ConflictVanished,
            (None, false) => return Err(PyValueError::new_err("a created key mapping names its key")),
        };
        self.stepper.answer("a key mapping write", |ask| match ask {
            Ask::CreateKeyMapping { reply, .. } => {
                reply.send(write);
                Ok(())
            }
            other => Err(other),
        })
    }

    #[pyo3(signature = (agent = None))]
    fn managed_agent(&self, agent: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let agent = agent
            .map(|agent| {
                let facts: ManagedAgentFacts = agent.extract()?;
                agent_mode(&facts.mode).map(|mode| ManagedAgent {
                    mode,
                    agent_id: facts.agent_id,
                    user_id: facts.user_id,
                    row: PyRow::new(agent.unbind()),
                })
            })
            .transpose()?;
        self.stepper.answer("a managed agent", |ask| match ask {
            Ask::ManagedAgent { reply, .. } => {
                reply.send(agent);
                Ok(())
            }
            other => Err(other),
        })
    }

    #[pyo3(signature = (failure = None))]
    fn agent_failure(&self, failure: Option<AgentFailureFacts>) -> PyResult<()> {
        let failure = failure.map(|failure| AgentIdentityFailure {
            policy_unavailable: failure.code == "policy_unavailable",
            message: failure.message,
        });
        self.stepper.answer("an agent authentication record", |ask| match ask {
            Ask::RecordAgentAuthentication { reply, .. } => {
                reply.send(failure);
                Ok(())
            }
            other => Err(other),
        })
    }

    #[pyo3(signature = (agent = None))]
    fn registered_agent(&self, agent: Option<RegisteredAgentFacts>) -> PyResult<()> {
        let agent = agent.map(|agent| RegisteredAgent {
            agent_id: agent.agent_id,
            identity_managed: agent.identity_managed,
            has_identity: agent.has_identity,
            has_legacy_identity: agent.has_legacy_identity,
        });
        self.stepper.answer("a registered agent", |ask| match ask {
            Ask::RegisteredAgent { reply, .. } => {
                reply.send(agent);
                Ok(())
            }
            other => Err(other),
        })
    }
}

/// `virtual_key_max_budget_alert_check`'s condition.
#[pyfunction]
pub(crate) fn auth_key_max_budget_warning_due(facts: KeyFacts) -> bool {
    key_budget::max_budget_warning_due(&KeyRecord::from(facts).budget)
}

/// `virtual_key_soft_budget_check`'s condition.
#[pyfunction]
pub(crate) fn auth_key_soft_budget_due(facts: KeyFacts) -> bool {
    key_budget::soft_budget_due(&KeyRecord::from(facts).budget)
}
