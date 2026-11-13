use http::Method;
use jiff::Timestamp;
use litellm_gateway_auth_core::{
    self as core, BudgetExceeded, Error, Identity, Outcome, PublicDenial, Request,
    credential::jwt::{
        self, AgentIdentityFailure, AgentMode, AliasRead, ClaimIdentity, Expected, IdentityRead,
        JwtDenial, JwtIdentity, JwtMapping, KeyMappingWrite, KeyOrigin, ManagedAgent, MappedJwt,
        MappedKey, ModelAccess, Org, Provisioning, RegisteredAgent, Team, TeamRead, User, UserRead,
    },
    credential::key::{
        CachedTeam, KeyContext, KeyHash, KeyIdentity, KeyRecord, Keyed, Owner, TeamGrant, Token,
    },
    plan::Plan,
    routes::passthrough::{ForwardedRoutes, PassthroughGrants},
};
use litellm_host_python::{from_py, to_py};
use litellm_python_compat::repr::float_repr;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use serde_json::Value;

use super::{
    convert::{py_method, py_optional_timestamp, py_timestamp, timestamp},
    host::{Ask, Finished, PyRow, PyRows, PythonAuthHost},
    settings::AuthSettings,
    stepper::Stepper,
};

fn presented_token(token: &Bound<'_, PyAny>) -> PyResult<Option<Token>> {
    Ok(token.extract::<Option<String>>()?.map(Token::new))
}

/// The request facts Python passes in, read from a frozen dataclass (`core.AuthRequest`).
#[derive(FromPyObject)]
pub(crate) struct RequestFacts {
    #[pyo3(attribute)]
    route: String,
    #[pyo3(attribute, from_py_with = py_method)]
    method: Option<Method>,
    #[pyo3(attribute, from_py_with = presented_token)]
    token: Option<Token>,
    #[pyo3(attribute)]
    model: Option<String>,
    #[pyo3(attribute)]
    end_user_id: Option<String>,
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
    #[pyo3(attribute, from_py_with = py_timestamp)]
    now: Timestamp,
    #[pyo3(attribute)]
    pass_through_open: bool,
    #[pyo3(attribute)]
    from_custom_auth: bool,
}

impl RequestFacts {
    fn request(&self) -> Request<'_> {
        Request {
            route: &self.route,
            method: self.method.as_ref(),
            token: self.token.as_ref(),
            model: self.model.as_deref(),
            end_user_id: self.end_user_id.as_deref(),
            team_header: self.team_header.as_deref(),
            is_llm_api_route: self.is_llm_api_route,
            auth_enforced_pass_through: self.auth_enforced_pass_through,
            forwarded_routes: ForwardedRoutes(&self.forwarded_routes),
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
    #[pyo3(attribute, from_py_with = py_optional_timestamp)]
    expires: Option<Timestamp>,
    #[pyo3(attribute)]
    is_proxy_admin: bool,
    #[pyo3(attribute)]
    user_id: Option<String>,
    #[pyo3(attribute)]
    team_id: Option<String>,
    #[pyo3(attribute)]
    team_object_permission_id: Option<String>,
    #[pyo3(attribute)]
    project_id: Option<String>,
    #[pyo3(attribute)]
    end_user_id: Option<String>,
    #[pyo3(attribute, from_py_with = py_optional_timestamp)]
    last_refreshed_at: Option<Timestamp>,
}

impl From<KeyFacts> for KeyRecord {
    fn from(facts: KeyFacts) -> Self {
        Self {
            token: facts.token.map(KeyHash::new),
            blocked: facts.blocked,
            expires: facts.expires,
            is_proxy_admin: facts.is_proxy_admin,
            user_id: facts.user_id,
            team_id: facts.team_id,
            team_object_permission_id: facts.team_object_permission_id,
            project_id: facts.project_id,
            end_user_id: facts.end_user_id,
            last_refreshed_at: facts.last_refreshed_at,
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
            subject: None,
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
    ValidatedEndUserId {
        end_user_id: String,
    },
    PrefetchIdentity {
        token: Option<String>,
        end_user_id: Option<String>,
    },
    EndUser {
        end_user_id: Option<String>,
    },
    CachedKey {
        hash: String,
    },
    UiKey {
        token: String,
    },
    EvictKey {
        hash: String,
    },
    CachedTeam {
        team_id: String,
    },
    WithTeam {
        key: Py<PyAny>,
        team: Py<PyAny>,
    },
    DbKey {
        hash: String,
        end_user: Py<PyAny>,
    },
    WithTempBudget {
        key: Py<PyAny>,
    },
    NormalizeModel {
        key: Py<PyAny>,
    },
    PrefetchAuthObjects {
        key: Py<PyAny>,
        end_user: Py<PyAny>,
    },
    KeyOwner {
        user_id: String,
    },
    KeySkipsBudgetChecks {
        key: Py<PyAny>,
    },
    TeamMemberBudget {
        key: Py<PyAny>,
        owner: Option<Py<PyAny>>,
    },
    KeyBudgets {
        key: Py<PyAny>,
        owner: Option<Py<PyAny>>,
        is_llm_api_route: bool,
    },
    KeyTeam {
        team_id: String,
    },
    ObjectPermission {
        id: String,
    },
    Project {
        project_id: String,
    },
    ProxyBudgetAlert {
        key: Py<PyAny>,
    },
    FetchJson {
        url: String,
    },
    FetchUserinfo {
        endpoint: String,
        token: String,
    },
    CachedJson {
        key: String,
    },
    CacheJson {
        key: String,
        value: Py<PyAny>,
        ttl_seconds: f64,
    },
    ForgetJson {
        key: String,
    },
    Team {
        team_id: String,
        upsert: bool,
    },
    TeamByAlias {
        alias: String,
    },
    GrantingTeam {
        team_id: String,
    },
    LoadTeamMembership {
        user_id: String,
        team_id: String,
    },
    TeamModelAccess {
        team: Py<PyAny>,
        model: String,
    },
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
    SetUserRole {
        user: Py<PyAny>,
        role: String,
    },
    PatchTeamMembership {
        user: Py<PyAny>,
        add: Vec<String>,
        remove: Vec<String>,
    },
    AddTeamMember {
        user_id: String,
        team_id: String,
    },
    Org {
        org_id: String,
    },
    OrgByAlias {
        alias: String,
    },
    JwtEndUser {
        end_user_id: String,
    },
    /// A `RustIdentity.Jwt` to finish a token from.
    FinishJwt {
        identity: Py<PyAny>,
    },
    /// The key auto-registered for a JWT, to finish a token from. `org_id` and `user` are those
    /// of `RustIdentity.AutoRegisteredJwtKey`.
    FinishAutoRegisteredJwt {
        claims: Py<PyAny>,
        end_user_id: Option<String>,
        user_email: Option<String>,
        org_id: Option<String>,
        user: Option<Py<PyAny>>,
        key: Py<PyAny>,
    },
    KeyMappingHash {
        claim_field: String,
        claim_value: String,
        issuer: Option<String>,
    },
    KeyByHash {
        hash: String,
    },
    ReusableKeyHash {
        user_id: String,
        team_id: Option<String>,
    },
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
    ManagedAgent {
        claims: Py<PyAny>,
    },
    RecordAgentAuthentication {
        agent: Py<PyAny>,
    },
    RegisteredAgent {
        claim: String,
    },
    DelegatedAgentTeam {
        user_id: Option<String>,
        agent_id: String,
        team_id: Option<String>,
        explicit_team: bool,
        allowed_team_ids: Option<Vec<String>>,
    },
    CustomValidate {
        claims: Py<PyAny>,
    },
    Done {
        outcome: Py<AuthOutcome>,
    },
}

/// How a call ended, by entry point.
enum Ended {
    Outcome(Outcome<PyRows>),
    Verified(Result<jwt::Claims, Error>),
    Authorized(Result<JwtIdentity<PyRows>, Error>),
    Resolved(Result<ClaimIdentity<PyRow>, Error>),
    Mapped(Result<JwtMapping<PyRow>, Error>),
}

/// What a call ended with. Python projects an identity into `UserAPIKeyAuth` and maps a denial to
/// its public error, each in one place (`core.project`, `core.raise_denial`).
#[pyclass(
    frozen,
    name = "RustAuthOutcome",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct AuthOutcome(Ended);

/// Which path authenticates the request, for the host's dispatch.
#[pyclass(
    frozen,
    eq,
    skip_from_py_object,
    name = "RustPlan",
    module = "litellm.rust_bridge._native"
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanView {
    Public,
    OAuth2Introspection,
    OAuth2Proxy,
    JwtWithOperatorCode,
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
            Plan::JwtWithOperatorCode => Self::JwtWithOperatorCode,
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
    Loaded {
        team: Py<PyAny>,
    },
    /// The key's own copy, with the object permission it names.
    KeyCopy {
        object_permission: Option<Py<PyAny>>,
    },
}

/// The rows a virtual key passed its checks with, which `core.project` converts.
#[pyclass(
    frozen,
    get_all,
    name = "RustKeyContext",
    module = "litellm.rust_bridge._native"
)]
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
            TeamGrant::Loaded(team) => TeamGrantView::Loaded {
                team: row(py, team),
            },
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
    NoMasterKey {
        token: Option<String>,
    },
    UserAuthRoute(),
    CachedAdmin {
        key: Py<PyAny>,
        end_user: Py<PyAny>,
    },
    MasterKey {
        hash: String,
        end_user: Py<PyAny>,
    },
    VirtualKey {
        context: Py<KeyContextView>,
    },
    /// A JWT mapped onto an existing virtual key that passed the key checks.
    MappedJwtKey {
        claims: Py<PyAny>,
        end_user_id: Option<String>,
        user_email: Option<String>,
        context: Py<KeyContextView>,
    },
    /// A JWT mapped onto a key written for it that passed the key checks. `org_id` is the JWT's
    /// and is set only for a key created for it. `user` is the user the policy loaded before the
    /// auto-registration.
    AutoRegisteredJwtKey {
        claims: Py<PyAny>,
        end_user_id: Option<String>,
        user_email: Option<String>,
        org_id: Option<String>,
        user: Option<Py<PyAny>>,
        context: Py<KeyContextView>,
    },
    /// The token the host's `finish_jwt` built.
    FinishedJwt {
        token: Py<PyAny>,
    },
    /// A JWT's ids and the rows behind them: a proxy admin's outcome or `authorize_jwt`'s result.
    /// `RustAuthStep.FinishJwt` carries one too, for the host to finish.
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

/// The JWT's `org_id`, which only a key created for the JWT takes.
fn minted_org_id(origin: &KeyOrigin) -> Option<String> {
    match origin {
        KeyOrigin::AutoRegistered {
            minted: true,
            org_id,
        } => org_id.clone(),
        KeyOrigin::Mapping | KeyOrigin::AutoRegistered { minted: false, .. } => None,
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

    fn mapped(
        py: Python<'_>,
        mapped: &MappedJwt<PyRow>,
        context: &KeyContext<PyRows>,
    ) -> PyResult<Self> {
        let claims = to_py(py, &mapped.claims)?;
        let end_user_id = mapped.end_user_id.clone();
        let user_email = mapped.user_email.clone();
        let context = KeyContextView::new(py, context)?;
        Ok(match &mapped.origin {
            KeyOrigin::Mapping => Self::MappedJwtKey {
                claims,
                end_user_id,
                user_email,
                context,
            },
            KeyOrigin::AutoRegistered { .. } => Self::AutoRegisteredJwtKey {
                claims,
                end_user_id,
                user_email,
                org_id: minted_org_id(&mapped.origin),
                user: rows(py, mapped.user.as_ref()),
                context,
            },
        })
    }

    fn new(py: Python<'_>, identity: &Identity<PyRows>) -> PyResult<Self> {
        Ok(match identity {
            Identity::Key(KeyIdentity::NoMasterKey { token }) => Self::NoMasterKey {
                token: token.as_ref().map(|token| token.expose().to_owned()),
            },
            Identity::Key(KeyIdentity::UserAuthRoute) => Self::UserAuthRoute(),
            Identity::Key(KeyIdentity::CachedAdmin { key, end_user }) => Self::CachedAdmin {
                key: row(py, key),
                end_user: row(py, end_user),
            },
            Identity::Key(KeyIdentity::MasterKey { hash, end_user }) => Self::MasterKey {
                hash: hash.as_str().to_owned(),
                end_user: row(py, end_user),
            },
            Identity::Key(KeyIdentity::VirtualKey(context)) => Self::VirtualKey {
                context: KeyContextView::new(py, context)?,
            },
            Identity::MappedJwtKey(mapped, context) => Self::mapped(py, mapped, context)?,
            Identity::AdminJwt(jwt) => Self::jwt(py, jwt)?,
            Identity::FinishedJwt(token) => Self::FinishedJwt {
                token: row(py, token),
            },
        })
    }
}

/// `JWTIdentity`
#[pyclass(
    frozen,
    get_all,
    name = "RustClaimIdentity",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct ClaimIdentityView {
    user_id: Option<String>,
    agent_id: Option<String>,
    user: Option<Py<PyAny>>,
}

/// What `_resolve_jwt_to_virtual_key` found. A `Key` is the one the host kept from `key_by_hash`.
#[pyclass(
    frozen,
    eq,
    skip_from_py_object,
    name = "RustJwtMapping",
    module = "litellm.rust_bridge._native"
)]
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
            PublicDenial::InvalidVirtualKey { status, detail } => {
                Self::InvalidVirtualKey { status, detail }
            }
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
                message: match &budget.subject {
                    Some(subject) => format!(
                        "Budget has been exceeded! {subject} Current cost: {}, Max budget: {}",
                        float_repr(budget.current_cost),
                        float_repr(budget.max_budget)
                    ),
                    None => budget.message,
                },
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
            Ended::Outcome(Outcome::Authenticated(identity)) => {
                IdentityView::new(py, identity).map(Some)
            }
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
            Ended::Verified(Err(Error::Jwt(JwtDenial::Verify(
                jwt::VerifyDenial::Assertion(_)
            ))))
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
            | Ended::Mapped(Err(error)) => error.public(),
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
            Ask::ValidatedEndUserId { end_user_id, .. } => Self::ValidatedEndUserId {
                end_user_id: end_user_id.clone(),
            },
            Ask::PrefetchIdentity {
                token, end_user_id, ..
            } => Self::PrefetchIdentity {
                token: token.as_ref().map(|token| token.expose().to_owned()),
                end_user_id: end_user_id.clone(),
            },
            Ask::EndUser { end_user_id, .. } => Self::EndUser {
                end_user_id: end_user_id.clone(),
            },
            Ask::CachedKey { hash, .. } => Self::CachedKey {
                hash: hash.as_str().to_owned(),
            },
            Ask::UiKey { token, .. } => Self::UiKey {
                token: token.expose().to_owned(),
            },
            Ask::EvictKey { hash, .. } => Self::EvictKey {
                hash: hash.as_str().to_owned(),
            },
            Ask::CachedTeam { team_id, .. } => Self::CachedTeam {
                team_id: team_id.clone(),
            },
            Ask::WithTeam { key, team, .. } => Self::WithTeam {
                key: row(py, key),
                team: row(py, team),
            },
            Ask::DbKey { hash, end_user, .. } => Self::DbKey {
                hash: hash.as_str().to_owned(),
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
            Ask::KeySkipsBudgetChecks { key, .. } => {
                Self::KeySkipsBudgetChecks { key: row(py, key) }
            }
            Ask::TeamMemberBudget { key, owner, .. } => Self::TeamMemberBudget {
                key: row(py, key),
                owner: rows(py, owner.as_ref()),
            },
            Ask::KeyBudgets {
                key,
                owner,
                is_llm_api_route,
                ..
            } => Self::KeyBudgets {
                key: row(py, key),
                owner: rows(py, owner.as_ref()),
                is_llm_api_route: *is_llm_api_route,
            },
            Ask::KeyTeam { team_id, .. } => Self::KeyTeam {
                team_id: team_id.clone(),
            },
            Ask::ObjectPermission { id, .. } => Self::ObjectPermission { id: id.clone() },
            Ask::Project { project_id, .. } => Self::Project {
                project_id: project_id.clone(),
            },
            Ask::ProxyBudgetAlert { key, .. } => Self::ProxyBudgetAlert { key: row(py, key) },
            Ask::FetchJson { url, .. } => Self::FetchJson { url: url.clone() },
            Ask::FetchUserinfo {
                endpoint, token, ..
            } => Self::FetchUserinfo {
                endpoint: endpoint.clone(),
                token: token.expose().to_owned(),
            },
            Ask::CachedJson { key, .. } => Self::CachedJson { key: key.clone() },
            Ask::CacheJson {
                key, value, ttl, ..
            } => Self::CacheJson {
                key: key.clone(),
                value: to_py(py, &**value)?,
                ttl_seconds: ttl.as_secs_f64(),
            },
            Ask::ForgetJson { key, .. } => Self::ForgetJson { key: key.clone() },
            Ask::Team {
                team_id, upsert, ..
            } => Self::Team {
                team_id: team_id.clone(),
                upsert: *upsert,
            },
            Ask::TeamByAlias { alias, .. } => Self::TeamByAlias {
                alias: alias.clone(),
            },
            Ask::GrantingTeam { team_id, .. } => Self::GrantingTeam {
                team_id: team_id.clone(),
            },
            Ask::LoadTeamMembership {
                user_id, team_id, ..
            } => Self::LoadTeamMembership {
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
            Ask::ResolveUserIdentity {
                lookup, team_id, ..
            } => Self::ResolveUserIdentity {
                user_id: lookup.user_id.clone(),
                user_email: lookup.user_email.clone(),
                sso_user_id: lookup.sso_user_id.clone(),
                upsert: lookup.upsert,
                team_id: team_id.clone(),
            },
            Ask::SetUserRole {
                user,
                role: new_role,
                ..
            } => Self::SetUserRole {
                user: row(py, user),
                role: role(*new_role),
            },
            Ask::PatchTeamMembership {
                user, add, remove, ..
            } => Self::PatchTeamMembership {
                user: row(py, user),
                add: add.clone(),
                remove: remove.clone(),
            },
            Ask::AddTeamMember {
                user_id, team_id, ..
            } => Self::AddTeamMember {
                user_id: user_id.clone(),
                team_id: team_id.clone(),
            },
            Ask::Org { org_id, .. } => Self::Org {
                org_id: org_id.clone(),
            },
            Ask::OrgByAlias { alias, .. } => Self::OrgByAlias {
                alias: alias.clone(),
            },
            Ask::JwtEndUser { end_user_id, .. } => Self::JwtEndUser {
                end_user_id: end_user_id.clone(),
            },
            Ask::FinishJwt {
                finished: Finished::Identity(identity),
                ..
            } => Self::FinishJwt {
                identity: IdentityView::jwt(py, identity)?
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            },
            Ask::FinishJwt {
                finished: Finished::AutoRegisteredKey(jwt, key),
                ..
            } => Self::FinishAutoRegisteredJwt {
                claims: to_py(py, &jwt.claims)?,
                end_user_id: jwt.end_user_id.clone(),
                user_email: jwt.user_email.clone(),
                org_id: minted_org_id(&jwt.origin),
                user: rows(py, jwt.user.as_ref()),
                key: row(py, key),
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
            Ask::KeyByHash { hash, .. } => Self::KeyByHash {
                hash: hash.as_str().to_owned(),
            },
            Ask::ReusableKeyHash {
                user_id, team_id, ..
            } => Self::ReusableKeyHash {
                user_id: user_id.clone(),
                team_id: team_id.clone(),
            },
            Ask::CreateKeyMapping { row, .. } => Self::CreateKeyMapping {
                claim_field: row.claim_field.clone(),
                claim_value: row.claim_value.clone(),
                issuer: row.issuer.clone(),
                behavior: row.behavior.to_owned(),
                existing_hash: row
                    .existing_hash
                    .as_ref()
                    .map(|hash| hash.as_str().to_owned()),
                team_id: row.team_id.clone(),
                user_id: row.user_id.clone(),
                org_id: row.org_id.clone(),
                agent_id: row.agent_id.clone(),
            },
            Ask::ManagedAgent { claims, .. } => Self::ManagedAgent {
                claims: to_py(py, claims)?,
            },
            Ask::RecordAgentAuthentication { agent, .. } => Self::RecordAgentAuthentication {
                agent: row(py, agent),
            },
            Ask::RegisteredAgent { claim, .. } => Self::RegisteredAgent {
                claim: claim.clone(),
            },
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
        })
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
/// `Done`. A step Python could not answer is reported with `fail`.
#[pyclass(frozen, name = "RustAuth", module = "litellm.rust_bridge._native")]
pub(crate) struct AuthCall {
    stepper: Stepper<Ask, Called>,
}

fn call<F>(body: impl FnOnce(PythonAuthHost) -> F) -> AuthCall
where
    F: Future<Output = Called> + Send + 'static,
{
    AuthCall {
        stepper: Stepper::new(move |co| body(PythonAuthHost { co })),
    }
}

fn missing(kind: &str, value: &str) -> PyErr {
    PyValueError::new_err(format!(
        "auth host answered {kind} with an unknown reason: {value}"
    ))
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
        other => Err(PyValueError::new_err(format!(
            "unknown managed agent mode: {other}"
        ))),
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
    fn authenticate_key(
        request: RequestFacts,
        key: Option<String>,
        settings: &AuthSettings,
    ) -> AuthCall {
        let (settings, key) = (settings.0.clone(), key.map(Token::new));
        call(move |host| async move {
            core::authenticate_key(&host, &request.request(), key.as_ref(), &settings)
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
            core::authenticate_jwt(&host, &request.request(), &settings)
                .await
                .map(|o| AuthOutcome(Ended::Outcome(o)))
        })
    }

    /// `JWTAuthManager.authorize_jwt`: `provisioning` applies the configured user and team upserts
    /// the way `auth_builder` does; the MCP token flows pass none.
    #[staticmethod]
    #[pyo3(signature = (request, settings, *, provisioning))]
    fn authorize_jwt(
        request: RequestFacts,
        settings: &AuthSettings,
        provisioning: bool,
    ) -> AuthCall {
        let settings = settings.0.clone();
        let provisioning = provisioning.then(|| Provisioning::from(&settings.jwt));
        call(move |host| async move {
            ended(
                core::authorize_jwt(&host, &request.request(), &settings, provisioning).await,
                Ended::Authorized,
            )
        })
    }

    #[staticmethod]
    fn resolve_jwt_identity(request: RequestFacts, settings: &AuthSettings) -> AuthCall {
        let settings = settings.0.clone();
        call(move |host| async move {
            ended(
                core::resolve_jwt_identity(&host, &request.request(), &settings).await,
                Ended::Resolved,
            )
        })
    }

    #[staticmethod]
    fn resolve_jwt_mapping(request: RequestFacts, settings: &AuthSettings) -> AuthCall {
        let settings = settings.0.clone();
        call(move |host| async move {
            ended(
                core::resolve_jwt_mapping(&host, &request.request(), &settings).await,
                Ended::Mapped,
            )
        })
    }

    #[staticmethod]
    fn verify_jwt(token: String, now: f64, settings: &AuthSettings) -> AuthCall {
        let (jwt, token) = (settings.0.jwt.clone(), Token::new(token));
        call(move |host| async move {
            ended(
                jwt::verify(&host, &token, &jwt, timestamp(now)).await,
                Ended::Verified,
            )
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
        let (jwt, token) = (settings.0.jwt.clone(), Token::new(token));
        let expected = Expected {
            audience,
            required_claims,
        };
        call(move |host| async move {
            ended(
                jwt::verify_assertion(&host, &token, &expected, &jwt, timestamp(now)).await,
                Ended::Verified,
            )
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

    fn done(&self) -> PyResult<()> {
        self.stepper.answer("a completed step", |ask| match ask {
            Ask::EvictKey { reply, .. }
            | Ask::ProxyBudgetAlert { reply, .. }
            | Ask::CacheJson { reply, .. }
            | Ask::ForgetJson { reply, .. }
            | Ask::NormalizeModel { reply, .. }
            | Ask::PrefetchIdentity { reply, .. }
            | Ask::PrefetchAuthObjects { reply, .. }
            | Ask::AddTeamMember { reply, .. } => {
                reply.send(());
                Ok(())
            }
            Ask::TeamMemberBudget { reply, .. } | Ask::KeyBudgets { reply, .. } => {
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
            Ask::TeamMemberBudget { reply, .. } | Ask::KeyBudgets { reply, .. } => {
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
            Ask::CustomValidate { reply, .. } | Ask::KeySkipsBudgetChecks { reply, .. } => {
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
            _ => {
                return Err(PyValueError::new_err(
                    "a key is answered with both its facts and its row",
                ));
            }
        };
        self.stepper.answer("a key", |ask| match (ask, key) {
            (
                Ask::CachedKey { reply, .. } | Ask::UiKey { reply, .. } | Ask::DbKey { reply, .. },
                key,
            ) => {
                reply.send(key);
                Ok(())
            }
            (Ask::WithTempBudget { reply, .. }, Some(key)) => {
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
            refreshed_at: refreshed_at.map(timestamp),
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
        let value: Option<Value> = value
            .filter(|value| !value.is_none())
            .map(|value| from_py(&value))
            .transpose()?;
        self.stepper.answer("a JSON document", |ask| match ask {
            Ask::FetchJson { reply, .. }
            | Ask::FetchUserinfo { reply, .. }
            | Ask::CachedJson { reply, .. } => {
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
        let access = if allowed {
            ModelAccess::Allowed
        } else {
            ModelAccess::Denied
        };
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
    fn user(
        &self,
        user: Option<UserFacts>,
        row: Option<Py<PyAny>>,
        membership: Option<Py<PyAny>>,
    ) -> PyResult<()> {
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
            _ => {
                return Err(PyValueError::new_err(
                    "a user is answered with both its facts and its row",
                ));
            }
        };
        let membership = membership.map(PyRow::new);
        self.stepper
            .answer("a user", |ask| match (ask, membership) {
                (Ask::User { reply, .. }, None) => {
                    reply.send(read);
                    Ok(())
                }
                (Ask::ResolveUserIdentity { reply, .. }, membership) => {
                    reply.send(IdentityRead {
                        user: read,
                        membership,
                    });
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
            _ => {
                return Err(PyValueError::new_err(
                    "an org is answered with both its id and its row",
                ));
            }
        };
        self.stepper.answer("an org", |ask| match ask {
            Ask::Org { reply, .. } | Ask::OrgByAlias { reply, .. } => {
                reply.send(org);
                Ok(())
            }
            other => Err(other),
        })
    }

    /// A key hash, a team id or an end-user id, or nothing.
    #[pyo3(signature = (value = None))]
    fn text(&self, value: Option<String>) -> PyResult<()> {
        self.stepper
            .answer("a key hash, team id or end-user id", |ask| match ask {
                Ask::KeyMappingHash { reply, .. } | Ask::ReusableKeyHash { reply, .. } => {
                    reply.send(value.map(KeyHash::new));
                    Ok(())
                }
                Ask::DelegatedAgentTeam { reply, .. } | Ask::ValidatedEndUserId { reply, .. } => {
                    reply.send(value);
                    Ok(())
                }
                other => Err(other),
            })
    }

    #[pyo3(signature = (facts = None, row = None, *, user_email = None))]
    fn mapped_key(
        &self,
        facts: Option<KeyFacts>,
        row: Option<Py<PyAny>>,
        user_email: Option<String>,
    ) -> PyResult<()> {
        let mapped = match (facts, row) {
            (Some(facts), Some(row)) => Some(MappedKey {
                key: keyed(facts, row),
                user_email,
            }),
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "a mapped key is answered with both its facts and its row",
                ));
            }
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
            (Some(hash), false) => KeyMappingWrite::Created(KeyHash::new(hash)),
            (Some(hash), true) => KeyMappingWrite::Conflict(KeyHash::new(hash)),
            (None, true) => KeyMappingWrite::ConflictVanished,
            (None, false) => {
                return Err(PyValueError::new_err("a created key mapping names its key"));
            }
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
        self.stepper
            .answer("an agent authentication record", |ask| match ask {
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

#[cfg(test)]
mod tests {
    use litellm_gateway_auth_core::credential::jwt::KeyOrigin;
    use rstest::rstest;

    use super::minted_org_id;

    fn auto_registered(minted: bool) -> KeyOrigin {
        KeyOrigin::AutoRegistered {
            minted,
            org_id: Some("jwt-org".to_owned()),
        }
    }

    #[rstest]
    #[case::created_for_the_jwt(auto_registered(true), Some("jwt-org"))]
    #[case::reused_for_the_jwt(auto_registered(false), None)]
    #[case::mapped_by_an_admin(KeyOrigin::Mapping, None)]
    fn only_a_key_created_for_the_jwt_takes_its_org(
        #[case] origin: KeyOrigin,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(minted_org_id(&origin).as_deref(), expected);
    }
}
