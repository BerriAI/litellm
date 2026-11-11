use crate::jwt::{ClaimDenial, JwtDenial, RbacDenial, TeamDenial, VerifyDenial};

const NOT_PREMIUM_USER: &str = "You must be a LiteLLM Enterprise user to use this feature. If you have a license please \
set `LITELLM_LICENSE` in your env. Get a 7 day trial key here: https://www.litellm.ai/enterprise#trial. \n\
Pricing: https://www.litellm.ai/#pricing";
pub(crate) const INVALID_VIRTUAL_KEY_ERROR_MESSAGE: &str = "LiteLLM Virtual Key expected";
pub(crate) const INVALID_API_KEY_ERROR_MESSAGE: &str =
    "Authentication Error, Invalid API key provided.";
const JWT_AUTH_DISABLED_HINT: &str = " This key has the structure of a JWT, but JWT auth is not enabled on this proxy, so it \
was treated as a virtual key. Set `enable_jwt_auth: true` under `general_settings` in your proxy config to authenticate \
with JWTs.";

/// Why a request does not authenticate: a denial the core renders, or a host that could not answer.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    PremiumRequired(#[from] PremiumRequired),
    #[error(transparent)]
    Key(#[from] KeyDenial),
    #[error(transparent)]
    Jwt(#[from] JwtDenial),
    #[error(transparent)]
    BudgetExceeded(#[from] BudgetExceeded),
    #[error(transparent)]
    Authorize(#[from] AuthorizeDenial),
    #[error(transparent)]
    Route(#[from] RouteDenial),
    #[error(transparent)]
    Unavailable(#[from] Unavailable),
    /// A host step raised. The host keeps that failure and surfaces it, so the core never renders this.
    #[error("the auth host failed a step")]
    HostFailed,
}

impl Error {
    /// The host could not answer, so the caller surfaces the host's own failure instead of a denial.
    pub fn is_host_failure(&self) -> bool {
        match self {
            Self::Unavailable(_) | Self::HostFailed => true,
            Self::PremiumRequired(_)
            | Self::Key(_)
            | Self::Jwt(_)
            | Self::BudgetExceeded(_)
            | Self::Authorize(_)
            | Self::Route(_) => false,
        }
    }
}

impl From<SessionDenial> for Error {
    fn from(denial: SessionDenial) -> Self {
        Self::Key(KeyDenial::Session(denial))
    }
}

impl From<VerifyDenial> for Error {
    fn from(denial: VerifyDenial) -> Self {
        Self::Jwt(JwtDenial::Verify(denial))
    }
}

impl From<ClaimDenial> for Error {
    fn from(denial: ClaimDenial) -> Self {
        Self::Jwt(JwtDenial::Claim(denial))
    }
}

impl From<RbacDenial> for Error {
    fn from(denial: RbacDenial) -> Self {
        Self::Jwt(JwtDenial::Rbac(denial))
    }
}

impl From<TeamDenial> for Error {
    fn from(denial: TeamDenial) -> Self {
        Self::Jwt(JwtDenial::Team(denial))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PremiumRequired {
    #[error("Oauth2 token validation is only available for premium users. {NOT_PREMIUM_USER}")]
    OAuth2,
    #[error("JWT Auth is an enterprise only feature. {NOT_PREMIUM_USER}")]
    Jwt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::EnumString)]
pub enum Dependency {
    #[strum(serialize = "db")]
    Db,
}

// TODO: the host writes the message until the Prisma outage handling moves to Rust
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct Unavailable {
    pub dependency: Dependency,
    /// The outage is expected to clear by itself, the only kind `allow_requests_on_db_unavailable`
    /// serves through.
    pub transient: bool,
    pub message: String,
}

// TODO: the host writes the message until budgets move to Rust
/// `litellm.BudgetExceededError` as a host-owned budget step reported it.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct BudgetExceeded {
    pub current_cost: f64,
    pub max_budget: f64,
    pub message: String,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum KeyDenial {
    #[error("Authentication Error, No api key passed in.")]
    MissingKey,
    #[error("Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix.")]
    MalformedKey,
    #[error("'allow_user_auth' not set or set to False")]
    UserAuthDisabled,
    #[error(
        "Authentication Error - Expired Key. Key Expiry time {} and current time {}",
        utc(*expires),
        utc(*now)
    )]
    Expired { expires: f64, now: f64, key: String },
    #[error("Authentication Error, Tried to access route={route}, which is only for MASTER KEY")]
    MasterKeyOnlyRoute { route: String },
    #[error("No connected db.")]
    NoDb,
    #[error(
        "{INVALID_VIRTUAL_KEY_ERROR_MESSAGE}. Received={received}, expected to start with 'sk-'.{}",
        if *jwt_hint { JWT_AUTH_DISABLED_HINT } else { "" }
    )]
    NotVirtualKey { received: String, jwt_hint: bool },
    #[error("{INVALID_API_KEY_ERROR_MESSAGE}")]
    NotFound,
    #[error(transparent)]
    Session(#[from] SessionDenial),
    #[error("Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin.")]
    Blocked,
    #[error(
        "Authentication Error, User={user_id} has been deactivated via SCIM. Keys owned by this user cannot be used."
    )]
    UserDeactivated { user_id: String },
    #[error("Invalid API key, no token associated")]
    NoToken,
}

/// Why `GrantResolver.resolve` would not refresh a session key's grants.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SessionDenial {
    #[error("Authentication Error, user '{user_id}' no longer exists.")]
    UserGone { user_id: String },
    #[error("Team doesn't exist in db. Team={team_id}. Create team via `/team/new` call.")]
    TeamGone { team_id: String },
    #[error("Team '{team_id}' is not in your team memberships.")]
    NotAMember { team_id: String },
}

fn utc(epoch: f64) -> String {
    chrono::DateTime::from_timestamp_micros((epoch * 1_000_000.0).round() as i64).map_or_else(
        || epoch.to_string(),
        |time| time.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true),
    )
}

/// Why `common_checks` refused an authenticated request.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AuthorizeDenial {
    #[error(
        "{subject} no pricing in the cost map, so litellm cannot price the request. Requests for unpriced models are blocked because 'block_requests_for_models_without_pricing' is enabled. Add pricing (input_cost_per_token/output_cost_per_token) to allow the request.",
        subject = unpriced_subject(.models)
    )]
    UnpricedModels { models: Vec<String> },
    #[error("Team={team_id} is blocked. Update via `/team/unblock` if you're an admin.")]
    TeamBlocked { team_id: String },
    #[error("Requests made with this agent's key must include the x-litellm-trace-id header.")]
    AgentTraceIdRequired,
    #[error(
        "Team '{team_id}' requires a trace ID on every LLM, MCP and agent request. Send an x-litellm-trace-id header. LLM requests can also send a W3C traceparent header or set metadata.trace_id in the body."
    )]
    TeamTraceIdRequired { team_id: String },
    #[error(
        "Client-side 'metadata.tags' not allowed in request. 'reject_clientside_metadata_tags'=True. Tags can only be set via key or team metadata, or by configured request_tag_mappings."
    )]
    ClientsideTags,
    #[error("'user' param not passed in. 'enforce_user_param'=True")]
    UserParamMissing,
    #[error(
        "User={user_id} has no membership in Team={team_id}, so their spend against the team member budget cannot be verified and fail_closed_budget_enforcement is enabled. Add them to the team with POST /team/member_add."
    )]
    TeamMemberWithoutMembership { user_id: String, team_id: String },
}

fn unpriced_subject(models: &[String]) -> String {
    let names = models
        .iter()
        .map(|model| format!("'{model}'"))
        .collect::<Vec<_>>()
        .join(", ");
    match models {
        [_] => format!("Model {names} has"),
        _ => format!("Models {names} have"),
    }
}

/// Why the organization role checks or the route checks refused an authenticated request.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RouteDenial {
    #[error("Only proxy admins can create new organizations. You are {role}")]
    OrganizationNewNeedsAdmin { role: String },
    #[error(
        "Tried to access route={route} but you are not a member of any organization. Please contact the proxy admin to request access."
    )]
    NoOrganization { route: String },
    #[error("Passed organization_id is None, please pass an organization_id in your request")]
    OrganizationIdMissing,
    #[error(
        "You do not have a role within the selected organization. Passed organization_id: {organization_id}. Please contact the organization admin to request access."
    )]
    NoOrganizationRole { organization_id: String },
    #[error(
        "You do not have the required role to perform {route} in Organization {organization_id}. Your role is {role} in Organization {organization_id}"
    )]
    NotOrganizationAdmin {
        route: String,
        organization_id: String,
        role: String,
    },
    #[error(
        "Passed organization_id is None, please specify the organization_id in your request. You are part of multiple organizations: {organizations}",
        organizations = serde_json::json!(organizations)
    )]
    TeamOrganizationMissing { organizations: Vec<String> },
    #[error(
        "You do not have the required role to call {route}. Your role is {role} in Organization {organization_id}"
    )]
    NotTeamOrganizationAdmin {
        route: String,
        organization_id: String,
        role: String,
    },
    #[error("Invalid proxy server token passed. valid_token=None.")]
    NoToken,
    #[error("user not allowed to access this route. Route={route} is an admin only route")]
    CustomAdminOnly { route: String },
    #[error(
        "Key/team denied access to passthrough route {route}. Matched `{denied_route}` in `denied_passthrough_routes`."
    )]
    PassThroughDenied { route: String, denied_route: String },
    #[error(
        "Key/team not allowed to access passthrough route {route}. Configure `allowed_passthrough_routes` on the team or key."
    )]
    PassThroughNotAllowed { route: String },
    #[error("key not allowed to access this user's info. user_id={user_id}, key's user_id={key_user_id}")]
    OtherUsersInfo { user_id: String, key_user_id: String },
    #[error("user not allowed to access this OpenAI routes, role= {role}")]
    ViewerInference { role: &'static str },
    #[error("user not allowed to access this route, role= {role}. Trying to access: {route}")]
    ViewerNotAllowed { role: &'static str, route: String },
    #[error(
        "user not allowed to access this route, role= {role}. Trying to access: {route} and updating invalid param: {param}. only user_email can be updated"
    )]
    ViewerInvalidParam {
        role: &'static str,
        route: String,
        param: String,
    },
    #[error(
        "Authentication Error, Only proxy admin can be used to generate, delete, update info for new keys/users/teams. Route={route}. Your role={role}. Your user_id={user_id}"
    )]
    AdminOnly {
        route: String,
        role: String,
        user_id: String,
    },
}
