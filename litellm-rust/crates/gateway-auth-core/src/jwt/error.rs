use std::{collections::BTreeSet, fmt};

use litellm_config::{RbacRole, UnregisteredJwtClientBehavior};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum JwtDenial {
    #[error(transparent)]
    Verify(#[from] VerifyDenial),
    #[error(transparent)]
    Claim(#[from] ClaimDenial),
    #[error(transparent)]
    Rbac(#[from] RbacDenial),
    #[error(transparent)]
    Team(#[from] TeamDenial),
    #[error("Invalid JWT token")]
    CustomValidate,
    #[error("Managed agents require trusted JWT issuer and audience validation")]
    ManagedIssuerUntrusted,
    #[error("Agent identities can only access inference and agent discovery routes")]
    ManagedRouteNotAllowed,
    #[error(transparent)]
    AgentIdentity(#[from] AgentIdentityFailure),
    #[error("No registered agent matches JWT claim {}={claim}", or_unset(agent_id_jwt_field.as_deref()))]
    UnregisteredAgent {
        agent_id_jwt_field: Option<String>,
        claim: String,
    },
    #[error("The granting team is not allowed to access this route")]
    GrantingTeamRouteNotAllowed,
    #[error(
        "Email domain not allowed. User email: {}. Allowed domain: {}",
        or_unset(user_email.as_deref()),
        or_unset(allowed_domain.as_deref())
    )]
    EmailDomainNotAllowed {
        user_email: Option<String>,
        allowed_domain: Option<String>,
    },
    #[error("User doesn't exist in db. 'user_id'={user_id}. Create user via `/user/new` call.")]
    UserNotFound { user_id: String },
    #[error(
        "No user or team id found in token. enforce_rbac is set to True. Token must belong to a proxy admin, team, or \
         user."
    )]
    NoObjectId,
    #[error(
        "User={} has been deactivated via SCIM. Keys owned by this user cannot be used.",
        or_unset(user_id.as_deref())
    )]
    UserDeactivated { user_id: Option<String> },
    #[error("Managed agents require direct JWT authentication without virtual-key mapping")]
    ManagedAgentNeedsDirectJwt,
    #[error(
        "JWT Key Mapping: Required claim '{claim_field}' is missing from the JWT. Access denied."
    )]
    MappingClaimMissing { claim_field: String },
    #[error(
        "JWT Key Mapping: No registered mapping for {claim_field}='{claim_value}'. Access denied."
    )]
    NoMapping {
        claim_field: String,
        claim_value: String,
    },
    #[error(
        "JWT Key Mapping: {} requires a database connection. Configure a database or change \
         unregistered_jwt_client_behavior.",
        upper(*behavior)
    )]
    MappingNeedsDb {
        behavior: UnregisteredJwtClientBehavior,
    },
    #[error("JWT Key Mapping: No eligible existing virtual key for this identity. Access denied.")]
    NoEligibleKey,
    #[error(
        "JWT Key Mapping: {} race resolution failed — winner's mapping was concurrently removed. Retry the request.",
        upper(*behavior)
    )]
    MappingRaceVanished {
        behavior: UnregisteredJwtClientBehavior,
    },
}

// TODO: the host writes the message until the managed agent policy moves to Rust
/// What `AgentIdentityStore.record_authentication` refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct AgentIdentityFailure {
    pub policy_unavailable: bool,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum VerifyDenial {
    /// `jwt.get_unverified_header` raised.
    #[error(transparent)]
    Malformed(TokenError),
    #[error(
        "an ID-JAG is an authorization grant, not a bearer token; redeem it at the token endpoint"
    )]
    IdJagNotBearer,
    #[error(transparent)]
    Key(KeyFailure),
    #[error("Token Expired")]
    Expired,
    #[error("Validation fails: {0}")]
    Invalid(TokenError),
    #[error(transparent)]
    Assertion(AssertionRejection),
    #[error(
        "OIDC UserInfo endpoint not configured. Set 'oidc_userinfo_endpoint' in JWT auth config."
    )]
    UserinfoNotConfigured,
}

/// Why the token itself is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TokenError {
    /// jsonwebtoken refused the token's encoding, header, algorithm or signature.
    #[error(transparent)]
    Jwt(#[from] jsonwebtoken::errors::Error),
    #[error("Invalid 'crit' header: must be a non-empty list")]
    CritEmpty,
    #[error("Unsupported critical extension: {0}")]
    CritUnsupported(String),
    #[error("Unencoded payloads ('b64': false) are not supported")]
    UnencodedPayload,
    #[error(transparent)]
    Key(#[from] KeyError),
    #[error("Token is missing the \"{0}\" claim")]
    MissingClaim(String),
    #[error("Issued At claim (iat) must be an integer.")]
    IatNotInteger,
    #[error("The token is not yet valid (iat)")]
    IssuedInFuture,
    #[error("Not Before claim (nbf) must be an integer.")]
    NbfNotInteger,
    #[error("The token is not yet valid (nbf)")]
    NotYetValid,
    #[error("Expiration Time claim (exp) must be an integer.")]
    ExpNotInteger,
    #[error("Signature has expired")]
    Expired,
    #[error("Payload Issuer (iss) must be a string")]
    IssuerNotString,
    #[error("Invalid issuer")]
    WrongIssuer,
    #[error("Invalid claim format in token")]
    AudienceNotStrings,
    #[error("Audience doesn't match")]
    WrongAudience,
    #[error("Subject must be a string")]
    SubjectNotString,
    #[error("JWT ID must be a string")]
    JtiNotString,
    /// `_get_decode_options`' `ValueError`, for an issuer with neither audience nor opt-out.
    #[error("audience must be provided unless disable_audience_validation=True")]
    AudienceRequired,
}

/// Why the selected JWK cannot check the token's signature (`PyJWK` and `prepare_key`).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum KeyError {
    #[error("kty is not found: {jwk}")]
    MissingKty { jwk: String },
    #[error("Unsupported kty: {kty}")]
    UnsupportedKty { kty: String },
    #[error("crv is not found: {jwk}")]
    MissingCrv { jwk: String },
    #[error("Unsupported crv: {crv}")]
    UnsupportedCrv { crv: String },
    #[error("Unable to construct key from JWK: {reason}")]
    Malformed { reason: String },
    #[error("{algorithm} keys cannot be verified by the gateway")]
    Unverifiable { algorithm: &'static str },
}

/// The URLs a kid was looked up in, as `NoMatchingJWTPublicKeyError` names them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Searched {
    Url(String),
    Urls(Vec<String>),
}

impl fmt::Display for Searched {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Url(url) => f.write_str(url),
            Self::Urls(urls) => f.write_str(&json_list(urls.iter())),
        }
    }
}

/// A signing key lookup that found nothing to verify with.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum KeyFailure {
    #[error("Missing JWT Public Key URL from environment.")]
    MissingUrl,
    #[error("No matching public key found. keys={keys}, kid={}", or_unset(kid.as_deref()))]
    NoMatch { keys: Searched, kid: Option<String> },
    #[error("JWT Auth: OIDC discovery document at {url} does not contain a 'jwks_uri' field.")]
    NoJwksUri { url: String },
    /// `JWKSUnreachableError`, the one failure a stale copy may cover.
    #[error(
        "Service Unavailable, the identity provider's JWKS endpoint is temporarily unreachable, so the JWT signature \
         could not be verified. Please retry shortly. Error: {error}"
    )]
    Unreachable { error: String },
}

/// `AssertionRejection`: why `decode_trusted_assertion` refused an assertion.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AssertionRejection {
    #[error("malformed assertion: {0}")]
    Malformed(TokenError),
    #[error("iss is not an issuer this gateway trusts")]
    UntrustedIssuer,
    #[error("no signing key for the assertion: {0}")]
    NoSigningKey(KeyFailure),
    #[error("assertion failed verification: {0}")]
    Invalid(TokenError),
    #[error("custom_validate rejected the assertion's claims")]
    CustomValidate,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ClaimDenial {
    /// `get_scopes` on a `scope` claim that is neither a string nor a list.
    #[error("Unmapped scope type - {kind}. Supported types - array, string.")]
    UnmappedScopeType { kind: &'static str },
}

/// The client half of `ModelAccessDeniedHTTPException`, which never names the models that are allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "The requested model '{model}' is not available for this API key, or the model name is invalid. Check the models \
     available to you and try again."
)]
pub struct ModelAccessDenied<'a> {
    pub model: &'a str,
}

/// Each model denial's message is the internal half of `ModelAccessDeniedHTTPException`, and
/// `ModelAccessDenied` its client half.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RbacDenial {
    /// `enforce_rbac` is on and the token maps to no role.
    #[error(
        "Unmatched token passed in. enforce_rbac is set to True. Token must belong to a proxy admin, team, or user."
    )]
    UnmatchedRole,
    #[error(
        "Role={} not allowed to call model={model}. Allowed models={}",
        <&str>::from(role),
        json_list(allowed.iter())
    )]
    RoleModel {
        role: RbacRole,
        model: String,
        allowed: Box<[String]>,
    },
    #[error(
        "Role={} not allowed to call route={route}. Allowed routes={}",
        <&str>::from(role),
        json_list(allowed.iter())
    )]
    RoleRoute {
        role: RbacRole,
        route: String,
        allowed: Box<[String]>,
    },
    #[error("model={model} not allowed. Allowed_models={}", json_list(allowed.iter()))]
    ScopeModel { model: String, allowed: Vec<String> },
    /// `allowed` is `admin_allowed_routes` with its groups expanded.
    #[error(
        "Admin not allowed to access this route. Route={route}, Allowed Routes={}",
        json_list(allowed.iter())
    )]
    AdminRoute { route: String, allowed: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TeamDenial {
    #[error(
        "No teams found in token. `enforce_team_based_model_access` is set to True. Token must belong to a team."
    )]
    NoTeamsInToken,
    #[error("{}", passthrough_route_denied(route, denied_route.as_deref()))]
    PassthroughRouteDenied {
        route: String,
        denied_route: Option<String>,
    },
    #[error(
        "Team {team_id} can access model {model} but route {route} is not in litellm_jwtauth.team_allowed_routes"
    )]
    RouteNotAllowed {
        team_id: String,
        model: String,
        route: String,
    },
    #[error(
        "No team has access to the requested model: {model}. Checked teams={}. Check `/models` to see all available \
         models.",
        json_list(team_ids.iter())
    )]
    NoModelAccess {
        model: String,
        team_ids: BTreeSet<String>,
    },
    #[error("Team doesn't exist in db. Team={team_id}. Create team via `/team/new` call.")]
    TeamNotFound { team_id: String },
    #[error("Team with alias '{alias}' doesn't exist in db. Create team via `/team/new` call.")]
    TeamAliasNotFound { alias: String },
    #[error(
        "Multiple teams found with alias '{alias}'. Please use team_id_jwt_field instead or ensure team aliases are \
         unique."
    )]
    TeamAliasAmbiguous { alias: String },
    #[error("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")]
    NoDb,
    #[error(
        "No team found in token. Checked team_id field '{}' and team_alias field '{}'.{}",
        or_unset(team_id_jwt_field.as_deref()),
        or_unset(team_alias_jwt_field.as_deref()),
        array_index_hint(team_id_jwt_field.as_deref().unwrap_or_default())
    )]
    NoTeamInToken {
        team_id_jwt_field: Option<String>,
        team_alias_jwt_field: Option<String>,
    },
    #[error(
        "x-litellm-team-id '{header_value}' does not resolve to a team id or a unique team alias in your JWT's allowed \
         teams. Allowed team ids: {}",
        json_list(allowed_team_ids.iter())
    )]
    HeaderTeamNotAllowed {
        header_value: String,
        allowed_team_ids: BTreeSet<String>,
    },
    #[error(
        "x-litellm-team-id '{header_value}' does not resolve to a team id or a unique team alias among your team \
         memberships."
    )]
    HeaderTeamNotMember { header_value: String },
    #[error(
        "Team '{header_value}' (from x-litellm-team-id header) is not allowed to access route '{route}'."
    )]
    HeaderTeamRouteNotAllowed { header_value: String, route: String },
    #[error(
        "No team you are a member of has access to the requested model: {model}. Check `/models` to see the models \
         available to you."
    )]
    NoMemberTeamHasModel { model: String },
    #[error("User is not a member of any team. Add the user to a team via the LiteLLM UI or API.")]
    NotTeamMember,
}

fn or_unset(value: Option<&str>) -> &str {
    value.unwrap_or("unset")
}

fn upper(behavior: UnregisteredJwtClientBehavior) -> String {
    <&str>::from(behavior).to_uppercase()
}

fn json_list<'a>(items: impl Iterator<Item = &'a String>) -> String {
    Value::from_iter(items.map(String::as_str)).to_string()
}

fn passthrough_route_denied(route: &str, denied_route: Option<&str>) -> String {
    match denied_route {
        Some(denied_route) => format!(
            "Key/team denied access to passthrough route {route}. Matched `{denied_route}` in \
             `denied_passthrough_routes`."
        ),
        None => format!(
            "Team not allowed to access passthrough route {route}. Configure `allowed_passthrough_routes` on the team."
        ),
    }
}

fn is_digits(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_ascii_digit())
}

fn is_word(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_alphanumeric() || c == '_')
}

fn array_index_hint(field: &str) -> String {
    if let Some((base, index)) = field.rsplit_once('.') {
        if !is_digits(index) {
            return String::new();
        }
        return format!(
            " Hint: dot-notation array indexing (e.g. '{field}') is not supported. Use '{base}' instead — LiteLLM \
             automatically uses the first element when the field value is a list."
        );
    }
    match field
        .strip_suffix(']')
        .and_then(|field| field.split_once('['))
    {
        Some((base, index)) if is_word(base) && is_digits(index) => format!(
            " Hint: array indexing (e.g. '{field}') is not supported in team_id_jwt_field. Use '{base}' instead — \
             LiteLLM automatically uses the first element when the field value is a list."
        ),
        _ => String::new(),
    }
}
