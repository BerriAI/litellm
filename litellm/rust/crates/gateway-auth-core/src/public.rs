use crate::{
    BudgetExceeded, Error, KeyDenial, PremiumRequired,
    credential::jwt::{
        JwtDenial, KeyFailure, ModelAccessDenied, RbacDenial, TeamDenial, VerifyDenial,
    },
};

/// `ProxyErrorTypes` values a denial carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum ErrorType {
    #[strum(serialize = "auth_error")]
    Auth,
    #[strum(serialize = "expired_key")]
    ExpiredKey,
    #[strum(serialize = "no_db_connection")]
    NoDbConnection,
    #[strum(serialize = "token_not_found_in_db")]
    TokenNotFoundInDb,
    #[strum(serialize = "auth_provider_unavailable")]
    AuthProviderUnavailable,
    #[strum(serialize = "model_cost_map_missing")]
    ModelCostMapMissing,
    #[strum(serialize = "bad_request_error")]
    BadRequest,
}

/// A denial as the caller sees it: the exception the Python proxy raised for it, with its status
/// and message. Each host builds its error from this alone.
#[derive(Clone, Debug, PartialEq)]
pub enum PublicDenial {
    /// `ProxyException(message, type, param, code)`
    Proxy {
        status: u16,
        message: String,
        error_type: ErrorType,
        param: Option<String>,
    },
    /// `HTTPException(status_code, detail)`
    Http { status: u16, detail: String },
    /// The 401 for a token that is not a virtual key, which log routing tells apart by its origin.
    InvalidVirtualKey { status: u16, detail: String },
    /// `ModelAccessDeniedHTTPException(internal_message, status_code, detail)`
    ModelAccess {
        status: u16,
        detail: String,
        internal_message: String,
    },
    /// A bare `Exception(message)`, which the auth exception handler reports as a 401.
    Plain { message: String },
    /// `litellm.BudgetExceededError`
    BudgetExceeded(BudgetExceeded),
    /// `TeamNotFoundError(team_id)`
    TeamNotFound { team_id: String },
}

fn proxy(
    status: u16,
    denial: &impl ToString,
    error_type: ErrorType,
    param: Option<&str>,
) -> PublicDenial {
    PublicDenial::Proxy {
        status,
        message: denial.to_string(),
        error_type,
        param: param.map(str::to_owned),
    }
}

fn auth(denial: &impl ToString) -> PublicDenial {
    proxy(401, denial, ErrorType::Auth, None)
}

fn http(status: u16, denial: &impl ToString) -> PublicDenial {
    PublicDenial::Http {
        status,
        detail: denial.to_string(),
    }
}

fn plain(denial: &impl ToString) -> PublicDenial {
    PublicDenial::Plain {
        message: denial.to_string(),
    }
}

impl PremiumRequired {
    pub fn public(&self) -> PublicDenial {
        proxy(403, self, ErrorType::Auth, Some("premium_user"))
    }
}

impl Error {
    /// The exception Python raises for this error, or `None` when the host surfaces its own failure.
    pub fn public(&self) -> Option<PublicDenial> {
        Some(match self {
            Self::PremiumRequired(premium) => premium.public(),
            Self::Key(denial) => denial.public(),
            Self::Jwt(denial) => denial.public(),
            Self::BudgetExceeded(exceeded) => PublicDenial::BudgetExceeded(exceeded.clone()),
            Self::HostFailed => return None,
        })
    }
}

impl KeyDenial {
    pub(crate) fn public(&self) -> PublicDenial {
        match self {
            Self::MissingKey
            | Self::MalformedKey
            | Self::MasterKeyOnlyRoute { .. }
            | Self::Blocked
            | Self::UserDeactivated { .. } => auth(self),
            Self::UserAuthDisabled => http(403, self),
            Self::Expired { key, .. } => proxy(401, self, ErrorType::ExpiredKey, Some(key)),
            Self::NoDb => proxy(400, self, ErrorType::NoDbConnection, None),
            Self::NotVirtualKey { .. } => PublicDenial::InvalidVirtualKey {
                status: 401,
                detail: self.to_string(),
            },
            Self::NotFound => proxy(401, self, ErrorType::TokenNotFoundInDb, Some("key")),
            Self::NoToken => http(401, self),
        }
    }
}

impl JwtDenial {
    pub(crate) fn public(&self) -> PublicDenial {
        match self {
            Self::Verify(denial) => denial.public(),
            Self::Rbac(denial) => denial.public(),
            Self::Team(denial) => denial.public(),
            Self::Claim(_) | Self::UserNotFound { .. } => plain(self),
            Self::AgentIdentity(failure) => {
                http(if failure.policy_unavailable { 503 } else { 403 }, self)
            }
            Self::EmailDomainNotAllowed { .. } => {
                proxy(403, self, ErrorType::Auth, Some("user_email"))
            }
            Self::UserDeactivated { .. } => http(401, self),
            Self::MappingNeedsDb { .. } | Self::CustomValidateUnavailable => http(500, self),
            Self::MappingRaceVanished { .. } => http(503, self),
            Self::CustomValidate
            | Self::ManagedIssuerUntrusted
            | Self::ManagedRouteNotAllowed
            | Self::UnregisteredAgent { .. }
            | Self::GrantingTeamRouteNotAllowed
            | Self::NoObjectId
            | Self::ManagedAgentNeedsDirectJwt
            | Self::MappingClaimMissing { .. }
            | Self::NoMapping { .. }
            | Self::NoEligibleKey => http(403, self),
        }
    }
}

impl VerifyDenial {
    pub(crate) fn public(&self) -> PublicDenial {
        match self {
            Self::Key(KeyFailure::Unreachable { .. }) => {
                proxy(503, self, ErrorType::AuthProviderUnavailable, None)
            }
            Self::Expired => proxy(401, self, ErrorType::ExpiredKey, None),
            Self::Malformed(_)
            | Self::IdJagNotBearer
            | Self::Key(_)
            | Self::Invalid(_)
            | Self::Assertion(_)
            | Self::UserinfoNotConfigured => plain(self),
        }
    }
}

impl RbacDenial {
    pub(crate) fn public(&self) -> PublicDenial {
        match self {
            Self::UnmatchedRole | Self::RoleRoute { .. } => http(403, self),
            Self::RoleModel { model, .. } | Self::ScopeModel { model, .. } => {
                PublicDenial::ModelAccess {
                    status: 403,
                    detail: ModelAccessDenied { model }.to_string(),
                    internal_message: self.to_string(),
                }
            }
            Self::AdminRoute { .. } => plain(self),
        }
    }
}

impl TeamDenial {
    pub(crate) fn public(&self) -> PublicDenial {
        match self {
            Self::TeamNotFound { .. } | Self::TeamAliasNotFound { .. } => http(404, self),
            Self::TeamAliasAmbiguous { .. } => http(400, self),
            Self::NoDb | Self::NoTeamInToken { .. } => plain(self),
            Self::NoTeamsInToken
            | Self::PassthroughRouteDenied { .. }
            | Self::RouteNotAllowed { .. }
            | Self::NoModelAccess { .. }
            | Self::HeaderTeamNotAllowed { .. }
            | Self::HeaderTeamNotMember { .. }
            | Self::HeaderTeamRouteNotAllowed { .. }
            | Self::NoMemberTeamHasModel { .. }
            | Self::NotTeamMember => http(403, self),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use litellm_config::RbacRole;
    use rstest::rstest;

    use super::*;
    use crate::{
        PremiumRequired,
        credential::jwt::{AssertionRejection, ClaimDenial, Searched, TokenError},
    };

    const JWKS: &str = "https://idp.example/jwks";

    fn strings(items: &[&str]) -> Box<[String]> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    fn ids(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

    fn http(status: u16, detail: &str) -> PublicDenial {
        PublicDenial::Http {
            status,
            detail: detail.to_owned(),
        }
    }

    fn plain(message: &str) -> PublicDenial {
        PublicDenial::Plain {
            message: message.to_owned(),
        }
    }

    fn no_team_in_token(team_id_jwt_field: Option<&str>) -> TeamDenial {
        TeamDenial::NoTeamInToken {
            team_id_jwt_field: team_id_jwt_field.map(str::to_owned),
            team_alias_jwt_field: None,
        }
    }

    fn over_budget() -> BudgetExceeded {
        BudgetExceeded {
            current_cost: 2.0,
            max_budget: 1.0,
            message: "Budget has been exceeded!".to_owned(),
            subject: None,
            entity_type: Some("key".to_owned()),
            entity_id: None,
        }
    }

    #[rstest]
    #[case::not_found(
        KeyDenial::NotFound.into(),
        PublicDenial::Proxy {
            status: 401,
            message: "Authentication Error, Invalid API key provided.".to_owned(),
            error_type: ErrorType::TokenNotFoundInDb,
            param: Some("key".to_owned()),
        }
    )]
    #[case::budget(
        Error::BudgetExceeded(over_budget()),
        PublicDenial::BudgetExceeded(over_budget())
    )]
    #[case::expired_key(
        KeyDenial::Expired {
            expires: jiff::Timestamp::from_millisecond(1_700_000_000_250).unwrap(),
            now: jiff::Timestamp::from_second(1_700_000_001).unwrap(),
            key: "sk-...cdef".to_owned(),
        }
        .into(),
        PublicDenial::Proxy {
            status: 401,
            message: "Authentication Error - Expired Key. Key Expiry time 2023-11-14T22:13:20.250Z and current time \
                      2023-11-14T22:13:21Z"
                .to_owned(),
            error_type: ErrorType::ExpiredKey,
            param: Some("sk-...cdef".to_owned()),
        }
    )]
    #[case::premium(
        PremiumRequired::Jwt.into(),
        PublicDenial::Proxy {
            status: 403,
            message: PremiumRequired::Jwt.to_string(),
            error_type: ErrorType::Auth,
            param: Some("premium_user".to_owned()),
        }
    )]
    fn denials_render_as_the_proxy_raised_them(
        #[case] error: Error,
        #[case] expected: PublicDenial,
    ) {
        assert_eq!(error.public(), Some(expected));
    }

    #[test]
    fn a_host_failure_is_left_to_the_host() {
        assert_eq!(Error::HostFailed.public(), None);
    }

    #[rstest]
    #[case::unmatched(
        RbacDenial::UnmatchedRole,
        PublicDenial::Http {
            status: 403,
            detail: "Unmatched token passed in. enforce_rbac is set to True. Token must belong to a proxy admin, team, or user."
                .to_owned(),
        }
    )]
    #[case::role_route(
        RbacDenial::RoleRoute { role: RbacRole::Team, route: "/x".to_owned(), allowed: strings(&["info_routes", "/y"]) },
        PublicDenial::Http {
            status: 403,
            detail: "Role=team not allowed to call route=/x. Allowed routes=[\"info_routes\",\"/y\"]".to_owned(),
        }
    )]
    #[case::role_model(
        RbacDenial::RoleModel { role: RbacRole::InternalUser, model: "m".to_owned(), allowed: strings(&["a"]) },
        PublicDenial::ModelAccess {
            status: 403,
            detail: "The requested model 'm' is not available for this API key, or the model name is invalid. Check the models available to you and try again.".to_owned(),
            internal_message: "Role=internal_user not allowed to call model=m. Allowed models=[\"a\"]".to_owned(),
        }
    )]
    #[case::scope_model(
        RbacDenial::ScopeModel { model: "m".to_owned(), allowed: vec![] },
        PublicDenial::ModelAccess {
            status: 403,
            detail: "The requested model 'm' is not available for this API key, or the model name is invalid. Check the models available to you and try again.".to_owned(),
            internal_message: "model=m not allowed. Allowed_models=[]".to_owned(),
        }
    )]
    fn denial_messages(#[case] denial: RbacDenial, #[case] expected: PublicDenial) {
        assert_eq!(denial.public(), expected);
    }

    #[rstest]
    #[case::no_teams_in_token(
        TeamDenial::NoTeamsInToken,
        http(
            403,
            "No teams found in token. `enforce_team_based_model_access` is set to True. Token must belong to a team."
        )
    )]
    #[case::passthrough_not_granted(
        TeamDenial::PassthroughRouteDenied { route: "/openai/x".to_owned(), denied_route: None },
        http(403, "Team not allowed to access passthrough route /openai/x. Configure `allowed_passthrough_routes` on the team.")
    )]
    #[case::passthrough_denied(
        TeamDenial::PassthroughRouteDenied { route: "/openai/x".to_owned(), denied_route: Some("/openai/*".to_owned()) },
        http(403, "Key/team denied access to passthrough route /openai/x. Matched `/openai/*` in `denied_passthrough_routes`.")
    )]
    #[case::route_not_allowed(
        TeamDenial::RouteNotAllowed { team_id: "t".to_owned(), model: "gpt".to_owned(), route: "/r".to_owned() },
        http(403, "Team t can access model gpt but route /r is not in litellm_jwtauth.team_allowed_routes")
    )]
    #[case::no_model_access(
        TeamDenial::NoModelAccess { model: "gpt".to_owned(), team_ids: ids(&["b", "a"]) },
        http(403, "No team has access to the requested model: gpt. Checked teams=[\"a\",\"b\"]. Check `/models` to see all available models.")
    )]
    #[case::team_not_found(
        TeamDenial::TeamNotFound { team_id: "t".to_owned() },
        http(404, "Team doesn't exist in db. Team=t. Create team via `/team/new` call.")
    )]
    #[case::alias_not_found(
        TeamDenial::TeamAliasNotFound { alias: "eng".to_owned() },
        http(404, "Team with alias 'eng' doesn't exist in db. Create team via `/team/new` call.")
    )]
    #[case::alias_ambiguous(
        TeamDenial::TeamAliasAmbiguous { alias: "eng".to_owned() },
        http(400, "Multiple teams found with alias 'eng'. Please use team_id_jwt_field instead or ensure team aliases are unique.")
    )]
    #[case::no_db(
        TeamDenial::NoDb,
        plain("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
    )]
    #[case::no_team_without_id_field(
        no_team_in_token(None),
        plain(
            "No team found in token. Checked team_id field 'unset' and team_alias field 'unset'."
        )
    )]
    #[case::no_team_plain_field(
        no_team_in_token(Some("groups")),
        plain(
            "No team found in token. Checked team_id field 'groups' and team_alias field 'unset'."
        )
    )]
    #[case::no_team_dotted_field(
        no_team_in_token(Some("realm.roles")),
        plain(
            "No team found in token. Checked team_id field 'realm.roles' and team_alias field 'unset'."
        )
    )]
    #[case::no_team_dot_index(
        no_team_in_token(Some("realm.roles.0")),
        plain(
            "No team found in token. Checked team_id field 'realm.roles.0' and team_alias field 'unset'. Hint: \
             dot-notation array indexing (e.g. 'realm.roles.0') is not supported. Use 'realm.roles' instead — LiteLLM \
             automatically uses the first element when the field value is a list."
        )
    )]
    #[case::no_team_dot_wins_over_brackets(
        no_team_in_token(Some("a.b[0]")),
        plain(
            "No team found in token. Checked team_id field 'a.b[0]' and team_alias field 'unset'."
        )
    )]
    #[case::no_team_bracket_index(
        no_team_in_token(Some("roles[12]")),
        plain(
            "No team found in token. Checked team_id field 'roles[12]' and team_alias field 'unset'. Hint: array \
             indexing (e.g. 'roles[12]') is not supported in team_id_jwt_field. Use 'roles' instead — LiteLLM \
             automatically uses the first element when the field value is a list."
        )
    )]
    #[case::no_team_bracket_not_index(
        no_team_in_token(Some("roles[x]")),
        plain(
            "No team found in token. Checked team_id field 'roles[x]' and team_alias field 'unset'."
        )
    )]
    #[case::no_team_bracket_without_base(
        no_team_in_token(Some("[0]")),
        plain("No team found in token. Checked team_id field '[0]' and team_alias field 'unset'.")
    )]
    #[case::header_not_allowed(
        TeamDenial::HeaderTeamNotAllowed { header_value: "h".to_owned(), allowed_team_ids: ids(&["b", "a"]) },
        http(403, "x-litellm-team-id 'h' does not resolve to a team id or a unique team alias in your JWT's allowed teams. Allowed team ids: [\"a\",\"b\"]")
    )]
    #[case::header_not_member(
        TeamDenial::HeaderTeamNotMember { header_value: "h".to_owned() },
        http(403, "x-litellm-team-id 'h' does not resolve to a team id or a unique team alias among your team memberships.")
    )]
    #[case::header_route_not_allowed(
        TeamDenial::HeaderTeamRouteNotAllowed { header_value: "h".to_owned(), route: "/r".to_owned() },
        http(403, "Team 'h' (from x-litellm-team-id header) is not allowed to access route '/r'.")
    )]
    #[case::no_member_team_has_model(
        TeamDenial::NoMemberTeamHasModel { model: "gpt".to_owned() },
        http(403, "No team you are a member of has access to the requested model: gpt. Check `/models` to see the models available to you.")
    )]
    #[case::not_team_member(
        TeamDenial::NotTeamMember,
        http(
            403,
            "User is not a member of any team. Add the user to a team via the LiteLLM UI or API."
        )
    )]
    fn denial_messages_match_the_proxy(#[case] denial: TeamDenial, #[case] expected: PublicDenial) {
        assert_eq!(denial.public(), expected);
    }

    #[rstest]
    #[case::expired(VerifyDenial::Expired, PublicDenial::Proxy {
        status: 401,
        message: "Token Expired".to_owned(),
        error_type: ErrorType::ExpiredKey,
        param: None,
    })]
    #[case::invalid(
        VerifyDenial::Invalid(TokenError::WrongAudience),
        PublicDenial::Plain { message: "Validation fails: Audience doesn't match".to_owned() }
    )]
    #[case::malformed(
        VerifyDenial::Malformed(TokenError::CritUnsupported("exp".to_owned())),
        PublicDenial::Plain { message: "Unsupported critical extension: exp".to_owned() }
    )]
    #[case::unknown_kid(
        VerifyDenial::Key(KeyFailure::NoMatch { keys: Searched::Url(JWKS.to_owned()), kid: None }),
        PublicDenial::Plain { message: format!("No matching public key found. keys={JWKS}, kid=unset") }
    )]
    #[case::assertion(
        VerifyDenial::Assertion(AssertionRejection::Invalid(TokenError::MissingClaim("jti".to_owned()))),
        PublicDenial::Plain { message: "assertion failed verification: Token is missing the \"jti\" claim".to_owned() }
    )]
    fn denials_keep_the_proxy_messages(
        #[case] denial: VerifyDenial,
        #[case] expected: PublicDenial,
    ) {
        assert_eq!(denial.public(), expected);
    }

    #[test]
    fn unmapped_scope_type_is_plain_exception() {
        assert_eq!(
            JwtDenial::Claim(ClaimDenial::UnmappedScopeType { kind: "object" }).public(),
            PublicDenial::Plain {
                message: "Unmapped scope type - object. Supported types - array, string."
                    .to_owned()
            }
        );
    }
}
