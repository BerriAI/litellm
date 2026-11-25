use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_error::Source;
use litellm_gateway_auth_core::{
    self as auth_core, AgentIdentityFailure, JwtDenial, KeyDenial, KeyFailure, NotACaller,
    RbacDenial, TeamDenial, VerifyDenial,
};

#[derive(Debug, thiserror::Error, strum::IntoStaticStr)]
pub enum Error {
    #[error("missing or invalid bearer token")]
    #[strum(serialize = "invalid_token")]
    InvalidToken,
    #[error("missing or invalid API key")]
    #[strum(serialize = "invalid_api_key")]
    InvalidApiKey,
    #[error("credential has expired")]
    #[strum(serialize = "expired")]
    Expired,
    #[error("operation is not permitted")]
    #[strum(serialize = "forbidden")]
    Forbidden,
    /// The credential is held, with something this server does not serve.
    #[error("credential is not served by this server")]
    #[strum(serialize = "not_native")]
    NotNative,
    #[error("authentication service unavailable")]
    #[strum(serialize = "unavailable")]
    Unavailable(#[source] Source),
    #[error("gateway auth not configured")]
    #[strum(serialize = "unconfigured")]
    Unconfigured,
    #[error("authenticated request context is missing")]
    #[strum(serialize = "missing_identity")]
    MissingIdentity,
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidToken | Self::InvalidApiKey | Self::Expired => StatusCode::UNAUTHORIZED,
            Self::Forbidden | Self::NotNative => StatusCode::FORBIDDEN,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Unconfigured | Self::MissingIdentity => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn is_denied(&self) -> bool {
        match self {
            Self::InvalidToken
            | Self::InvalidApiKey
            | Self::Expired
            | Self::Forbidden
            | Self::NotNative => true,
            Self::Unavailable(_) | Self::Unconfigured | Self::MissingIdentity => false,
        }
    }
}

/// What this server answers a credential the core did not authenticate. The core's own text is
/// the proxy's, which answers instead wherever one stands behind this server.
impl From<auth_core::Error> for Error {
    fn from(error: auth_core::Error) -> Self {
        match error {
            auth_core::Error::HostDeclined => Self::NotNative,
            auth_core::Error::HostFailed => Self::Unavailable(Source::new(error)),
            auth_core::Error::Key(KeyDenial::Expired { .. }) => Self::Expired,
            auth_core::Error::Key(
                KeyDenial::MasterKeyOnlyRoute { .. }
                | KeyDenial::UserAuthDisabled
                | KeyDenial::UserDeactivated { .. },
            )
            | auth_core::Error::PremiumRequired(_)
            | auth_core::Error::BudgetExceeded(_)
            | auth_core::Error::EndUserBudgetExceeded(_) => Self::Forbidden,
            auth_core::Error::Jwt(ref denial) => match denial {
                JwtDenial::Verify(VerifyDenial::Expired) => Self::Expired,
                JwtDenial::Verify(VerifyDenial::Key(KeyFailure::Unreachable { .. }))
                | JwtDenial::MappingRaceVanished { .. }
                | JwtDenial::AgentIdentity(AgentIdentityFailure {
                    policy_unavailable: true,
                    ..
                }) => Self::Unavailable(Source::new(error)),
                JwtDenial::MappingNeedsDb { .. } | JwtDenial::CustomValidateUnavailable => {
                    Self::Unconfigured
                }
                JwtDenial::Rbac(
                    RbacDenial::UnmatchedRole
                    | RbacDenial::RoleRoute { .. }
                    | RbacDenial::RoleModel { .. }
                    | RbacDenial::ScopeModel { .. },
                )
                | JwtDenial::Team(
                    TeamDenial::TeamAliasAmbiguous { .. }
                    | TeamDenial::NoTeamsInToken
                    | TeamDenial::PassthroughRouteDenied { .. }
                    | TeamDenial::RouteNotAllowed { .. }
                    | TeamDenial::NoModelAccess { .. }
                    | TeamDenial::HeaderTeamNotAllowed { .. }
                    | TeamDenial::HeaderTeamNotMember { .. }
                    | TeamDenial::HeaderTeamRouteNotAllowed { .. }
                    | TeamDenial::NoMemberTeamHasModel { .. }
                    | TeamDenial::NotTeamMember,
                )
                | JwtDenial::EmailDomainNotAllowed { .. }
                | JwtDenial::CustomValidate
                | JwtDenial::ManagedIssuerUntrusted
                | JwtDenial::ManagedRouteNotAllowed
                | JwtDenial::UnregisteredAgent { .. }
                | JwtDenial::GrantingTeamRouteNotAllowed
                | JwtDenial::NoObjectId
                | JwtDenial::ManagedAgentNeedsDirectJwt
                | JwtDenial::MappingClaimMissing { .. }
                | JwtDenial::NoMapping { .. }
                | JwtDenial::NoEligibleKey
                | JwtDenial::AgentIdentity(AgentIdentityFailure {
                    policy_unavailable: false,
                    ..
                }) => Self::Forbidden,
                JwtDenial::Verify(
                    VerifyDenial::Malformed(_)
                    | VerifyDenial::Key(_)
                    | VerifyDenial::Invalid(_)
                    | VerifyDenial::Assertion(_)
                    | VerifyDenial::UserinfoNotConfigured
                    | VerifyDenial::IdJagNotBearer,
                )
                | JwtDenial::Rbac(RbacDenial::AdminRoute { .. })
                | JwtDenial::Team(
                    TeamDenial::TeamNotFound { .. }
                    | TeamDenial::TeamAliasNotFound { .. }
                    | TeamDenial::NoDb
                    | TeamDenial::NoTeamInToken { .. },
                )
                | JwtDenial::Claim(_)
                | JwtDenial::UserNotFound { .. }
                | JwtDenial::UserDeactivated { .. } => Self::InvalidToken,
            },
            auth_core::Error::Key(
                KeyDenial::MissingKey
                | KeyDenial::MalformedKey
                | KeyDenial::NoDb
                | KeyDenial::NoDbConfigured
                | KeyDenial::NotVirtualKey { .. }
                | KeyDenial::NotFound { .. }
                | KeyDenial::Blocked
                | KeyDenial::NoToken,
            ) => Self::InvalidToken,
        }
    }
}

/// An identity the core authenticated and this server serves no request for. Only a missing
/// master key is the server's own fault. The others are held by a host that serves them.
impl From<NotACaller> for Error {
    fn from(identity: NotACaller) -> Self {
        match identity {
            NotACaller::NoMasterKey => Self::Unconfigured,
            NotACaller::UserAuthRoute | NotACaller::CachedAdmin | NotACaller::AutoRegisteredJwt => {
                Self::NotNative
            }
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.status(), self.to_string()).into_response()
    }
}

#[cfg(test)]
mod tests {
    use jiff::Timestamp;
    use litellm_config::UnregisteredJwtClientBehavior;
    use litellm_gateway_auth_core::{KeyHash, PremiumRequired};
    use rstest::rstest;

    use super::*;

    fn expired() -> auth_core::Error {
        KeyDenial::Expired {
            expires: Timestamp::UNIX_EPOCH,
            now: Timestamp::UNIX_EPOCH,
            key: "sk-...".to_owned(),
        }
        .into()
    }

    fn not_found() -> auth_core::Error {
        KeyDenial::NotFound {
            hash: KeyHash::new("hash"),
        }
        .into()
    }

    fn master_key_only() -> auth_core::Error {
        KeyDenial::MasterKeyOnlyRoute {
            route: "/route".to_owned(),
        }
        .into()
    }

    fn agent_identity_refused(policy_unavailable: bool) -> auth_core::Error {
        JwtDenial::AgentIdentity(AgentIdentityFailure {
            policy_unavailable,
            message: "refused".to_owned(),
        })
        .into()
    }

    #[rstest]
    #[case::a_host_that_failed(auth_core::Error::HostFailed, "unavailable")]
    #[case::a_key_the_host_does_not_serve(auth_core::Error::HostDeclined, "not_native")]
    #[case::an_expired_key(expired(), "expired")]
    #[case::a_route_for_the_master_key_alone(master_key_only(), "forbidden")]
    #[case::a_feature_the_license_lacks(PremiumRequired::Jwt.into(), "forbidden")]
    #[case::no_key(KeyDenial::MissingKey.into(), "invalid_token")]
    #[case::a_key_nothing_can_look_up(KeyDenial::NoDbConfigured.into(), "invalid_token")]
    #[case::an_unknown_key(not_found(), "invalid_token")]
    #[case::a_blocked_key(KeyDenial::Blocked.into(), "invalid_token")]
    #[case::an_expired_jwt(JwtDenial::Verify(VerifyDenial::Expired).into(), "expired")]
    #[case::an_unreachable_jwks(
        JwtDenial::Verify(VerifyDenial::Key(KeyFailure::Unreachable {
            error: "connection refused".to_owned(),
        }))
        .into(),
        "unavailable"
    )]
    #[case::a_jwt_with_no_jwks_url_configured(
        JwtDenial::Verify(VerifyDenial::Key(KeyFailure::MissingUrl)).into(),
        "invalid_token"
    )]
    #[case::a_jwt_with_unmatched_role(JwtDenial::Rbac(RbacDenial::UnmatchedRole).into(), "forbidden")]
    #[case::an_admin_jwt_on_a_route_admins_may_not_call(
        JwtDenial::Rbac(RbacDenial::AdminRoute {
            route: "/v1/messages".to_owned(),
            allowed: Vec::new(),
        })
        .into(),
        "invalid_token"
    )]
    #[case::a_jwt_with_disallowed_team_route(
        JwtDenial::Team(TeamDenial::RouteNotAllowed {
            route: "/v1/messages".to_owned(),
            team_id: "team-1".to_owned(),
            model: "claude".to_owned(),
        })
        .into(),
        "forbidden"
    )]
    #[case::a_jwt_with_missing_team(
        JwtDenial::Team(TeamDenial::TeamNotFound {
            team_id: "team-1".to_owned(),
        })
        .into(),
        "invalid_token"
    )]
    #[case::a_jwt_with_unavailable_custom_validate(
        JwtDenial::CustomValidateUnavailable.into(),
        "unconfigured"
    )]
    #[case::a_jwt_for_a_deactivated_user(
        JwtDenial::UserDeactivated { user_id: None }.into(),
        "invalid_token"
    )]
    #[case::a_jwt_key_mapping_with_no_database(
        JwtDenial::MappingNeedsDb {
            behavior: UnregisteredJwtClientBehavior::AutoRegister,
        }
        .into(),
        "unconfigured"
    )]
    #[case::a_jwt_key_mapping_whose_race_winner_vanished(
        JwtDenial::MappingRaceVanished {
            behavior: UnregisteredJwtClientBehavior::AutoRegister,
        }
        .into(),
        "unavailable"
    )]
    #[case::an_agent_the_identity_policy_refused(agent_identity_refused(false), "forbidden")]
    #[case::an_agent_whose_identity_policy_is_unavailable(
        agent_identity_refused(true),
        "unavailable"
    )]
    fn a_credential_the_core_did_not_authenticate_is_answered_in_this_servers_words(
        #[case] denied: auth_core::Error,
        #[case] answered: &str,
    ) {
        assert_eq!(<&'static str>::from(Error::from(denied)), answered);
    }

    #[rstest]
    #[case::no_master_key(NotACaller::NoMasterKey, "unconfigured")]
    #[case::the_user_auth_route(NotACaller::UserAuthRoute, "not_native")]
    #[case::a_cached_admin_key(NotACaller::CachedAdmin, "not_native")]
    #[case::an_auto_registered_jwt(NotACaller::AutoRegisteredJwt, "not_native")]
    fn an_identity_no_host_serves_is_refused(#[case] identity: NotACaller, #[case] answered: &str) {
        assert_eq!(<&'static str>::from(Error::from(identity)), answered);
    }

    #[rstest]
    #[tokio::test]
    async fn rejection_body_hides_backend_causes() {
        let error =
            Error::Unavailable(Source::new(std::io::Error::other("private-backend-detail")));
        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("private-backend-detail")
        );
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert!(
            !std::str::from_utf8(&body)
                .unwrap()
                .contains("private-backend-detail")
        );
    }
}
