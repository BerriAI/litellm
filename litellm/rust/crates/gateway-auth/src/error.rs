use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_error::Source;
use litellm_gateway_auth_core::{self as auth_core, KeyDenial, NotACaller};

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
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Unconfigured | Self::MissingIdentity => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn is_denied(&self) -> bool {
        match self {
            Self::InvalidToken | Self::InvalidApiKey | Self::Expired | Self::Forbidden => true,
            Self::Unavailable(_) | Self::Unconfigured | Self::MissingIdentity => false,
        }
    }
}

/// What this server answers a credential the core did not authenticate. The core's own text is
/// the proxy's, which answers instead wherever one stands behind this server.
impl From<auth_core::Error> for Error {
    fn from(error: auth_core::Error) -> Self {
        match error {
            auth_core::Error::HostFailed => Self::Unavailable(Source::new(error)),
            auth_core::Error::Key(KeyDenial::Expired { .. }) => Self::Expired,
            auth_core::Error::Key(
                KeyDenial::MasterKeyOnlyRoute { .. }
                | KeyDenial::UserAuthDisabled
                | KeyDenial::UserDeactivated { .. },
            )
            | auth_core::Error::PremiumRequired(_)
            | auth_core::Error::BudgetExceeded(_) => Self::Forbidden,
            auth_core::Error::Key(
                KeyDenial::MissingKey
                | KeyDenial::MalformedKey
                | KeyDenial::NoDb
                | KeyDenial::NoDbConfigured
                | KeyDenial::NotVirtualKey { .. }
                | KeyDenial::NotFound { .. }
                | KeyDenial::Blocked
                | KeyDenial::NoToken,
            )
            | auth_core::Error::Jwt(_) => Self::InvalidToken,
        }
    }
}

/// An identity the core authenticated and this server serves no request for. Only a missing
/// master key is the server's own fault.
impl From<NotACaller> for Error {
    fn from(identity: NotACaller) -> Self {
        match identity {
            NotACaller::NoMasterKey => Self::Unconfigured,
            NotACaller::UserAuthRoute | NotACaller::CachedAdmin | NotACaller::Jwt => {
                Self::InvalidToken
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

    #[rstest]
    #[case::a_host_that_failed(auth_core::Error::HostFailed, "unavailable")]
    #[case::an_expired_key(expired(), "expired")]
    #[case::a_route_for_the_master_key_alone(master_key_only(), "forbidden")]
    #[case::a_feature_the_license_lacks(PremiumRequired::Jwt.into(), "forbidden")]
    #[case::no_key(KeyDenial::MissingKey.into(), "invalid_token")]
    #[case::a_key_nothing_can_look_up(KeyDenial::NoDbConfigured.into(), "invalid_token")]
    #[case::an_unknown_key(not_found(), "invalid_token")]
    #[case::a_blocked_key(KeyDenial::Blocked.into(), "invalid_token")]
    fn a_credential_the_core_did_not_authenticate_is_answered_in_this_servers_words(
        #[case] denied: auth_core::Error,
        #[case] answered: &str,
    ) {
        assert_eq!(<&'static str>::from(Error::from(denied)), answered);
    }

    #[rstest]
    #[case::no_master_key(NotACaller::NoMasterKey, "unconfigured")]
    #[case::the_user_auth_route(NotACaller::UserAuthRoute, "invalid_token")]
    #[case::a_cached_admin_key(NotACaller::CachedAdmin, "invalid_token")]
    #[case::a_jwt(NotACaller::Jwt, "invalid_token")]
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
