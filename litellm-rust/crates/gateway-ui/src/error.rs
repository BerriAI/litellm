use axum::{
    Json,
    extract::rejection::{JsonRejection, QueryRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

#[derive(serde::Serialize)]
struct ErrorBody {
    error: ErrorMessage,
}

#[derive(serde::Serialize)]
struct ErrorMessage {
    message: String,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid username or password")]
    InvalidCredentials,
    #[error("Too many login attempts, try again in one minute")]
    RateLimited,
    #[error("invalid login request")]
    Body(#[source] JsonRejection),
    #[error("invalid query string")]
    Query(#[source] QueryRejection),
    #[error("UI login unavailable")]
    Login(#[from] axum_login::Error<litellm_gateway_auth::UiBackend>),
    #[error("UI session unavailable")]
    Session(#[from] tower_sessions::session::Error),
    #[error("UI token unavailable")]
    Token(#[from] jsonwebtoken::errors::Error),
    #[error(transparent)]
    Auth(#[from] litellm_gateway_auth::Error),
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidCredentials => StatusCode::UNAUTHORIZED,
            Self::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            Self::Body(rejection) => rejection.status(),
            Self::Query(rejection) => rejection.status(),
            Self::Auth(error) => error.status(),
            Self::Login(_) | Self::Session(_) | Self::Token(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = Json(ErrorBody {
            error: ErrorMessage {
                message: litellm_error::Report::new(&self).to_string(),
            },
        });
        match status {
            StatusCode::TOO_MANY_REQUESTS => {
                (status, [(header::RETRY_AFTER, "60")], body).into_response()
            }
            _ => (status, body).into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth::Error as AuthError;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::rejected_session(Error::Auth(AuthError::InvalidSession), StatusCode::UNAUTHORIZED)]
    #[case::session_layer_missing(
        Error::Auth(AuthError::MissingSessionLayer),
        StatusCode::INTERNAL_SERVER_ERROR
    )]
    #[case::operation_outside_session_scope(
        Error::Auth(AuthError::Forbidden),
        StatusCode::FORBIDDEN
    )]
    #[case::rate_limited(Error::RateLimited, StatusCode::TOO_MANY_REQUESTS)]
    fn status_follows_who_is_at_fault(#[case] error: Error, #[case] status: StatusCode) {
        assert_eq!(error.status(), status);
    }

    #[rstest]
    #[case::rate_limited(Error::RateLimited, true)]
    #[case::invalid_credentials(Error::InvalidCredentials, false)]
    fn only_rate_limits_ask_to_retry_later(#[case] error: Error, #[case] retry_after: bool) {
        assert_eq!(
            error
                .into_response()
                .headers()
                .contains_key(header::RETRY_AFTER),
            retry_after
        );
    }
}
