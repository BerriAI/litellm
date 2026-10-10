use axum::{
    Json,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use litellm_gateway_auth::UiAuthError;

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
    #[error("UI authentication unavailable")]
    Auth(#[from] axum_login::Error<litellm_gateway_auth::UiBackend>),
    #[error("UI session unavailable")]
    Session(#[from] tower_sessions::session::Error),
    #[error("UI token unavailable")]
    Token(#[from] jsonwebtoken::errors::Error),
    #[error("Invalid UI session")]
    Unauthorized(#[from] UiAuthError),
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::InvalidCredentials | Self::Unauthorized(UiAuthError::Unauthorized) => {
                StatusCode::UNAUTHORIZED
            }
            Self::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let response = (
            status,
            Json(ErrorBody {
                error: ErrorMessage {
                    message: self.to_string(),
                },
            }),
        )
            .into_response();

        if matches!(self, Self::RateLimited) {
            return ([(header::RETRY_AFTER, "60")], response).into_response();
        }

        response
    }
}
