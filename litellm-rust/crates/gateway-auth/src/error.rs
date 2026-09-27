use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("operation is not permitted")]
    Forbidden,
    #[error("credential has expired")]
    Expired,
    #[error("authenticated request context is missing")]
    MissingIdentity,
    #[error("authentication service unavailable")]
    Unavailable,
    #[error("gateway auth not configured")]
    Unconfigured,
    #[error("missing or invalid bearer token")]
    InvalidToken,
    #[error("gateway authentication unavailable")]
    Secret(#[from] litellm_secrets::Error),
}

impl Error {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidToken | Self::Expired => StatusCode::UNAUTHORIZED,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::MissingIdentity | Self::Unconfigured | Self::Secret(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.status(), self.to_string()).into_response()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum UiAuthError {
    #[error("UI_USERNAME and UI_PASSWORD must be nonempty")]
    Unconfigured,
    #[error("invalid UI credentials or session")]
    Unauthorized,
    #[error("UI authentication unavailable")]
    Unavailable,
    #[error("UI session unavailable")]
    Session(#[from] tower_sessions::session::Error),
}

impl IntoResponse for UiAuthError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, self.to_string()).into_response()
    }
}
