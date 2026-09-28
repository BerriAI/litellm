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
