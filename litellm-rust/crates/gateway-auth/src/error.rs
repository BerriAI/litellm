use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[derive(Debug, thiserror::Error)]
pub enum KeyError {
    #[error("missing or invalid virtual key")]
    Invalid,
    #[error("virtual key has expired")]
    Expired,
    #[error("virtual key lookup unavailable")]
    Lookup(#[source] Box<dyn std::error::Error + Send + Sync>),
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("gateway auth not configured")]
    Unconfigured,
    #[error("missing or invalid bearer token")]
    InvalidToken,
    #[error("gateway authentication unavailable")]
    Secret(#[from] litellm_secrets::Error),
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::InvalidToken => StatusCode::UNAUTHORIZED,
            Self::Unconfigured | Self::Secret(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, self.to_string()).into_response()
    }
}
