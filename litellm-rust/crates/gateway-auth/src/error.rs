use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

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
