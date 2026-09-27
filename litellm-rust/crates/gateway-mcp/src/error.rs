use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use rmcp::{
    model::{ErrorCode, ErrorData},
    service::ServiceError,
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    InvalidRequest(String),
    #[error("User not allowed to access this MCP server or operation.")]
    Forbidden,
    #[error("{0}")]
    Configuration(String),
    #[error("MCP upstream request failed")]
    Upstream(#[from] ServiceError),
    #[error("MCP operation returned an unexpected result")]
    UnexpectedResult,
    #[error("MCP request cancelled")]
    Cancelled,
}

impl Error {
    pub fn into_mcp(self) -> ErrorData {
        match self {
            Self::Upstream(ServiceError::McpError(error)) => error,
            Self::InvalidRequest(message) => ErrorData::invalid_params(message, None),
            Self::Forbidden => ErrorData::new(ErrorCode(-32003), self.to_string(), None),
            _ => ErrorData::internal_error(self.to_string(), None),
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::Upstream(_) => StatusCode::BAD_GATEWAY,
            Self::Cancelled => StatusCode::REQUEST_TIMEOUT,
            Self::Configuration(_) | Self::UnexpectedResult => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(json!({"detail": self.to_string()}))).into_response()
    }
}
