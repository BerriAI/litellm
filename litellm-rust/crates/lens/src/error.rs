use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use litellm_traces_cache::ReadError;
use litellm_traces_clickhouse::Error as StoreError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{schema} response invalid after two attempts: {detail}")]
    ModelValidation {
        schema: &'static str,
        detail: String,
    },
    #[error(
        "The gateway rejected a worker request (HTTP {status}): {}", diagnostic.as_deref().unwrap_or("Check worker access, model availability and investigation budget.")
    )]
    Control {
        status: u16,
        retry_after: Option<u64>,
        diagnostic: Option<String>,
    },
    #[error(
        "The worker received an invalid response. Check that the gateway and worker versions match."
    )]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Analysis(&'static str),
    #[error("The analysis conversation exceeds the model context window.")]
    Context(Box<crate::wire::ModelRequest>),
    #[error("invalid Lens configuration: {0}")]
    Configuration(&'static str),
    #[error("credential is invalid or expired")]
    Unauthorized,
    #[error("Lens is temporarily unavailable")]
    Unavailable,
    #[error("request exceeds the size limit")]
    TooLarge,
    #[error("invalid request")]
    InvalidRequest,
    #[error("trace changed; restart pagination")]
    TraceChanged,
    #[error("trace storage failed")]
    Storage(#[from] StoreError),
    #[error("HTTP client configuration failed")]
    Http(#[from] litellm_http::Error),
    #[error("HTTP request failed")]
    Request(#[from] reqwest::Error),
    #[error("service I/O failed")]
    Io(#[from] std::io::Error),
}

impl Error {
    pub fn is_control_failure(&self) -> bool {
        matches!(self, Self::Control { .. } | Self::Request(_))
    }
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::Request(_)
                | Self::Control {
                    status: 429 | 502 | 503 | 504,
                    ..
                }
        )
    }

    pub fn status(&self) -> StatusCode {
        match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::TooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::InvalidRequest => StatusCode::BAD_REQUEST,
            Self::TraceChanged => StatusCode::CONFLICT,
            Self::Storage(error) => storage_status(error),
            _ => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

fn storage_status(error: &StoreError) -> StatusCode {
    use litellm_storage_clickhouse::Error as TransportError;
    match error {
        StoreError::Decode(litellm_traces::Error::TooLarge)
        | StoreError::InsertTooLarge
        | StoreError::Storage(TransportError::InsertTooLarge) => StatusCode::PAYLOAD_TOO_LARGE,
        StoreError::Decode(_)
        | StoreError::InvalidRow
        | StoreError::InvalidQuery
        | StoreError::InvalidParameters
        | StoreError::InvalidScope
        | StoreError::Storage(TransportError::QueryFailed(400 | 404)) => StatusCode::BAD_REQUEST,
        StoreError::Cached(error) => storage_status(error),
        _ => StatusCode::SERVICE_UNAVAILABLE,
    }
}

impl From<ReadError<StoreError>> for Error {
    fn from(error: ReadError<StoreError>) -> Self {
        match error {
            ReadError::InvalidParameters
            | ReadError::InvalidCursor(_)
            | ReadError::AmbiguousTrace => Self::InvalidRequest,
            ReadError::TraceChanged => Self::TraceChanged,
            ReadError::TooLarge => Self::TooLarge,
            ReadError::Store(error) => Self::Storage(StoreError::Cached(error)),
            ReadError::Encode(_) => Self::Unavailable,
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = self.status();
        let code = match status {
            StatusCode::BAD_REQUEST => "invalid_request",
            StatusCode::CONFLICT => "trace_changed",
            StatusCode::PAYLOAD_TOO_LARGE => "too_large",
            StatusCode::UNAUTHORIZED => "unauthorized",
            _ => "unavailable",
        };
        let mut response = (status, Json(serde_json::json!({"code": code}))).into_response();
        if status == StatusCode::SERVICE_UNAVAILABLE {
            response
                .headers_mut()
                .insert("retry-after", http::HeaderValue::from_static("5"));
        }
        response
    }
}
