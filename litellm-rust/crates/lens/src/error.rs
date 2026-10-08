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
    #[error("Trace content ended before its truncated span was complete")]
    EvidenceIncomplete,
    #[error("Trace span disappeared during a content read")]
    EvidenceSpanMissing,
    #[error("Trace content repeated a pagination cursor")]
    EvidenceCursorRepeated,
    #[error("Trace content returned a different execution")]
    EvidenceExecutionChanged,
    #[error("Trace content could not be read. Check Lens storage availability.")]
    EvidenceUnavailable,
    #[error("Python computation cancelled")]
    PythonCancelled,
    #[error("Python exceeded its 60-second elapsed-time limit")]
    PythonTimedOut,
    #[error("Python analysis requires the Linux Lens image with Landlock and seccomp support")]
    PythonUnsupportedPlatform,
    #[error("Python exceeded its scratch directory-depth limit")]
    PythonScratchTooDeep,
    #[error("Python exceeded its scratch storage or file-count limit")]
    PythonScratchTooLarge,
    #[error("Python output exceeded 4 MiB on one stream. Print a smaller result.")]
    PythonOutputTooLarge,
    #[error("Python syscall policy is missing from the worker image")]
    PythonPolicyMissing,
    #[error("Python resource monitoring failed: {0}")]
    PythonMonitorIo(#[source] std::io::Error),
    #[error(
        "The Lens task alone exceeds the model context window. Use a model with more context or shorten the investigation instructions."
    )]
    TaskContext,
    #[error(
        "The compacted task exceeds the model context window. Use a larger-context model or shorter instructions."
    )]
    CompactedContext,
    #[error("History reply exceeds 32 MiB. Select a smaller turn range, then a character range.")]
    HistoryTooLarge,
    #[error(
        "Investigation journal exceeded 512 MiB. Reduce the sample or split the investigation."
    )]
    JournalTooLarge,
    #[error("Python input exceeds 256 MiB. Select fewer executions or spans.")]
    PythonInputTooLarge,
    #[error("Unknown span IDs in Python request")]
    UnknownPythonSpan,
    #[error("Unknown execution IDs in Python request")]
    UnknownPythonExecution,
    #[error(
        "Tool output exceeds 8 MiB. Select narrower spans or a character range, or use Python to summarize the evidence."
    )]
    ToolOutputTooLarge,
    #[error("The smallest candidate comparison exceeds model context. Use a larger-context model.")]
    CandidateContext,
    #[error("The analysis conversation exceeds the model context window.")]
    Context(Box<crate::wire::ModelRequest>),
    #[error("invalid Lens configuration: {0}")]
    Configuration(&'static str),
    #[error("credential is invalid or expired")]
    Unauthorized,
    #[error("tracing credentials have not propagated yet")]
    CredentialsPending,
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
            Self::CredentialsPending => StatusCode::TOO_MANY_REQUESTS,
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
            StatusCode::TOO_MANY_REQUESTS => "pending_credentials",
            _ => "unavailable",
        };
        let mut response = (status, Json(serde_json::json!({"code": code}))).into_response();
        if matches!(
            status,
            StatusCode::SERVICE_UNAVAILABLE | StatusCode::TOO_MANY_REQUESTS
        ) {
            response
                .headers_mut()
                .insert("retry-after", http::HeaderValue::from_static("5"));
        }
        response
    }
}
