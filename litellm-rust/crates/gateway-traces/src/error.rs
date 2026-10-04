use axum::{
    Json,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use litellm_traces_cache::ReadError;
use serde::Serialize;

const RETRY_AFTER_SECONDS: &str = "2";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadFailureCode {
    InvalidRequest,
    TraceChanged,
    TooLarge,
    Unavailable,
}

#[derive(Debug)]
pub struct ReadFailure {
    code: ReadFailureCode,
    message: String,
}

impl<E: std::error::Error> From<ReadError<E>> for ReadFailure {
    fn from(error: ReadError<E>) -> Self {
        let (code, message) = match &error {
            ReadError::InvalidParameters
            | ReadError::InvalidCursor(_)
            | ReadError::AmbiguousTrace => (ReadFailureCode::InvalidRequest, error.to_string()),
            ReadError::TraceChanged => (ReadFailureCode::TraceChanged, error.to_string()),
            ReadError::TooLarge => (
                ReadFailureCode::TooLarge,
                "Trace is too large for this view. Use a filtered trace query.".to_owned(),
            ),
            ReadError::Encode(_) | ReadError::Store(_) => {
                tracing::warn!(%error, "trace read unavailable");
                (
                    ReadFailureCode::Unavailable,
                    "Traces are temporarily unavailable. Please try again.".to_owned(),
                )
            }
        };
        Self { code, message }
    }
}

#[derive(Serialize)]
struct Body<'a> {
    detail: Detail<'a>,
}

#[derive(Serialize)]
struct Detail<'a> {
    code: ReadFailureCode,
    message: &'a str,
}

impl IntoResponse for ReadFailure {
    fn into_response(self) -> Response {
        let status = match self.code {
            ReadFailureCode::InvalidRequest => StatusCode::BAD_REQUEST,
            ReadFailureCode::TraceChanged => StatusCode::CONFLICT,
            ReadFailureCode::TooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            ReadFailureCode::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        };
        let body = Json(Body {
            detail: Detail {
                code: self.code,
                message: &self.message,
            },
        });
        match self.code {
            ReadFailureCode::Unavailable => (
                status,
                [(
                    header::RETRY_AFTER,
                    HeaderValue::from_static(RETRY_AFTER_SECONDS),
                )],
                body,
            )
                .into_response(),
            _ => (status, body).into_response(),
        }
    }
}
