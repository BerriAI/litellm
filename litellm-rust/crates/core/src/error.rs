use thiserror::Error;

pub type CoreResult<T> = Result<T, CoreError>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("{0}")]
    Auth(String),
    #[error("OCR request failed with status {status}: {body}")]
    Http { status: u16, body: String },
    #[error("OCR network error: {0}")]
    Network(String),
    #[error("routing error: {0}")]
    Routing(String),
}
