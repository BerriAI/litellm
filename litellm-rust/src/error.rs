//! OCR error type. Pure: no PyO3.

use std::fmt;

/// Errors that can occur while performing an OCR call.
#[derive(Debug)]
pub enum OcrError {
    /// Missing/invalid credentials or other env validation failure.
    Auth(String),
    /// A transform (request build or response parse) failed.
    Transform(String),
    /// Upstream returned a non-2xx status.
    Http { status: u16, body: String },
    /// Network-level failure (connection, timeout, etc.).
    Network(String),
    /// Failed to parse the upstream response body as JSON.
    Parse(String),
}

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OcrError::Auth(msg) => write!(f, "{msg}"),
            OcrError::Transform(msg) => write!(f, "{msg}"),
            OcrError::Http { status, body } => {
                write!(f, "Mistral OCR request failed with status {status}: {body}")
            }
            OcrError::Network(msg) => write!(f, "Mistral OCR network error: {msg}"),
            OcrError::Parse(msg) => write!(f, "Error parsing Mistral OCR response: {msg}"),
        }
    }
}

impl std::error::Error for OcrError {}
