use thiserror::Error;

use crate::error::TransportError;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OcrRequestError {
    #[error("Invalid `req_format`. Expected 'native' or 'litellm'.")]
    RequestFormat,
    #[error("invalid OCR request field: {path}")]
    RequestField { path: String },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OcrResponseError {
    #[error("invalid OCR response field: {path}")]
    ResponseField { path: String },
}

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("{0}")]
    Request(#[from] OcrRequestError),
    #[error("{0}")]
    Response(#[from] OcrResponseError),
    #[error("{0}")]
    Transport(#[from] TransportError),
    #[error("{0}")]
    Public(#[from] crate::Error),
}

impl From<OcrError> for crate::Error {
    fn from(error: OcrError) -> Self {
        match error {
            OcrError::Request(error) => error.into(),
            OcrError::Response(error) => error.into(),
            OcrError::Transport(error) => error.into(),
            OcrError::Public(error) => error,
        }
    }
}
