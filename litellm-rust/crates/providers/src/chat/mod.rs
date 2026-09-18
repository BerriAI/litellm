use thiserror::Error;

pub const EMPTY_TEXT_PLACEHOLDER: &str = " ";

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum Error {
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
}

pub mod conversation;
pub mod response_utils;
pub mod types;
