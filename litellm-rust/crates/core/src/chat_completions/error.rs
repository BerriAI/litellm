#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Transport(#[from] crate::transport::Error),
    #[error(transparent)]
    Headers(#[from] crate::http_utils::HeaderError),
    #[error(transparent)]
    Aws(#[from] litellm_auth_aws::Error),
}

impl From<litellm_providers::chat::Error> for Error {
    fn from(error: litellm_providers::chat::Error) -> Self {
        match error {
            litellm_providers::chat::Error::MissingField(field) => Self::MissingField(field),
            litellm_providers::chat::Error::InvalidRequest(message) => {
                Self::InvalidRequest(message)
            }
            litellm_providers::chat::Error::InvalidResponse(message) => {
                Self::InvalidResponse(message)
            }
            litellm_providers::chat::Error::Unsupported(reason) => Self::Unsupported(reason),
            litellm_providers::chat::Error::Auth(error) => Self::Auth(error),
        }
    }
}
