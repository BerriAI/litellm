use crate::failure::{Phase, Rejection};

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

impl Error {
    pub fn phase(&self) -> Phase<'_> {
        match self {
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidProvider(_)
            | Self::InvalidRequest(_)
            | Self::Headers(_) => Phase::BeforeProvider(Rejection::InvalidRequest),
            Self::Unsupported(_) => Phase::BeforeProvider(Rejection::Unsupported),
            Self::Auth(_) | Self::Aws(_) => Phase::BeforeProvider(Rejection::Credential),
            Self::Transport(error) => error.phase(),
            Self::InvalidResponse(_) => Phase::AfterProvider,
        }
    }
}
