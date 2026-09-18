use crate::failure::{Phase, Rejection};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("routing error: {0}")]
    Routing(String),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Transport(#[from] crate::transport::Error),
    #[error(transparent)]
    Headers(#[from] crate::http_utils::HeaderError),
}

impl Error {
    pub fn phase(&self) -> Phase<'_> {
        match self {
            Self::InvalidProvider(_)
            | Self::InvalidRequest(_)
            | Self::Routing(_)
            | Self::Headers(_) => Phase::BeforeProvider(Rejection::InvalidRequest),
            Self::Auth(_) => Phase::BeforeProvider(Rejection::Credential),
            Self::Transport(error) => error.phase(),
            Self::InvalidResponse(_) => Phase::AfterProvider,
        }
    }
}
