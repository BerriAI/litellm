use crate::failure::{Phase, Rejection};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("unsupported by the Rust messages route: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Transport(#[from] crate::transport::Error),
    #[error(transparent)]
    Headers(#[from] crate::http_utils::HeaderError),
    #[error("stream framing failed: {0}")]
    StreamFraming(String),
    #[error("Anthropic SSE frame has no data")]
    MissingStreamData,
    #[error("Anthropic stream event is invalid: {0}")]
    InvalidStreamEvent(String),
    #[error("Bedrock event payload is invalid: {0}")]
    InvalidBedrockPayload(String),
    #[error("Bedrock event payload has invalid base64: {0}")]
    InvalidBedrockBase64(String),
}

impl Error {
    pub fn phase(&self) -> Phase<'_> {
        match self {
            Self::InvalidProvider(_)
            | Self::MissingField(_)
            | Self::InvalidRequest(_)
            | Self::Headers(_) => Phase::BeforeProvider(Rejection::InvalidRequest),
            Self::Unsupported(_) => Phase::BeforeProvider(Rejection::Unsupported),
            Self::Auth(_) => Phase::BeforeProvider(Rejection::Credential),
            Self::Transport(error) => error.phase(),
            Self::InvalidResponse(_)
            | Self::StreamFraming(_)
            | Self::MissingStreamData
            | Self::InvalidStreamEvent(_)
            | Self::InvalidBedrockPayload(_)
            | Self::InvalidBedrockBase64(_) => Phase::AfterProvider,
        }
    }
}
