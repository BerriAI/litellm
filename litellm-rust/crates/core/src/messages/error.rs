use litellm_providers::base_llm::chat::transformation::Error as LlmError;

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

impl From<LlmError> for Error {
    fn from(error: LlmError) -> Self {
        match error {
            error @ LlmError::InvalidType { .. } => Self::InvalidRequest(error.to_string()),
            LlmError::MissingField(field) => Self::MissingField(field),
            LlmError::InvalidRequest(message) => Self::InvalidRequest(message),
            LlmError::InvalidResponse(message) => Self::InvalidResponse(message),
            LlmError::Unsupported(reason) => Self::Unsupported(reason),
            LlmError::Auth(error) => Self::Auth(error),
        }
    }
}

impl Error {
    pub fn is_request(&self) -> bool {
        match self {
            Self::InvalidProvider(_)
            | Self::MissingField(_)
            | Self::InvalidRequest(_)
            | Self::Unsupported(_)
            | Self::Headers(_) => true,
            Self::Auth(error) => !matches!(error, litellm_auth::Error::MissingApiKey { .. }),
            _ => false,
        }
    }

    pub fn is_response(&self) -> bool {
        matches!(
            self,
            Self::InvalidResponse(_)
                | Self::StreamFraming(_)
                | Self::MissingStreamData
                | Self::InvalidStreamEvent(_)
                | Self::InvalidBedrockPayload(_)
                | Self::InvalidBedrockBase64(_)
        )
    }
}
