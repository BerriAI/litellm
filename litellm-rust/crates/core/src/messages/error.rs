use litellm_llms::base_llm::chat::transformation::Error as LlmError;

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
    Transport(#[from] litellm_llms::custom_httpx::transport::Error),
    #[error(transparent)]
    Headers(#[from] litellm_llms::custom_httpx::http_handler::HeaderError),
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
        matches!(self, Self::InvalidResponse(_))
    }
}
