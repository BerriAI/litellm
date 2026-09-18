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

impl From<litellm_providers::audio_transcription::Error> for Error {
    fn from(error: litellm_providers::audio_transcription::Error) -> Self {
        match error {
            litellm_providers::audio_transcription::Error::InvalidType { expected, actual } => {
                Self::InvalidType { expected, actual }
            }
            litellm_providers::audio_transcription::Error::MissingField(field) => {
                Self::MissingField(field)
            }
            litellm_providers::audio_transcription::Error::InvalidRequest(message) => {
                Self::InvalidRequest(message)
            }
            litellm_providers::audio_transcription::Error::InvalidResponse(message) => {
                Self::InvalidResponse(message)
            }
            litellm_providers::audio_transcription::Error::Auth(error) => Self::Auth(error),
        }
    }
}
