use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("failed to load tokenizer")]
    Load(#[source] tokenizers::Error),
    #[error("invalid encoding JSON")]
    EncodingJson(#[source] serde_json::Error),
    #[error("tokenization failed")]
    Encode(#[source] tokenizers::Error),
    #[error("token decoding failed")]
    Decode(#[source] tokenizers::Error),
}

impl Error {
    pub fn is_load(&self) -> bool {
        match self {
            Self::Load(_) | Self::EncodingJson(_) => true,
            Self::Encode(_) | Self::Decode(_) => false,
        }
    }
}
