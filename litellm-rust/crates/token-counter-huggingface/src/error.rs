use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("failed to load tokenizer: {0}")]
    Load(#[source] tokenizers::Error),
    #[error("failed to download tokenizer: {0}")]
    Download(#[source] tokenizers::Error),
    #[error("tokenization failed: {0}")]
    Encode(#[source] tokenizers::Error),
    #[error("token decoding failed: {0}")]
    Decode(#[source] tokenizers::Error),
}
