use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("failed to load tokenizer: {0}")]
    Load(#[source] tokenizers::Error),
    #[error("failed to load tokenizer: tiktoken rank file: {0}")]
    Ranks(String),
    #[error("failed to load tokenizer: Unicode character classes are unavailable")]
    UnicodeClasses,
    #[error("tokenization failed: {0}")]
    Encode(#[source] tokenizers::Error),
}
