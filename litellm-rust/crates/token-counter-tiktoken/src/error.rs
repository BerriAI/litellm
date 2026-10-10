use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
#[error("unsupported tokenizer: {0}")]
pub struct UnsupportedTokenizer(pub String);
