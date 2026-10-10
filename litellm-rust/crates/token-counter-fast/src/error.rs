use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("failed to load tokenizer: {0}")]
    Load(#[source] tokenizers::Error),
    #[error("failed to load tokenizer: tiktoken rank file: {0}")]
    Ranks(#[source] RankError),
    #[error("failed to load tokenizer: Unicode character classes are unavailable")]
    UnicodeClasses,
    #[error("tokenization failed: {0}")]
    Encode(#[source] tokenizers::Error),
}

#[derive(Debug, ThisError)]
pub enum RankError {
    #[error("rank {0} is reserved")]
    Reserved(u32),
    #[error("byte 0x{0:02X} has no token")]
    MissingByte(u8),
    #[error("line without a rank: {0:?}")]
    MissingRank(String),
    #[error("token is not base64: {0}")]
    InvalidToken(#[source] base64::DecodeError),
    #[error("rank is not an integer: {0}")]
    InvalidRank(#[source] std::num::ParseIntError),
}
