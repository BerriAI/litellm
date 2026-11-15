use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("failed to load tokenizer")]
    Load(#[source] tokenizers::Error),
    #[error("failed to load tokenizer from a tiktoken rank file")]
    Ranks(#[source] RankError),
    #[error("failed to load tokenizer: Unicode character classes are unavailable")]
    UnicodeClasses,
    #[error("tokenization failed")]
    Encode(#[source] tokenizers::Error),
}

impl Error {
    pub fn is_load(&self) -> bool {
        match self {
            Self::Load(_) | Self::Ranks(_) | Self::UnicodeClasses => true,
            Self::Encode(_) => false,
        }
    }
}

#[derive(Debug, ThisError)]
pub enum RankError {
    #[error("rank {0} is reserved")]
    Reserved(u32),
    #[error("byte 0x{0:02X} has no token")]
    MissingByte(u8),
    #[error("line without a rank: {0:?}")]
    MissingRank(String),
    #[error("token is not base64")]
    InvalidToken(#[source] base64::DecodeError),
    #[error("rank is not an integer")]
    InvalidRank(#[source] std::num::ParseIntError),
}
