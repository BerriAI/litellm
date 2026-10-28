use std::{io, num::ParseIntError};

use litellm_error::Source;
use thiserror::Error as ThisError;
use tiktoken_rs::{DecodeKeyError, EncodeError};

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("unsupported tokenizer: {0}")]
    UnsupportedTokenizer(String),
    #[error("could not read tiktoken rank file {file}")]
    Read {
        file: &'static str,
        source: io::Error,
    },
    #[error("invalid tiktoken rank file")]
    Ranks(#[from] RankError),
    #[error("could not build the tiktoken encoder")]
    Encoder(#[source] Source),
    #[error("tokenization failed")]
    Encode(#[source] EncodeError),
    #[error("token decoding failed")]
    Decode(#[source] DecodeKeyError),
}

impl Error {
    pub fn is_load(&self) -> bool {
        match self {
            Self::UnsupportedTokenizer(_)
            | Self::Read { .. }
            | Self::Ranks(_)
            | Self::Encoder(_) => true,
            Self::Encode(_) | Self::Decode(_) => false,
        }
    }
}

#[derive(Debug, ThisError)]
pub enum RankError {
    #[error("line without a rank: {0:?}")]
    MissingRank(String),
    #[error("token is not base64")]
    InvalidToken(#[source] base64::DecodeError),
    #[error("rank is not an integer")]
    InvalidRank(#[source] ParseIntError),
    #[error("a rank is assigned to more than one token")]
    DuplicateRank,
    #[error("byte 0x{0:02X} has no token")]
    MissingByte(u8),
}
