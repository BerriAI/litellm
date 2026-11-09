use std::string::FromUtf8Error;

use litellm_error::Source;
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[cfg(feature = "fast")]
    #[error(transparent)]
    Fast(#[from] litellm_token_counter_fast::Error),
    #[cfg(feature = "huggingface")]
    #[error(transparent)]
    HuggingFace(#[from] litellm_token_counter_huggingface::Error),
    #[cfg(feature = "tiktoken")]
    #[error(transparent)]
    Tiktoken(#[from] litellm_token_counter_tiktoken::Error),
    #[error("unsupported by the rust token counter")]
    Unsupported(#[from] Unsupported),
    #[error("token counting task failed")]
    Task(#[source] Source),
}

impl Error {
    pub fn is_unsupported(&self) -> bool {
        match self {
            Self::Unsupported(_) => true,
            #[cfg(feature = "fast")]
            Self::Fast(_) => false,
            #[cfg(feature = "huggingface")]
            Self::HuggingFace(_) => false,
            #[cfg(feature = "tiktoken")]
            Self::Tiktoken(_) => false,
            Self::Task(_) => false,
        }
    }

    pub fn is_load(&self) -> bool {
        match self {
            #[cfg(feature = "fast")]
            Self::Fast(error) => error.is_load(),
            #[cfg(feature = "huggingface")]
            Self::HuggingFace(error) => error.is_load(),
            #[cfg(feature = "tiktoken")]
            Self::Tiktoken(error) => error.is_load(),
            Self::Unsupported(_) | Self::Task(_) => false,
        }
    }
}

#[derive(Debug, ThisError)]
pub enum Unsupported {
    #[error("request body could not be parsed")]
    RequestParse(#[source] serde_json::Error),
    #[error("request has no countable input")]
    MissingInput,
    #[error("float text values are counted by the python path")]
    FloatText,
    #[error("content block type is counted by the python path")]
    ContentBlock,
    #[error("array parameter without items")]
    ArrayItems,
    #[error("text value could not be serialized")]
    JsonSerialization(#[source] serde_json::Error),
    #[error("serialized text value is not UTF-8")]
    JsonUtf8(#[source] FromUtf8Error),
}
