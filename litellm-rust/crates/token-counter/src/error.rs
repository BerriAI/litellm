use std::string::FromUtf8Error;

use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("unsupported tokenizer: {0}")]
    UnsupportedTokenizer(String),
    #[error("failed to load tokenizer: {0}")]
    Load(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("failed to load tokenizer: tiktoken rank file: {0}")]
    Ranks(String),
    #[error("failed to load tokenizer: Unicode character classes are unavailable")]
    UnicodeClasses,
    #[error("unsupported by the rust token counter: request body could not be parsed: {0}")]
    RequestParse(#[source] serde_json::Error),
    #[error("unsupported by the rust token counter: request has no countable input")]
    MissingInput,
    #[error(
        "unsupported by the rust token counter: float text values are counted by the python path"
    )]
    FloatText,
    #[error(
        "unsupported by the rust token counter: content block type is counted by the python path"
    )]
    ContentBlock,
    #[error("unsupported by the rust token counter: array parameter without items")]
    ArrayItems,
    #[error("unsupported by the rust token counter: text value could not be serialized: {0}")]
    JsonSerialization(#[source] serde_json::Error),
    #[error("unsupported by the rust token counter: serialized text value is not UTF-8: {0}")]
    JsonUtf8(#[source] FromUtf8Error),
    #[error("tokenization failed: {0}")]
    Encode(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("token decoding failed: {0}")]
    Decode(String),
    #[error("token counting task failed: {0}")]
    Task(String),
}
