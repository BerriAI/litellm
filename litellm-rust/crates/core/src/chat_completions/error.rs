use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChatRequestError {
    #[error("chat completions requires at least one message")]
    EmptyMessages,
    #[error("invalid chat completions messages: {0}")]
    InvalidMessages(String),
    #[error("failed to serialize chat completions request: {0}")]
    Serialization(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChatResponseError {
    #[error("{api} response is not an object")]
    NotObject { api: &'static str },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("non-text response content block")]
    NonTextContent,
    #[error("invalid chat completions response JSON: {0}")]
    InvalidJson(String),
}
