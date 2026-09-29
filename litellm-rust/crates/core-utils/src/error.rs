#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ApiUrlError {
    #[error("invalid URL: {0}")]
    Parse(#[from] url::ParseError),
    #[error("URL cannot be used as a base")]
    CannotBeBase,
    #[error("URL must have a host and a supported transport scheme")]
    InvalidTransport,
    #[error("invalid endpoint segment")]
    InvalidSegment,
    #[error("conflicting endpoint query parameter: {0}")]
    QueryConflict(String),
}
