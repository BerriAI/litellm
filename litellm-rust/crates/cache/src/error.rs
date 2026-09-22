#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("cache is unavailable")]
    Unavailable,
    #[error("invalid cache entry")]
    InvalidEntry,
    #[error("flushing Redis requires an explicit namespace")]
    UnscopedFlush,
    #[error("operation is not supported by this cache")]
    UnsupportedOperation,
    #[error("semantic cache requires request messages")]
    MissingPrompt,
}
