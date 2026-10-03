#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("trace snapshot serialization failed")]
    Serialization(#[from] serde_json::Error),
    #[error("trace snapshot exceeds the size limit")]
    ReadTooLarge,
}

#[derive(Debug, thiserror::Error)]
pub enum ReadError<E> {
    #[error("invalid trace read parameters")]
    InvalidParameters,
    #[error("Invalid {0} cursor")]
    InvalidCursor(&'static str),
    #[error("Multiple traces have this ID; provide trace_ref")]
    AmbiguousTrace,
    #[error("Trace changed while paging; refresh the trace to continue")]
    TraceChanged,
    #[error("Trace exceeds the interactive read budget; use a filtered trace query")]
    TooLarge,
    #[error("trace could not be encoded")]
    Encode(#[source] serde_json::Error),
    #[error(transparent)]
    Store(E),
}
