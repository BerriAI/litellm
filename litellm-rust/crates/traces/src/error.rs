#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid OTLP trace payload")]
    InvalidPayload,
    #[error("OTLP trace payload exceeds the decoding budget")]
    TooLarge,
    #[error("OTLP token count is outside the storage range")]
    TokenCountOutOfRange,
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

#[derive(Debug, thiserror::Error)]
#[error("invalid trace query scope")]
pub struct InvalidScope;

#[derive(Debug, thiserror::Error)]
#[error("unknown ClickHouse read query")]
pub struct InvalidQuery;

#[derive(Debug, thiserror::Error)]
#[error("invalid trace call key")]
pub struct InvalidCallKey;
