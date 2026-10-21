use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("trace snapshot serialization failed")]
    Serialization(#[from] serde_json::Error),
    #[error("trace snapshot exceeds the size limit")]
    ReadTooLarge,
}

/// Cheap to clone so one failed single-flight read can be returned to every waiting caller.
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
    Encode(#[source] Arc<serde_json::Error>),
    #[error(transparent)]
    Store(Arc<E>),
}

impl<E> Clone for ReadError<E> {
    fn clone(&self) -> Self {
        match self {
            Self::InvalidParameters => Self::InvalidParameters,
            Self::InvalidCursor(kind) => Self::InvalidCursor(kind),
            Self::AmbiguousTrace => Self::AmbiguousTrace,
            Self::TraceChanged => Self::TraceChanged,
            Self::TooLarge => Self::TooLarge,
            Self::Encode(error) => Self::Encode(Arc::clone(error)),
            Self::Store(error) => Self::Store(Arc::clone(error)),
        }
    }
}

impl<E> From<Error> for ReadError<E> {
    fn from(error: Error) -> Self {
        match error {
            Error::ReadTooLarge => Self::TooLarge,
            Error::Serialization(error) => Self::Encode(Arc::new(error)),
        }
    }
}
