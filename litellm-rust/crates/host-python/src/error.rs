use litellm_error::Report;
use pyo3::{
    PyErr,
    exceptions::{PyRuntimeError, PyTypeError},
};

/// A failure host-python raises itself, surfacing in Python as the exception `PyErr::from` picks.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("missing native call state")]
    MissingState,
    #[error("execution is already running")]
    ExecutionRunning,
    #[error("execution is closed")]
    ExecutionClosed,
    #[error("can't send non-None value to a just-started execution")]
    SendBeforeStart,
    #[error("execution requires start before resume and can only start once")]
    ExecutionOutOfOrder,
    #[error("execution was closed while running")]
    ExecutionClosedWhileRunning,
    #[error("native execution closed")]
    NativeExecutionClosed,
    #[error("synchronous native routes cannot run from a Tokio context; use the async route")]
    SyncInsideTokio,
    #[error("the context body produced no value")]
    ContextBodyProducedNothing,
    #[error("the context body ran more than once")]
    ContextBodyRanTwice,
    #[error("the blocking pool dropped a Python call")]
    BlockingCallCancelled(#[source] tokio::task::JoinError),
    #[error("file read must return bytes or str, got {got}")]
    FileContentType { got: String },
}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        let message = Report::new(&error).to_string();
        match error {
            Error::SendBeforeStart | Error::FileContentType { .. } => PyTypeError::new_err(message),
            Error::MissingState
            | Error::ExecutionRunning
            | Error::ExecutionClosed
            | Error::ExecutionOutOfOrder
            | Error::ExecutionClosedWhileRunning
            | Error::NativeExecutionClosed
            | Error::SyncInsideTokio
            | Error::ContextBodyProducedNothing
            | Error::ContextBodyRanTwice
            | Error::BlockingCallCancelled(_) => PyRuntimeError::new_err(message),
        }
    }
}

/// Why a custom operation the host answered did not produce a result: the route's own code
/// rejected it, which the route classifies like any other native failure, or Python code
/// raised, which reaches the caller as it was raised.
#[derive(Debug, thiserror::Error)]
pub enum InvokeError<E> {
    #[error(transparent)]
    Native(E),
    #[error(transparent)]
    Python(#[from] PyErr),
}
