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
    #[error("sync call suspended")]
    SyncCallSuspended,
    #[error("synchronous native routes cannot run from a Tokio context; use the async route")]
    SyncInsideTokio,
    #[error("the context body produced no value")]
    ContextBodyProducedNothing,
    #[error("file read must return bytes or str, got {got}")]
    FileContentType { got: String },
}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        match error {
            Error::SendBeforeStart | Error::FileContentType { .. } => {
                PyTypeError::new_err(error.to_string())
            }
            Error::MissingState
            | Error::ExecutionRunning
            | Error::ExecutionClosed
            | Error::ExecutionOutOfOrder
            | Error::ExecutionClosedWhileRunning
            | Error::NativeExecutionClosed
            | Error::SyncCallSuspended
            | Error::SyncInsideTokio
            | Error::ContextBodyProducedNothing => PyRuntimeError::new_err(error.to_string()),
        }
    }
}

/// Why a custom operation the host answered did not produce a result: the route's own code
/// rejected it, which the route classifies like any other native failure, or Python code
/// raised, which reaches the caller as it was raised.
#[derive(Debug, thiserror::Error)]
pub enum InvokeError<E> {
    #[error("{0}")]
    Native(E),
    #[error(transparent)]
    Python(#[from] PyErr),
}
