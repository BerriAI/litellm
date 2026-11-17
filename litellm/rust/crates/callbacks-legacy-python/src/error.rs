use litellm_error::Report;
use pyo3::{PyErr, exceptions::PyRuntimeError};

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("call logging is not initialized")]
    LoggingNotInitialized,
    #[error("the Messages input could not be serialized for logging")]
    LoggedInput(#[source] serde_json::Error),
}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        let message = Report::new(&error).to_string();
        match error {
            Error::LoggingNotInitialized | Error::LoggedInput(_) => {
                PyRuntimeError::new_err(message)
            }
        }
    }
}
