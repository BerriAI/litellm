use http::header::{HeaderName, InvalidHeaderName, InvalidHeaderValue};
use litellm_error::Report;
use pyo3::{
    PyErr,
    exceptions::{PyRuntimeError, PyValueError},
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("call logging is not initialized")]
    LoggingNotInitialized,
    #[error("a pre-call callback set the header name {name:?}, which is not a valid header name")]
    CallbackHeaderName {
        name: String,
        #[source]
        source: InvalidHeaderName,
    },
    #[error(
        "a pre-call callback set a value for the header {name} that is not a valid header value"
    )]
    CallbackHeaderValue {
        name: HeaderName,
        #[source]
        source: InvalidHeaderValue,
    },
}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        let message = Report::new(&error).to_string();
        match error {
            Error::LoggingNotInitialized => PyRuntimeError::new_err(message),
            Error::CallbackHeaderName { .. } | Error::CallbackHeaderValue { .. } => {
                PyValueError::new_err(message)
            }
        }
    }
}
