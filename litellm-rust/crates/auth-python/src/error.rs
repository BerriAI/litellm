use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error(transparent)]
    Python(#[from] PyErr),
    #[error("{value_name} must be a string, got {actual_type}")]
    InvalidReturn {
        value_name: &'static str,
        actual_type: String,
    },
    #[error("{failure_context}: {message}")]
    Callback {
        failure_context: &'static str,
        message: String,
        #[source]
        source: PyErr,
    },
}

impl Error {
    pub(crate) fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            Self::Python(error) => error,
            Self::InvalidReturn {
                value_name,
                actual_type,
            } => PyTypeError::new_err(format!("{value_name} must be a string, got {actual_type}")),
            Self::Callback {
                failure_context,
                message,
                source,
            } => {
                let error = PyRuntimeError::new_err(format!("{failure_context}: {message}"));
                error.set_cause(py, Some(source));
                error
            }
        }
    }
}
