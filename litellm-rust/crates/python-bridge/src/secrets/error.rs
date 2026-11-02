use std::fmt;

use litellm_error::Report;
use litellm_secrets::Error;
use pyo3::{
    exceptions::{PyBaseException, PyRuntimeError, PyValueError},
    prelude::*,
};

/// A secret manager call the bridge rejects, with the text Python's own managers raise.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ManagerError {
    #[error("native secret manager must be recreated after fork")]
    Forked,
    #[error("unknown secret manager system")]
    UnknownSystem(#[source] serde_json::Error),
    #[error("invalid secret manager settings")]
    Settings(#[source] litellm_error::Source),
    #[error("native secret manager system does not match configuration")]
    SystemMismatch,
    #[error("secret manager does not support provider reads")]
    ProviderReads,
    #[error("invalid timeout")]
    Timeout(#[source] std::time::TryFromFloatSecsError),
    #[error("No secret found in CyberArk Secret Manager for {0}")]
    CyberarkSecretMissing(String),
    #[error("Current secret {0} not found")]
    CurrentMissing(String),
    #[error("Failed to verify new secret {0}")]
    ReplacementMissing(String),
    #[error("New secret value mismatch")]
    ReplacementMismatch,
    #[error("native secret manager mutation is unavailable")]
    MutationUnsupported,
    #[error(transparent)]
    Secret(Error),
}

impl ManagerError {
    pub(crate) fn settings(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Settings(litellm_error::Source::new(error))
    }
}

impl From<ManagerError> for PyErr {
    fn from(error: ManagerError) -> Self {
        let message = Report::new(&error).to_string();
        match error {
            ManagerError::Forked => PyRuntimeError::new_err(message),
            ManagerError::UnknownSystem(_)
            | ManagerError::Settings(_)
            | ManagerError::SystemMismatch
            | ManagerError::ProviderReads
            | ManagerError::Timeout(_)
            | ManagerError::CyberarkSecretMissing(_)
            | ManagerError::CurrentMissing(_)
            | ManagerError::ReplacementMissing(_)
            | ManagerError::ReplacementMismatch
            | ManagerError::MutationUnsupported
            | ManagerError::Secret(_) => PyValueError::new_err(message),
        }
    }
}

#[derive(thiserror::Error)]
#[error("Python secret manager failed")]
struct PythonSecretError(Py<PyBaseException>);

impl fmt::Debug for PythonSecretError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PythonSecretError")
    }
}

pub(super) fn external_error(py: Python<'_>, error: PyErr) -> Error {
    Error::ExternalManager(litellm_error::Source::new(PythonSecretError(
        error.into_value(py),
    )))
}

pub(crate) fn read_error(py: Python<'_>, error: PyErr) -> Error {
    Error::ExternalRead(litellm_error::Source::new(PythonSecretError(
        error.into_value(py),
    )))
}

/// The exception a Python secret manager raised anywhere in the chain, so a route surfaces it
/// unchanged instead of wrapping it.
pub(crate) fn python_error(
    py: Python<'_>,
    error: &(dyn std::error::Error + 'static),
) -> Option<PyErr> {
    litellm_error::find::<PythonSecretError>(error)
        .map(|error| PyErr::from_value(error.0.clone_ref(py).into_bound(py).into_any()))
}
