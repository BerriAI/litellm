use std::fmt;

use litellm_secrets::Error;
use pyo3::{exceptions::PyBaseException, prelude::*};

#[derive(thiserror::Error)]
#[error("Python secret manager failed")]
struct PythonSecretError(Py<PyBaseException>);

impl fmt::Debug for PythonSecretError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PythonSecretError")
    }
}

pub(super) fn external_error(py: Python<'_>, error: PyErr) -> Error {
    Error::ExternalManager(Box::new(PythonSecretError(error.into_value(py))))
}

pub(super) fn read_error(py: Python<'_>, error: PyErr) -> Error {
    Error::ExternalRead(Box::new(PythonSecretError(error.into_value(py))))
}

pub(crate) fn python_error(py: Python<'_>, error: &Error) -> Option<PyErr> {
    let (Error::ExternalManager(source) | Error::ExternalRead(source)) = error else {
        return None;
    };
    source
        .downcast_ref::<PythonSecretError>()
        .map(|error| PyErr::from_value(error.0.clone_ref(py).into_bound(py).into_any()))
}
