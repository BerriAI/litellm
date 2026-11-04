mod error;
mod mutation;
mod operations;
mod provider;
mod python;
pub(crate) mod runtime;
mod vault;

use std::sync::Arc;

#[cfg(test)]
pub(crate) use error::external_error;
pub(crate) use error::{ManagerError, python_error};
use litellm_secrets::source::{EnvironmentSecrets, SecretSource};
use pyo3::prelude::*;
use python::PythonSecrets;

use crate::python_settings::missing_module;

/// Where a Rust route reads provider secrets from: Python's `get_secret_str`, the one resolver for
/// the configured secret manager. A bare extension module without the litellm package reads the
/// process environment.
pub(crate) fn source(py: Python<'_>) -> PyResult<Arc<dyn SecretSource>> {
    match PythonSecrets::new(py) {
        Ok(secrets) => Ok(Arc::new(secrets)),
        Err(error) if missing_module(py, &error, "litellm")? => Ok(Arc::new(
            EnvironmentSecrets::python_compatible(crate::http::host_client(
                py,
                litellm_http::ClientVariant::Provider,
            )?),
        )),
        Err(error) => Err(error),
    }
}
