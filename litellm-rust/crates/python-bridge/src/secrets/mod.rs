pub(crate) mod callback;
pub(crate) mod config;
mod error;
mod mutation;
mod operations;
mod provider;
mod python;
pub(crate) mod resolved;
pub(crate) mod runtime;
mod vault;

use std::sync::Arc;

pub(crate) use error::python_error;
use litellm_secrets::source::{EnvironmentSecrets, SecretSource};
use pyo3::prelude::*;
use python::PythonSecrets;
use resolved::ResolvedSecrets;

use crate::{coercion::FieldSpec, python_settings::PythonSettings};

const NATIVE: FieldSpec<bool> = FieldSpec::new("native", |field| field.schema_bool());

/// Where a Rust route reads provider secrets from. Python's `get_secret_str` until a
/// `SecretManagerRule` in `catalog.py` moves the configured system off `PYTHON_ONLY`, then the
/// native secret manager. A bare extension module without the litellm package reads the process
/// environment.
pub(crate) fn source(py: Python<'_>) -> PyResult<Arc<dyn SecretSource>> {
    let Some(snapshot) = PythonSettings::SecretManager.read_or_unset(py)? else {
        return Ok(Arc::new(EnvironmentSecrets::python_compatible(
            crate::http::host_client(py, litellm_http::ClientVariant::Provider)?,
        )));
    };
    if snapshot.read(&NATIVE)? {
        let context = litellm_host_python::PythonContext::capture(py)?;
        let client = crate::http::host_client(py, litellm_http::ClientVariant::Provider)?;
        return Ok(Arc::new(ResolvedSecrets::new(
            config::read(py)?,
            context,
            client,
        )));
    }
    Ok(Arc::new(PythonSecrets::new(py)?))
}
