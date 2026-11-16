pub(crate) mod callback;
pub(crate) mod config;
mod error;
mod local;
mod mutation;
mod operations;
mod provider;
mod python;
pub(crate) mod resolved;
pub(crate) mod runtime;
pub(crate) mod seeded;
mod vault;
mod web_identity;

use std::sync::Arc;

#[cfg(test)]
pub(crate) use error::read_error;
pub(crate) use error::{ManagerError, python_error};
use litellm_secrets::source::{EnvironmentSecrets, SecretSource};
use local::LocalSecrets;
use pyo3::prelude::*;
use python::PythonSecrets;
use resolved::ResolvedSecrets;
pub(crate) use web_identity::HostWebIdentityTokens;

use crate::{
    coercion::FieldSpec,
    python_settings::{PythonSettings, Snapshot},
};

const SECRET_MANAGER_MODULE: &str = "litellm.rust_bridge.host.secret_manager";
const READABLE: FieldSpec<bool> = FieldSpec::new("readable", |field| field.schema_bool());
const NATIVE: FieldSpec<bool> = FieldSpec::new("native", |field| field.schema_bool());

#[derive(Debug, PartialEq, Eq)]
enum Selection {
    Local,
    Native,
    Python,
}

fn select(snapshot: &Snapshot<'_>) -> PyResult<Selection> {
    Ok(if !snapshot.read(&READABLE)? {
        Selection::Local
    } else if snapshot.read(&NATIVE)? {
        Selection::Native
    } else {
        Selection::Python
    })
}

/// Where a Rust route reads provider secrets from, decided per call from the settings projection.
/// With no readable secret manager Python's `get_secret_str` only reads the environment, so the
/// route reads it in Rust. A readable manager is the native backend when the configured client
/// captures one, otherwise every read goes through Python's `get_secret_str`. A bare extension
/// module without the litellm package reads the process environment.
pub(crate) fn source(py: Python<'_>) -> PyResult<Arc<dyn SecretSource>> {
    let Some(snapshot) = PythonSettings::SecretManager.read_or_unset(py)? else {
        return Ok(Arc::new(EnvironmentSecrets::python_compatible(
            crate::http::host_client(py, litellm_http::ClientVariant::Provider)?,
        )));
    };
    match select(&snapshot)? {
        Selection::Local => Ok(Arc::new(LocalSecrets::new(Arc::new(PythonSecrets::new(
            py,
        )?)))),
        Selection::Native => {
            let context = litellm_host_python::PythonContext::capture(py)?;
            let client = crate::http::host_client(py, litellm_http::ClientVariant::Provider)?;
            Ok(Arc::new(ResolvedSecrets::new(
                config::read(py)?,
                context,
                client,
            )))
        }
        Selection::Python => Ok(Arc::new(PythonSecrets::new(py)?)),
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{prelude::*, types::PyDict};
    use rstest::rstest;

    use super::{Selection, select};
    use crate::python_settings::PythonSettings;

    #[rstest]
    #[case::no_manager(false, false, Selection::Local)]
    #[case::unreadable_native(false, true, Selection::Local)]
    #[case::python_manager(true, false, Selection::Python)]
    #[case::native_manager(true, true, Selection::Native)]
    fn the_projection_selects_where_secrets_are_read(
        #[case] readable: bool,
        #[case] native: bool,
        #[case] expected: Selection,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("readable", readable).unwrap();
            locals.set_item("native", native).unwrap();
            let value = py
                .eval(
                    c"__import__('types').SimpleNamespace(readable=readable, native=native)",
                    None,
                    Some(&locals),
                )
                .unwrap();
            let selected = select(&PythonSettings::SecretManager.snapshot(value)).unwrap();
            assert_eq!(selected, expected);
        });
    }
}
