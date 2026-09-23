pub(crate) mod callback;
pub(crate) mod config;
mod error;
mod mutation;
mod operations;
mod provider;
pub(crate) mod resolved;
pub(crate) mod runtime;
mod vault;

use std::sync::Arc;

use litellm_secrets::source::{EnvironmentSecrets, SecretSource};
use pyo3::prelude::*;

pub(crate) use error::python_error;
use resolved::ResolvedSecrets;

use crate::{
    coercion::FieldSpec,
    errors::RustBridgeDeclined,
    python_settings::{PythonSettings, Snapshot},
};

const READABLE: FieldSpec<bool> = FieldSpec::new("readable", |field| field.schema_bool());
const NATIVE: FieldSpec<bool> = FieldSpec::new("native", |field| field.schema_bool());

/// Where a Rust route reads provider secrets from, as `litellm.get_secret` would.
pub(crate) fn source(py: Python<'_>) -> PyResult<Arc<dyn SecretSource>> {
    select(&PythonSettings::SecretManager.read(py)?, || {
        Ok(Arc::new(ResolvedSecrets::new(config::read(py)?)))
    })
}

fn select(
    manager: &Snapshot<'_>,
    resolved: impl FnOnce() -> PyResult<Arc<dyn SecretSource>>,
) -> PyResult<Arc<dyn SecretSource>> {
    if !manager.read(&READABLE)? {
        return Ok(Arc::new(EnvironmentSecrets::python_compatible()));
    }
    if !manager.read(&NATIVE)? {
        return Err(RustBridgeDeclined::new_err(
            "the configured secret manager is not enabled for the Rust bridge",
        ));
    }
    resolved()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_secrets::source::{EnvironmentSecrets, SecretSource};
    use pyo3::{prelude::*, types::PyDict};
    use rstest::rstest;

    use super::select;
    use crate::{errors::RustBridgeDeclined, python_settings::PythonSettings};

    enum Selected {
        Environment,
        Declined,
        Resolved,
    }

    #[rstest]
    #[case::unreadable(false, false, Selected::Environment)]
    #[case::unreadable_even_if_native(false, true, Selected::Environment)]
    #[case::readable_python_only(true, false, Selected::Declined)]
    #[case::readable_native(true, true, Selected::Resolved)]
    fn readable_and_native_select_the_secret_source(
        #[case] readable: bool,
        #[case] native: bool,
        #[case] expected: Selected,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("readable", readable).unwrap();
            locals.set_item("native", native).unwrap();
            let manager = py
                .eval(
                    c"__import__('types').SimpleNamespace(readable=readable, native=native)",
                    None,
                    Some(&locals),
                )
                .unwrap();
            let mut resolved_called = false;
            let selected = select(&PythonSettings::SecretManager.snapshot(manager), || {
                resolved_called = true;
                Ok(Arc::new(EnvironmentSecrets::python_compatible()) as Arc<dyn SecretSource>)
            });
            match expected {
                Selected::Environment => assert!(selected.is_ok() && !resolved_called),
                Selected::Resolved => assert!(selected.is_ok() && resolved_called),
                Selected::Declined => {
                    let error = selected.err().expect("the Rust route declines");
                    assert!(error.is_instance_of::<RustBridgeDeclined>(py));
                    assert!(!resolved_called);
                }
            }
        });
    }
}
