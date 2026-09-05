use std::sync::Arc;

use litellm_core::auth::AuthRuntime;
use pyo3::prelude::*;

use crate::errors::{RustBridgeDeclined, core_error_to_pyerr};

#[pyclass(frozen)]
struct PythonAuthRuntime(Arc<AuthRuntime>);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "__auth_runtime__",
        PythonAuthRuntime(litellm_config::auth::runtime().map_err(core_error_to_pyerr)?),
    )
}

pub fn runtime(module: &Bound<'_, PyModule>) -> PyResult<Arc<AuthRuntime>> {
    Ok(module
        .getattr("__auth_runtime__")?
        .extract::<PyRef<'_, PythonAuthRuntime>>()?
        .0
        .clone())
}

pub(crate) fn preflight(auth_provider: Option<Py<PyAny>>) -> PyResult<()> {
    if auth_provider.is_some() {
        return Err(RustBridgeDeclined::new_err(
            "Python token-provider authentication is not implemented",
        ));
    }
    Ok(())
}
