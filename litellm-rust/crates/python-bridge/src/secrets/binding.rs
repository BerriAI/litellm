use std::sync::Arc;

use pyo3::prelude::*;

use super::{config, service::SecretManagerService};
use crate::python_settings::PythonSettings;

pub(crate) enum SecretManagerBinding {
    Local,
    #[allow(
        dead_code,
        reason = "native Python handles are introduced in the follow-up PR"
    )]
    Native(Arc<SecretManagerService>),
    PythonCallback(Py<PyAny>),
}

pub(crate) struct ResolvedSecretManager {
    pub(crate) binding: SecretManagerBinding,
    pub(crate) snapshot: config::SecretManagerSnapshot,
    pub(crate) python_settings: Option<Py<PyAny>>,
}

pub(crate) fn resolve(py: Python<'_>) -> PyResult<ResolvedSecretManager> {
    let snapshot = config::project(&PythonSettings::SecretManager.read(py)?)?;
    let module = py.import("litellm")?;
    let client = module.getattr("secret_manager_client")?;
    let binding = if client.is_none() {
        SecretManagerBinding::Local
    } else {
        SecretManagerBinding::PythonCallback(client.unbind())
    };
    let settings = module.getattr("_key_management_settings")?;
    let python_settings = (!settings.is_none()).then(|| settings.unbind());
    Ok(ResolvedSecretManager {
        binding,
        snapshot,
        python_settings,
    })
}
