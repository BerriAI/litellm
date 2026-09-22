use std::{future::Future, pin::Pin};

use litellm_core_utils::settings::Lookup;
use litellm_secrets::{
    Error, ExternalSecretManager, KeyManagementSettings, KeyManagementSystem, Secret, SecretValue,
};
use pyo3::{prelude::*, types::PyDict};

pub(crate) struct PythonSecretManager {
    client: Py<PyAny>,
    system: Option<KeyManagementSystem>,
    settings: Option<Py<PyAny>>,
}

impl PythonSecretManager {
    pub(crate) fn new(
        client: Py<PyAny>,
        system: Option<KeyManagementSystem>,
        settings: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            client,
            system,
            settings,
        }
    }

    fn read(&self, py: Python<'_>, name: &str) -> PyResult<Option<String>> {
        if self.system.is_none() {
            let client = self.client.bind(py);
            if client.hasattr("sync_read_secret")? {
                let kwargs = PyDict::new(py);
                kwargs.set_item("secret_name", name)?;
                return client
                    .call_method("sync_read_secret", (), Some(&kwargs))?
                    .extract();
            }
        }
        let key_manager = self.system.map_or_else(
            || Ok("local".to_owned()),
            |system| {
                serde_json::to_value(system)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("secret manager system is invalid")
                    })
            },
        )?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("client", self.client.bind(py))?;
        kwargs.set_item("key_manager", key_manager)?;
        kwargs.set_item("secret_name", name)?;
        kwargs.set_item(
            "key_management_settings",
            self.settings
                .as_ref()
                .map_or_else(|| py.None(), |settings| settings.clone_ref(py)),
        )?;
        py.import("litellm.secret_managers.secret_manager_handler")?
            .getattr("get_secret_from_manager")?
            .call((), Some(&kwargs))?
            .extract()
    }
}

impl ExternalSecretManager for PythonSecretManager {
    fn system(&self) -> KeyManagementSystem {
        self.system.unwrap_or(KeyManagementSystem::Custom)
    }

    fn read_secret<'a>(
        &'a self,
        name: &'a str,
        _settings: &'a KeyManagementSettings,
        _environment: &'a (dyn Lookup + Send + Sync),
    ) -> Pin<Box<dyn Future<Output = Result<Option<Secret>, Error>> + Send + 'a>> {
        Box::pin(async move {
            Python::attach(|py| self.read(py, name))
                .map(|value| value.map(SecretValue::new).map(Secret::String))
                .map_err(|_| Error::ExternalManager)
        })
    }
}
