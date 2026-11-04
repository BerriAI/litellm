use std::{collections::BTreeMap, sync::Arc};

use litellm_core_utils::settings::{Lookup, ProcessEnvironment};
use litellm_host_python::{from_py, json_object_field, to_py};
use litellm_secrets::{
    KeyManagementSettings, KeyManagementSystem, Secret, SecretManager, load_native_manager,
    read_secret_from_python_manager,
};
use litellm_secrets_types::PythonSecretRead;
use pyo3::{prelude::*, types::PyDict};

use super::ManagerError;
use crate::execution::{run_async_value, run_sync_value};

struct Configuration {
    system: KeyManagementSystem,
    settings: KeyManagementSettings,
    environment: BTreeMap<String, String>,
    enterprise_enabled: bool,
}

#[pyclass(frozen, name = "_SecretManagerRuntime")]
pub(crate) struct NativeSecretManager {
    backend: SecretManager,
    configuration: Configuration,
    pid: u32,
}

impl NativeSecretManager {
    pub(super) fn backend(&self) -> PyResult<SecretManager> {
        if self.pid != std::process::id() {
            return Err(ManagerError::Forked.into());
        }
        Ok(self.backend.clone())
    }

    fn build(py: Python<'_>, configuration: Configuration) -> PyResult<Self> {
        let values = configuration.environment.clone();
        let environment: Arc<dyn Lookup + Send + Sync> =
            Arc::new(move |name: &str| values.get(name).cloned());
        let system = configuration.system;
        let settings = configuration.settings.clone();
        let enterprise_enabled = configuration.enterprise_enabled;
        let http_config = crate::http::call_config(py, &PyDict::new(py), false)?;
        let backend = run_sync_value(py, async move {
            load_native_manager(
                crate::http::pool(),
                &http_config,
                system,
                settings,
                environment,
                enterprise_enabled,
            )
            .await
            .map_err(|error| ManagerError::Secret(error).into())
        })?;
        Ok(Self {
            backend,
            configuration,
            pid: std::process::id(),
        })
    }
}

#[pymethods]
impl NativeSecretManager {
    #[staticmethod]
    #[pyo3(signature = (system, environment, settings=None, enterprise_enabled=false))]
    fn from_config(
        py: Python<'_>,
        system: &str,
        environment: BTreeMap<String, String>,
        settings: Option<&Bound<'_, PyAny>>,
        enterprise_enabled: bool,
    ) -> PyResult<Self> {
        let system = serde_json::from_value(serde_json::Value::String(system.to_owned()))
            .map_err(ManagerError::UnknownSystem)?;
        let settings = parse_settings(settings)?;
        Self::build(
            py,
            Configuration {
                system,
                settings,
                environment,
                enterprise_enabled,
            },
        )
    }

    #[getter]
    fn system(&self) -> &'static str {
        self.configuration.system.into()
    }

    #[pyo3(signature = (name, settings=None))]
    fn read_secret(
        &self,
        py: Python<'_>,
        name: String,
        settings: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let backend = self.backend()?;
        let settings = settings
            .map(|value| parse_settings(Some(value)))
            .transpose()?
            .unwrap_or_else(|| self.configuration.settings.clone());
        run_sync_value(py, async move {
            read_secret_from_python_manager(&backend, &name, &settings, &ProcessEnvironment)
                .await
                .map_err(|error| python_read_error(backend.system(), &name, error))
                .and_then(|value| python_secret_value(value, &name))
        })
    }
    #[pyo3(signature = (secret_name, optional_params=None, timeout=None, primary_secret_name=None))]
    fn sync_read_secret(
        &self,
        py: Python<'_>,
        secret_name: String,
        optional_params: Option<&Bound<'_, PyAny>>,
        timeout: Option<&Bound<'_, PyAny>>,
        primary_secret_name: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let backend = self.backend()?;
        let request = super::provider::read_request(
            self.configuration.system,
            secret_name,
            optional_params,
            timeout,
            primary_secret_name,
            true,
        )?;
        run_sync_value(py, async move {
            super::operations::read_python_provider(&backend, &request, &ProcessEnvironment)
                .await
                .map_err(|error| ManagerError::Secret(error).into())
                .and_then(|value| python_secret_value(value, &request.secret_name))
        })
    }

    #[pyo3(signature = (secret_name, optional_params=None, timeout=None, primary_secret_name=None))]
    fn async_read_secret<'py>(
        &self,
        py: Python<'py>,
        secret_name: String,
        optional_params: Option<&Bound<'py, PyAny>>,
        timeout: Option<&Bound<'py, PyAny>>,
        primary_secret_name: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.backend()?;
        let request = super::provider::read_request(
            self.configuration.system,
            secret_name,
            optional_params,
            timeout,
            primary_secret_name,
            false,
        )?;
        run_async_value(py, async move {
            super::operations::read_python_provider(&backend, &request, &ProcessEnvironment)
                .await
                .map_err(|error| ManagerError::Secret(error).into())
                .and_then(|value| python_secret_value(value, &request.secret_name))
        })
    }

    #[pyo3(signature = (secret_name, secret_value, description=None, optional_params=None, timeout=None, tags=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "preserves the Python secret-manager write signature"
    )]
    fn async_write_secret<'py>(
        &self,
        py: Python<'py>,
        secret_name: String,
        secret_value: String,
        description: Option<&Bound<'py, PyAny>>,
        optional_params: Option<&Bound<'py, PyAny>>,
        timeout: Option<&Bound<'py, PyAny>>,
        tags: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.backend()?;
        let _ = tags;
        let context = litellm_secrets_types::SecretWriteContext {
            operation: super::provider::mutation_context(
                self.configuration.system,
                optional_params,
                timeout,
            )?,
            description: if self.configuration.system == KeyManagementSystem::HashicorpVault {
                description
                    .filter(|value| !value.is_none())
                    .map(|value| {
                        if value.is_truthy()? {
                            value.extract().map(Some)
                        } else {
                            Ok(None)
                        }
                    })
                    .transpose()?
                    .flatten()
            } else {
                None
            },
            ..litellm_secrets_types::SecretWriteContext::default()
        };
        let error_context =
            super::vault::ErrorContext::capture(py, self.configuration.system, timeout)?;
        run_async_value(py, async move {
            super::mutation::mutation_value(
                super::operations::write_python_provider_with_context(
                    &backend,
                    &secret_name,
                    &litellm_secret_value::SecretValue::new(secret_value),
                    &context,
                )
                .await,
                &error_context,
            )
        })
    }

    #[pyo3(signature = (secret_name, recovery_window_in_days=None, optional_params=None, timeout=None))]
    fn async_delete_secret<'py>(
        &self,
        py: Python<'py>,
        secret_name: String,
        recovery_window_in_days: Option<&Bound<'py, PyAny>>,
        optional_params: Option<&Bound<'py, PyAny>>,
        timeout: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.backend()?;
        let _ = recovery_window_in_days;
        let context =
            super::provider::mutation_context(self.configuration.system, optional_params, timeout)?;
        let error_context =
            super::vault::ErrorContext::capture(py, self.configuration.system, timeout)?;
        run_async_value(py, async move {
            super::mutation::mutation_value(
                super::operations::delete_python_provider_with_context(
                    &backend,
                    &secret_name,
                    &context,
                )
                .await,
                &error_context,
            )
        })
    }

    #[pyo3(signature = (current_secret_name, new_secret_name, new_secret_value, optional_params=None, timeout=None))]
    fn async_rotate_secret<'py>(
        &self,
        py: Python<'py>,
        current_secret_name: String,
        new_secret_name: String,
        new_secret_value: String,
        optional_params: Option<&Bound<'py, PyAny>>,
        timeout: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.backend()?;
        let context =
            super::provider::mutation_context(self.configuration.system, optional_params, timeout)?;
        let error_context =
            super::vault::ErrorContext::capture(py, self.configuration.system, timeout)?;
        run_async_value(py, async move {
            super::mutation::mutation_value(
                super::operations::rotate_python_provider_with_context(
                    &backend,
                    &current_secret_name,
                    &new_secret_name,
                    &litellm_secret_value::SecretValue::new(new_secret_value),
                    &context,
                )
                .await,
                &error_context,
            )
        })
    }

    #[pyo3(signature = (name, settings=None))]
    fn read_secret_async<'py>(
        &self,
        py: Python<'py>,
        name: String,
        settings: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.backend()?;
        let settings = settings
            .map(|value| parse_settings(Some(value)))
            .transpose()?
            .unwrap_or_else(|| self.configuration.settings.clone());
        run_async_value(py, async move {
            read_secret_from_python_manager(&backend, &name, &settings, &ProcessEnvironment)
                .await
                .map_err(|error| python_read_error(backend.system(), &name, error))
                .and_then(|value| python_secret_value(value, &name))
        })
    }
}

fn parse_settings(value: Option<&Bound<'_, PyAny>>) -> PyResult<KeyManagementSettings> {
    value
        .map(|value| {
            Ok(serde_json::from_value(from_py::<serde_json::Value>(value)?)
                .map_err(ManagerError::settings)?)
        })
        .transpose()
        .map(Option::unwrap_or_default)
}

fn python_secret_value(payload: PythonSecretRead, name: &str) -> PyResult<Py<PyAny>> {
    let value = match payload {
        PythonSecretRead::Value(value) => value,
        PythonSecretRead::PrimaryJson(document) => {
            return Python::attach(|py| json_object_field(py, document.expose(), name));
        }
    };
    let value = match value {
        None => serde_json::Value::Null,
        Some(Secret::String(value)) => serde_json::Value::String(value.expose().to_owned()),
        Some(Secret::Bool(value)) => serde_json::Value::Bool(value),
        Some(Secret::Json(value)) => value,
    };
    Python::attach(|py| to_py(py, &value))
}

fn python_read_error(
    system: KeyManagementSystem,
    name: &str,
    error: litellm_secrets::Error,
) -> PyErr {
    match (system, error) {
        (KeyManagementSystem::Cyberark, litellm_secrets::Error::ManagedSecretMissing) => {
            ManagerError::CyberarkSecretMissing(name.to_owned()).into()
        }
        (_, error) => ManagerError::Secret(error).into(),
    }
}
