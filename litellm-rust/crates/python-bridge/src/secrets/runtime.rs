use std::{collections::BTreeMap, sync::Arc};

use litellm_core_utils::settings::{Lookup, ProcessEnvironment};
use litellm_host_python::{from_py, json_object_field, run_async_value, run_sync_value, to_py};
use litellm_secrets::{
    KeyManagementSettings, KeyManagementSystem, Secret, SecretManager, load_native_manager,
    read_secret_from_python_manager,
};
use litellm_secrets_types::PythonSecretRead;
use pyo3::{
    exceptions::{PyAttributeError, PyRuntimeError, PyValueError},
    prelude::*,
};

#[derive(Clone, PartialEq)]
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
            return Err(PyRuntimeError::new_err(
                "native secret manager must be recreated after fork",
            ));
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
        let backend = run_sync_value(py, async move {
            load_native_manager(system, settings, environment, enterprise_enabled)
                .await
                .map_err(|error| PyValueError::new_err(error.to_string()))
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
            .map_err(|_| PyValueError::new_err("unknown secret manager system"))?;
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

    #[staticmethod]
    pub(super) fn from_client(client: &Bound<'_, PyAny>) -> PyResult<Option<Py<Self>>> {
        let py = client.py();
        if let Ok(native) = client.extract::<Py<Self>>() {
            native.borrow(py).backend()?;
            return Ok(Some(native));
        }
        let config = py
            .import("litellm.rust_bridge.secret_manager")?
            .getattr("native_secret_manager_config")?
            .call1((client,))?;
        if config.is_none() {
            return Ok(None);
        }
        if !config.getattr("owner_type")?.is(client.get_type()) {
            return Ok(None);
        }
        let methods = config
            .getattr("methods")?
            .extract::<Vec<(String, Py<PyAny>)>>()?;
        for (name, original) in methods {
            let current = client.getattr(name.as_str())?;
            let implementation = optional_attribute(&current, "__func__")?.unwrap_or(current);
            if !implementation.is(original.bind(py)) {
                return Ok(None);
            }
        }
        let environment_attributes: BTreeMap<String, String> = config
            .getattr("environment_attributes")?
            .extract::<Vec<(String, String)>>()?
            .into_iter()
            .collect();
        let captured = config
            .getattr("environment")?
            .extract::<Vec<(String, String)>>()?;
        let overrides = environment_attributes
            .iter()
            .map(|(key, attribute)| {
                let value = attribute_path(client, attribute)?;
                Ok((
                    key.clone(),
                    if value.is_none() {
                        None
                    } else {
                        Some(value.str()?.extract::<String>()?)
                    },
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let settings =
            from_py::<serde_json::Map<String, serde_json::Value>>(&config.getattr("settings")?)
                .map_err(|_| PyValueError::new_err("invalid secret manager settings"))?;
        let attributes = config
            .getattr("settings_attributes")?
            .extract::<Vec<String>>()?;
        let setting_overrides = attributes
            .into_iter()
            .map(|name| {
                let value = from_py::<serde_json::Value>(&client.getattr(name.as_str())?)?;
                Ok((name, value))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let configuration = Configuration {
            system: serde_json::from_value(serde_json::Value::String(
                config.getattr("system")?.extract()?,
            ))
            .map_err(|_| PyValueError::new_err("unknown secret manager system"))?,
            settings: serde_json::from_value(serde_json::Value::Object(
                settings.into_iter().chain(setting_overrides).collect(),
            ))
            .map_err(|_| PyValueError::new_err("invalid secret manager settings"))?,
            environment: captured
                .into_iter()
                .filter(|(key, _)| !environment_attributes.contains_key(key))
                .chain(
                    overrides
                        .into_iter()
                        .filter_map(|(key, value)| value.map(|value| (key, value))),
                )
                .collect(),
            enterprise_enabled: config.getattr("enterprise_enabled")?.extract()?,
        };
        if let Some(native) = cached(client, &configuration)? {
            return Ok(Some(native));
        }
        let runtime = Self::build(py, configuration)?;
        if let Some(native) = cached(client, &runtime.configuration)? {
            return Ok(Some(native));
        }
        let native = Py::new(py, runtime)?;
        client.setattr("_litellm_native_secret_manager", native.bind(py))?;
        Ok(Some(native))
    }

    #[getter]
    fn system(&self) -> String {
        serde_json::to_value(self.configuration.system)
            .expect("serializable system")
            .as_str()
            .expect("string system")
            .to_owned()
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
                .map_err(|error| PyValueError::new_err(error.to_string()))
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
                .map_err(|error| PyValueError::new_err(error.to_string()))
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
                    &litellm_secrets::SecretValue::new(secret_value),
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
                    &litellm_secrets::SecretValue::new(new_secret_value),
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

fn optional_attribute<'py>(
    object: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match object.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.is_instance_of::<PyAttributeError>(object.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

fn attribute_path<'py>(object: &Bound<'py, PyAny>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    match path.split_once('.') {
        Some((head, tail)) => attribute_path(&object.getattr(head)?, tail),
        None => object.getattr(path),
    }
}

fn cached(
    client: &Bound<'_, PyAny>,
    configuration: &Configuration,
) -> PyResult<Option<Py<NativeSecretManager>>> {
    let Some(value) = optional_attribute(client, "_litellm_native_secret_manager")? else {
        return Ok(None);
    };
    let native = value.extract::<Py<NativeSecretManager>>()?;
    let same_configuration = native.borrow(client.py()).pid == std::process::id()
        && &native.borrow(client.py()).configuration == configuration;
    Ok(same_configuration.then_some(native))
}

fn parse_settings(value: Option<&Bound<'_, PyAny>>) -> PyResult<KeyManagementSettings> {
    value
        .map(|value| {
            serde_json::from_value(from_py::<serde_json::Value>(value)?)
                .map_err(|_| PyValueError::new_err("invalid secret manager settings"))
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
    let message = match (system, error) {
        (KeyManagementSystem::Cyberark, litellm_secrets::Error::ManagedSecretMissing) => {
            format!("No secret found in CyberArk Secret Manager for {name}")
        }
        (_, error) => error.to_string(),
    };
    PyValueError::new_err(message)
}
