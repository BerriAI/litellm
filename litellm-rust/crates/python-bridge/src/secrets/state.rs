use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use litellm_core_utils::settings::ProcessEnvironment;
use litellm_secrets::{SecretManager, SecretManagerState};
use pyo3::prelude::*;

use super::config::SecretManagerSnapshot;
use crate::coercion::ProjectionError;

static STATES: LazyLock<Mutex<HashMap<SecretManagerSnapshot, Arc<SecretManagerState>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn secret_manager_state(
    py: Python<'_>,
    snapshot: SecretManagerSnapshot,
) -> Result<Arc<SecretManagerState>, ProjectionError> {
    let Some(system) = snapshot.system else {
        return Ok(Arc::new(SecretManagerState::default()));
    };
    if let Some(state) = STATES
        .lock()
        .expect("secret manager state cache is not poisoned")
        .get(&snapshot)
        .cloned()
    {
        return Ok(state);
    }
    let environment = Arc::new(ProcessEnvironment);
    let backend = match system {
        litellm_secrets::KeyManagementSystem::Local => SecretManager::Local,
        litellm_secrets::KeyManagementSystem::AwsSecretManager => {
            SecretManager::AwsSecretsManagerV2(
                litellm_secrets_aws::AwsSecretsManagerV2::load_aws_secret_manager(
                    Some(true),
                    snapshot.settings.clone(),
                    environment.clone(),
                )
                .map_err(|error| invalid_configuration(system, error))?
                .ok_or_else(|| invalid_configuration(system, "loader returned no backend"))?,
            )
        }
        litellm_secrets::KeyManagementSystem::AwsKms => SecretManager::AwsKms(
            litellm_secrets_aws::load_aws_kms(Some(true), &snapshot.settings, environment.clone())
                .map_err(|error| invalid_configuration(system, error))?
                .ok_or_else(|| invalid_configuration(system, "loader returned no backend"))?,
        ),
        litellm_secrets::KeyManagementSystem::GoogleSecretManager => {
            SecretManager::GoogleSecretManager(
                litellm_secrets_google::GoogleSecretManager::new(environment.clone(), true)
                    .map_err(|error| invalid_configuration(system, error))?,
            )
        }
        litellm_secrets::KeyManagementSystem::GoogleKms => SecretManager::GoogleKms(
            litellm_host_python::run_sync_value(py, async move {
                Ok(
                    litellm_secrets_google::load_google_kms(Some(true), environment.clone())
                        .await
                        .map_err(|error| error.to_string()),
                )
            })
            .map_err(ProjectionError::Python)?
            .map_err(|error| invalid_configuration(system, error))?
            .ok_or_else(|| invalid_configuration(system, "loader returned no backend"))?,
        ),
        litellm_secrets::KeyManagementSystem::AzureKeyVault
        | litellm_secrets::KeyManagementSystem::HashicorpVault
        | litellm_secrets::KeyManagementSystem::Cyberark
        | litellm_secrets::KeyManagementSystem::Custom => {
            return Err(ProjectionError::InvalidConfiguration(format!(
                "secret manager system `{}` is not supported by the Rust bridge yet",
                system_name(system)
            )));
        }
    };
    let state = Arc::new(SecretManagerState::new(backend, snapshot.settings.clone()));
    STATES
        .lock()
        .expect("secret manager state cache is not poisoned")
        .insert(snapshot, state.clone());
    Ok(state)
}

fn invalid_configuration(
    system: litellm_secrets::KeyManagementSystem,
    error: impl std::fmt::Display,
) -> ProjectionError {
    ProjectionError::InvalidConfiguration(format!(
        "{system:?} secret manager could not be constructed: {error}"
    ))
}

fn system_name(system: litellm_secrets::KeyManagementSystem) -> String {
    serde_json::to_value(system)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| format!("{system:?}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_secrets::{AccessMode, KeyManagementSettings, KeyManagementSystem};
    use pyo3::prelude::*;

    use super::{SecretManagerSnapshot, secret_manager_state};

    #[rstest::rstest]
    #[case("azure_key_vault")]
    #[case("hashicorp_vault")]
    #[case("cyberark")]
    #[case("custom")]
    fn unsupported_systems_raise_a_typed_configuration_error(#[case] system: &str) {
        Python::initialize();
        Python::attach(|py| {
            let system = serde_json::from_value(serde_json::Value::String(system.into())).unwrap();
            let snapshot = SecretManagerSnapshot {
                system: Some(system),
                settings: KeyManagementSettings::default(),
            };
            let error: PyErr = match secret_manager_state(py, snapshot) {
                Ok(_) => panic!("unsupported system unexpectedly constructed"),
                Err(error) => error.into(),
            };
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
            assert!(!error.is_instance_of::<crate::errors::RustBridgeDeclined>(py));
        });
    }

    #[test]
    fn same_snapshot_reuses_the_cached_state() {
        Python::initialize();
        Python::attach(|py| {
            let snapshot = SecretManagerSnapshot {
                system: Some(KeyManagementSystem::Local),
                settings: KeyManagementSettings {
                    access_mode: AccessMode::ReadOnly,
                    ..Default::default()
                },
            };
            let first = secret_manager_state(py, snapshot.clone()).unwrap();
            let second = secret_manager_state(py, snapshot).unwrap();
            assert!(Arc::ptr_eq(&first, &second));

            let different = SecretManagerSnapshot {
                system: Some(KeyManagementSystem::Local),
                settings: KeyManagementSettings {
                    hosted_keys: Some(vec!["different".into()]),
                    ..Default::default()
                },
            };
            let third = secret_manager_state(py, different).unwrap();
            assert!(!Arc::ptr_eq(&first, &third));
        });
    }
}
