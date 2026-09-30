use litellm_core_utils::settings::Lookup;
use litellm_python_compat::{Value, literal::literal_eval};
use litellm_secrets_types::PythonSecretRead;

use crate::{
    Error, KeyManagementSettings, KeyManagementSystem, Secret, SecretManager, SecretValue,
    get_secret_from_manager,
};

pub async fn get_secret_from_python_manager(
    manager: &SecretManager,
    name: &str,
    settings: &KeyManagementSettings,
    environment: &(dyn Lookup + Send + Sync),
) -> Result<Option<Secret>, Error> {
    #[cfg(feature = "aws")]
    if let SecretManager::AwsSecretsManagerV2(client) = manager {
        return client
            .read_secret_for_python(name, settings.primary_secret_name.as_deref(), environment)
            .await
            .map_err(Error::from);
    }
    let result = match manager {
        #[cfg(feature = "cyberark")]
        SecretManager::Cyberark(client) => Ok(client
            .read_with_retry(
                name,
                &Default::default(),
                litellm_secrets_cyberark::AuthenticationRetry::Never,
            )
            .await
            .unwrap_or(None)
            .map(Secret::String)),
        #[cfg(feature = "google")]
        SecretManager::GoogleSecretManager(client) => client
            .get_secret_for_python(name)
            .await
            .map_err(Error::from),
        _ => get_secret_from_manager(manager, name, settings, environment).await,
    };
    match result {
        #[cfg(feature = "google")]
        Err(Error::Google(litellm_secrets_google::Error::Status(404))) => {
            Err(Error::ManagedSecretMissing)
        }
        #[cfg(feature = "azure")]
        Err(Error::Azure(litellm_secrets_azure::Error::MissingValue)) => Ok(None),
        Ok(None)
            if matches!(manager, SecretManager::External(_))
                && manager.system() == KeyManagementSystem::AzureKeyVault =>
        {
            Ok(None)
        }
        Ok(None) => Err(Error::ManagedSecretMissing),
        result => result,
    }
}

pub(crate) fn python_manager_string(value: SecretValue) -> Secret {
    match literal_eval(value.expose()) {
        Ok(Value::Bool(boolean)) => Secret::Bool(boolean),
        _ => Secret::String(value),
    }
}

pub async fn read_secret_from_python_manager(
    manager: &SecretManager,
    name: &str,
    settings: &KeyManagementSettings,
    environment: &(dyn Lookup + Send + Sync),
) -> Result<PythonSecretRead, Error> {
    #[cfg(feature = "aws")]
    if let SecretManager::AwsSecretsManagerV2(client) = manager {
        return client
            .read_payload_for_python(name, settings.primary_secret_name.as_deref(), environment)
            .await
            .map_err(Error::from);
    }
    get_secret_from_python_manager(manager, name, settings, environment)
        .await
        .map(PythonSecretRead::Value)
}
