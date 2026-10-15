use std::sync::Arc;

use litellm_core_utils::settings::Lookup;
use litellm_http::{HttpClientConfig, HttpClientPool};

use crate::{Error, KeyManagementSettings, KeyManagementSystem, SecretManager};

pub async fn load_native_manager(
    _pool: &HttpClientPool,
    _config: &HttpClientConfig,
    system: KeyManagementSystem,
    settings: KeyManagementSettings,
    environment: Arc<dyn Lookup + Send + Sync>,
    enterprise_enabled: bool,
) -> Result<SecretManager, Error> {
    match (system, settings, environment, enterprise_enabled) {
        #[cfg(feature = "aws")]
        (KeyManagementSystem::AwsSecretManager, settings, environment, _) => {
            crate::aws::AwsSecretsManagerV2::load_aws_secret_manager(
                Some(true),
                settings,
                environment,
            )?
            .map(SecretManager::AwsSecretsManagerV2)
            .ok_or(Error::NativeBackendUnavailable)
        }
        #[cfg(feature = "aws")]
        (KeyManagementSystem::AwsKms, settings, environment, _) => {
            crate::aws::load_aws_kms(Some(true), &settings, environment)?
                .map(SecretManager::AwsKms)
                .ok_or(Error::NativeBackendUnavailable)
        }
        #[cfg(feature = "azure")]
        (KeyManagementSystem::AzureKeyVault, _, environment, _) => Ok(
            SecretManager::AzureKeyVault(crate::azure::AzureKeyVault::new(
                _pool.client(_config, litellm_http::ClientVariant::Provider)?,
                environment,
            )?),
        ),
        #[cfg(feature = "google")]
        (KeyManagementSystem::GoogleSecretManager, _, environment, enterprise_enabled) => Ok(
            SecretManager::GoogleSecretManager(crate::google::GoogleSecretManager::new(
                _pool.client(_config, litellm_http::ClientVariant::Provider)?,
                environment,
                enterprise_enabled,
            )?),
        ),
        #[cfg(feature = "google")]
        (KeyManagementSystem::GoogleKms, _, environment, _) => {
            crate::google::load_google_kms(Some(true), environment)
                .await?
                .map(SecretManager::GoogleKms)
                .ok_or(Error::NativeBackendUnavailable)
        }
        #[cfg(feature = "hashicorp")]
        (KeyManagementSystem::HashicorpVault, _, environment, enterprise_enabled) => {
            Ok(SecretManager::HashicorpVault(
                crate::hashicorp::HashicorpVault::new(environment, enterprise_enabled)?,
            ))
        }
        #[cfg(feature = "cyberark")]
        (KeyManagementSystem::Cyberark, _, environment, enterprise_enabled) => Ok(
            SecretManager::Cyberark(crate::cyberark::CyberArkSecretManager::new(
                _pool,
                _config,
                environment,
                enterprise_enabled,
            )?),
        ),
        _ => Err(Error::NativeBackendUnavailable),
    }
}
