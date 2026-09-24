use std::{future::Future, pin::Pin, sync::Arc};

use litellm_core_utils::settings::Lookup;

use crate::{Error, KeyManagementSettings, KeyManagementSystem, Secret};

#[cfg(any(feature = "aws", feature = "google"))]
use crate::SecretValue;

#[cfg(any(
    feature = "google",
    feature = "hashicorp",
    feature = "azure",
    feature = "cyberark"
))]
use litellm_secrets_types::BaseSecretManager;

pub trait ExternalSecretManager: Send + Sync {
    fn system(&self) -> KeyManagementSystem;

    fn read_secret<'a>(
        &'a self,
        name: &'a str,
        settings: &'a KeyManagementSettings,
        environment: &'a (dyn Lookup + Send + Sync),
    ) -> Pin<Box<dyn Future<Output = Result<Option<Secret>, Error>> + Send + 'a>>;
}

#[derive(Clone)]
pub enum SecretManager {
    External(Arc<dyn ExternalSecretManager>),
    #[cfg(feature = "aws")]
    AwsKms(crate::aws::AwsKms),
    #[cfg(feature = "aws")]
    AwsSecretsManagerV2(crate::aws::AwsSecretsManagerV2),
    #[cfg(feature = "google")]
    GoogleKms(crate::google::GoogleKms),
    #[cfg(feature = "google")]
    GoogleSecretManager(crate::google::GoogleSecretManager),
    #[cfg(feature = "hashicorp")]
    HashicorpVault(crate::hashicorp::HashicorpVault),
    #[cfg(feature = "azure")]
    AzureKeyVault(crate::azure::AzureKeyVault),
    #[cfg(feature = "cyberark")]
    Cyberark(crate::cyberark::CyberArkSecretManager),
}

impl SecretManager {
    pub fn system(&self) -> KeyManagementSystem {
        match self {
            Self::External(manager) => manager.system(),
            #[cfg(feature = "aws")]
            Self::AwsKms(_) => KeyManagementSystem::AwsKms,
            #[cfg(feature = "aws")]
            Self::AwsSecretsManagerV2(_) => KeyManagementSystem::AwsSecretManager,
            #[cfg(feature = "google")]
            Self::GoogleKms(_) => KeyManagementSystem::GoogleKms,
            #[cfg(feature = "google")]
            Self::GoogleSecretManager(_) => KeyManagementSystem::GoogleSecretManager,
            #[cfg(feature = "hashicorp")]
            Self::HashicorpVault(_) => KeyManagementSystem::HashicorpVault,
            #[cfg(feature = "azure")]
            Self::AzureKeyVault(_) => KeyManagementSystem::AzureKeyVault,
            #[cfg(feature = "cyberark")]
            Self::Cyberark(_) => KeyManagementSystem::Cyberark,
        }
    }
}

pub async fn get_secret_from_manager(
    client: &SecretManager,
    secret_name: &str,
    _settings: &KeyManagementSettings,
    environment: &(dyn Lookup + Send + Sync),
) -> Result<Option<Secret>, Error> {
    match client {
        SecretManager::External(manager) => {
            manager
                .read_secret(secret_name, _settings, environment)
                .await
        }
        #[cfg(feature = "aws")]
        SecretManager::AwsKms(client) => {
            let ciphertext = environment
                .get(secret_name)
                .ok_or(Error::MissingCiphertext)?;
            let plaintext = client
                .decrypt(decode_ciphertext(&ciphertext, Base64Mode::Permissive)?)
                .await?;
            let value = String::from_utf8(plaintext).map_err(|_| Error::Utf8)?;
            Ok(Some(Secret::String(SecretValue::new(value.trim()))))
        }
        #[cfg(feature = "google")]
        SecretManager::GoogleKms(client) => {
            let ciphertext = environment
                .get(secret_name)
                .ok_or(Error::MissingCiphertext)?;
            let plaintext = client
                .decrypt(decode_ciphertext(&ciphertext, Base64Mode::Canonical)?)
                .await?;
            let value = String::from_utf8(plaintext).map_err(|_| Error::Utf8)?;
            Ok(Some(Secret::String(SecretValue::new(value))))
        }
        #[cfg(feature = "aws")]
        SecretManager::AwsSecretsManagerV2(client) => client
            .read_secret_for_resolver(
                secret_name,
                _settings.primary_secret_name.as_deref(),
                environment,
            )
            .await
            .map_err(Error::from),
        #[cfg(feature = "google")]
        SecretManager::GoogleSecretManager(client) => read_manager(client, secret_name).await,
        #[cfg(feature = "hashicorp")]
        SecretManager::HashicorpVault(client) => read_manager(client, secret_name).await,
        #[cfg(feature = "azure")]
        SecretManager::AzureKeyVault(client) => read_manager(client, secret_name).await,
        #[cfg(feature = "cyberark")]
        SecretManager::Cyberark(client) => read_manager(client, secret_name).await,
    }
}

#[cfg(any(feature = "aws", feature = "google"))]
#[derive(Clone, Copy)]
enum Base64Mode {
    #[cfg(feature = "google")]
    Canonical,
    #[cfg(feature = "aws")]
    Permissive,
}

#[cfg(any(feature = "aws", feature = "google"))]
fn decode_ciphertext(value: &str, mode: Base64Mode) -> Result<Vec<u8>, Error> {
    use base64::{Engine, engine::general_purpose::STANDARD};
    let canonical = match mode {
        #[cfg(feature = "google")]
        Base64Mode::Canonical => true,
        #[cfg(feature = "aws")]
        Base64Mode::Permissive => false,
    };
    let encoded = if canonical {
        value.to_owned()
    } else {
        value
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '='))
            .collect()
    };
    let ciphertext = STANDARD
        .decode(&encoded)
        .map_err(|_| Error::InvalidCiphertext)?;
    if canonical && STANDARD.encode(&ciphertext) != encoded {
        return Err(Error::InvalidCiphertext);
    }
    Ok(ciphertext)
}

#[cfg(any(
    feature = "google",
    feature = "hashicorp",
    feature = "azure",
    feature = "cyberark"
))]
async fn read_manager<M: BaseSecretManager>(
    manager: &M,
    name: &str,
) -> Result<Option<Secret>, Error>
where
    Error: From<M::Error>,
{
    manager
        .async_read_secret(name, &M::Context::default())
        .await
        .map(|value| value.map(Secret::String))
        .map_err(Error::from)
}
