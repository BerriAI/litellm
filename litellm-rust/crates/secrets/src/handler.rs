use litellm_core_utils::settings::Lookup;

use crate::{Error, KeyManagementSettings, KeyManagementSystem, Secret, SecretValue};

#[derive(Clone)]
pub enum SecretManager {
    Local,
    #[cfg(feature = "aws")]
    AwsKms(crate::aws::AwsKms),
    #[cfg(feature = "aws")]
    AwsSecretsManagerV2(crate::aws::AwsSecretsManagerV2),
    #[cfg(feature = "google")]
    GoogleKms(crate::google::GoogleKms),
    #[cfg(feature = "google")]
    GoogleSecretManager(crate::google::GoogleSecretManager),
}

impl SecretManager {
    pub fn system(&self) -> KeyManagementSystem {
        match self {
            Self::Local => KeyManagementSystem::Local,
            #[cfg(feature = "aws")]
            Self::AwsKms(_) => KeyManagementSystem::AwsKms,
            #[cfg(feature = "aws")]
            Self::AwsSecretsManagerV2(_) => KeyManagementSystem::AwsSecretManager,
            #[cfg(feature = "google")]
            Self::GoogleKms(_) => KeyManagementSystem::GoogleKms,
            #[cfg(feature = "google")]
            Self::GoogleSecretManager(_) => KeyManagementSystem::GoogleSecretManager,
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
        SecretManager::Local => Ok(environment
            .get(secret_name)
            .map(SecretValue::new)
            .map(Secret::String)),
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
        SecretManager::GoogleSecretManager(client) => client
            .get_secret_from_google_secret_manager(secret_name)
            .await
            .map_err(Error::from),
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
