#[cfg(any(feature = "hashicorp", feature = "cyberark"))]
use crate::Secret;
use crate::{Error, SecretManager};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{PythonSecretRead, SecretOperationContext};

pub struct PythonReadRequest {
    pub secret_name: String,
    pub primary_secret_name: Option<String>,
    pub context: SecretOperationContext,
    pub synchronous: bool,
}

pub async fn read_python_provider(
    manager: &SecretManager,
    request: &PythonReadRequest,
    _environment: &(dyn Lookup + Send + Sync),
) -> Result<PythonSecretRead, Error> {
    match (manager, &request.context) {
        #[cfg(feature = "aws")]
        (SecretManager::AwsSecretsManagerV2(client), SecretOperationContext::Aws(context)) => {
            client
                .read_provider_payload_for_python(
                    &request.secret_name,
                    request.primary_secret_name.as_deref(),
                    context,
                    request.synchronous,
                    _environment,
                )
                .await
                .map_err(Error::from)
        }
        #[cfg(feature = "hashicorp")]
        (SecretManager::HashicorpVault(client), SecretOperationContext::Hashicorp(context)) => {
            Ok(PythonSecretRead::Value(
                client
                    .async_read_secret_with_context(&request.secret_name, context)
                    .await
                    .unwrap_or(None)
                    .map(Secret::String),
            ))
        }
        #[cfg(feature = "cyberark")]
        (SecretManager::Cyberark(client), SecretOperationContext::Cyberark(context)) => {
            Ok(PythonSecretRead::Value(
                client
                    .async_read_secret_with_context(&request.secret_name, context)
                    .await
                    .unwrap_or(None)
                    .map(Secret::String),
            ))
        }
        #[cfg(feature = "google")]
        (SecretManager::GoogleSecretManager(client), SecretOperationContext::Google(_)) => client
            .get_secret_for_python(&request.secret_name)
            .await
            .map(PythonSecretRead::Value)
            .map_err(Error::from),
        _ => Err(Error::NativeBackendUnavailable),
    }
}
