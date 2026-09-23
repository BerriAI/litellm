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
        (SecretManager::Cyberark(client), SecretOperationContext::Cyberark(_)) => {
            Ok(PythonSecretRead::Value(
                client
                    .read_for_python(&request.secret_name)
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

#[derive(Debug)]
pub enum PythonMutationError {
    Unsupported,
    #[cfg(feature = "hashicorp")]
    Vault(Box<litellm_secrets_hashicorp::PythonFailure>),
    #[cfg(feature = "cyberark")]
    CyberarkWrite {
        name: String,
        failure: Box<litellm_secrets_cyberark::PythonWriteFailure>,
    },
    CurrentMissing(String),
    ReplacementMissing(String),
    ReplacementMismatch,
}

pub async fn write_python_provider(
    manager: &SecretManager,
    name: &str,
    value: &crate::SecretValue,
) -> Result<serde_json::Value, PythonMutationError> {
    #[cfg(not(feature = "cyberark"))]
    let _ = (name, value);
    match manager {
        #[cfg(feature = "cyberark")]
        SecretManager::Cyberark(client) => {
            client
                .write_for_python(name, value)
                .await
                .map_err(|failure| PythonMutationError::CyberarkWrite {
                    name: name.to_owned(),
                    failure: Box::new(failure),
                })?;
            Ok(write_success(name))
        }
        _ => Err(PythonMutationError::Unsupported),
    }
}

pub async fn delete_python_provider(
    manager: &SecretManager,
    name: &str,
) -> Result<serde_json::Value, PythonMutationError> {
    #[cfg(not(feature = "cyberark"))]
    let _ = name;
    match manager {
        #[cfg(feature = "cyberark")]
        SecretManager::Cyberark(client) => {
            client
                .async_delete_secret(name, None)
                .await
                .map_err(|failure| PythonMutationError::CyberarkWrite {
                    name: name.to_owned(),
                    failure: Box::new(litellm_secrets_cyberark::PythonWriteFailure {
                        source: failure,
                        request_url: None,
                        authentication: false,
                    }),
                })?;
            Ok(serde_json::json!({
                "status": "not_supported",
                "message": "CyberArk Conjur does not support direct secret deletion. Use policy updates to remove variables.",
            }))
        }
        _ => Err(PythonMutationError::Unsupported),
    }
}

pub async fn rotate_python_provider(
    manager: &SecretManager,
    current_name: &str,
    new_name: &str,
    value: &crate::SecretValue,
) -> Result<serde_json::Value, PythonMutationError> {
    #[cfg(not(feature = "cyberark"))]
    let _ = (current_name, new_name, value);
    match manager {
        #[cfg(feature = "cyberark")]
        SecretManager::Cyberark(client) => {
            use litellm_secrets_cyberark::PythonRotationFailure;
            client
                .rotate_for_python(current_name, new_name, value)
                .await
                .map_err(|failure| match failure {
                    PythonRotationFailure::CurrentMissing => {
                        PythonMutationError::CurrentMissing(current_name.to_owned())
                    }
                    PythonRotationFailure::Write(failure) => PythonMutationError::CyberarkWrite {
                        name: new_name.to_owned(),
                        failure: Box::new(failure),
                    },
                    PythonRotationFailure::ReplacementMissing => {
                        PythonMutationError::ReplacementMissing(new_name.to_owned())
                    }
                    PythonRotationFailure::ReplacementMismatch => {
                        PythonMutationError::ReplacementMismatch
                    }
                })?;
            Ok(write_success(new_name))
        }
        _ => Err(PythonMutationError::Unsupported),
    }
}

#[cfg(feature = "cyberark")]
fn write_success(name: &str) -> serde_json::Value {
    serde_json::json!({"status": "success", "message": format!("Secret {name} written successfully")})
}

pub enum PythonMutationResponse {
    Value(serde_json::Value),
    Json(Vec<u8>),
}

pub async fn write_python_provider_with_context(
    manager: &SecretManager,
    name: &str,
    value: &crate::SecretValue,
    context: &litellm_secrets_types::SecretWriteContext,
) -> Result<PythonMutationResponse, PythonMutationError> {
    #[cfg(not(feature = "hashicorp"))]
    let _ = context;
    #[cfg(feature = "hashicorp")]
    if let (SecretManager::HashicorpVault(client), SecretOperationContext::Hashicorp(operation)) =
        (manager, &context.operation)
    {
        return client
            .write_for_python(
                name,
                value,
                &litellm_secrets_types::SecretWriteContext {
                    description: context.description.clone(),
                    tags: context.tags.clone(),
                    operation: operation.clone(),
                },
            )
            .await
            .map(PythonMutationResponse::Json)
            .map_err(|failure| PythonMutationError::Vault(Box::new(failure)));
    }
    write_python_provider(manager, name, value)
        .await
        .map(PythonMutationResponse::Value)
}

pub async fn delete_python_provider_with_context(
    manager: &SecretManager,
    name: &str,
    context: &SecretOperationContext,
) -> Result<PythonMutationResponse, PythonMutationError> {
    #[cfg(not(feature = "hashicorp"))]
    let _ = context;
    #[cfg(feature = "hashicorp")]
    if let (SecretManager::HashicorpVault(client), SecretOperationContext::Hashicorp(context)) =
        (manager, context)
    {
        client
            .delete_for_python(name, context)
            .await
            .map_err(|failure| PythonMutationError::Vault(Box::new(failure)))?;
        return Ok(PythonMutationResponse::Value(serde_json::json!({
            "status": "success", "message": format!("Secret {name} deleted successfully"),
        })));
    }
    delete_python_provider(manager, name)
        .await
        .map(PythonMutationResponse::Value)
}

pub async fn rotate_python_provider_with_context(
    manager: &SecretManager,
    current_name: &str,
    new_name: &str,
    value: &crate::SecretValue,
    context: &SecretOperationContext,
) -> Result<PythonMutationResponse, PythonMutationError> {
    #[cfg(not(feature = "hashicorp"))]
    let _ = context;
    #[cfg(feature = "hashicorp")]
    if let (SecretManager::HashicorpVault(client), SecretOperationContext::Hashicorp(context)) =
        (manager, context)
    {
        return client
            .rotate_for_python(current_name, new_name, value, context)
            .await
            .map(PythonMutationResponse::Json)
            .map_err(|failure| PythonMutationError::Vault(Box::new(failure)));
    }
    rotate_python_provider(manager, current_name, new_name, value)
        .await
        .map(PythonMutationResponse::Value)
}
