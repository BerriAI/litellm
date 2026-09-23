use litellm_core_utils::settings::Lookup;
use litellm_secrets::Secret;
use litellm_secrets::{Error, SecretManager};
use litellm_secrets_types::{PythonSecretRead, SecretOperationContext};

pub(super) struct PythonReadRequest {
    pub secret_name: String,
    pub primary_secret_name: Option<String>,
    pub context: SecretOperationContext,
    pub synchronous: bool,
}

pub(super) async fn read_python_provider(
    manager: &SecretManager,
    request: &PythonReadRequest,
    _environment: &(dyn Lookup + Send + Sync),
) -> Result<PythonSecretRead, Error> {
    match (manager, &request.context) {
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
        (SecretManager::HashicorpVault(client), SecretOperationContext::Hashicorp(context)) => {
            Ok(PythonSecretRead::Value(
                client
                    .async_read_secret_with_context(&request.secret_name, context)
                    .await
                    .unwrap_or(None)
                    .map(Secret::String),
            ))
        }
        (SecretManager::Cyberark(client), SecretOperationContext::Cyberark(_)) => {
            Ok(PythonSecretRead::Value(
                client
                    .read_for_python(&request.secret_name)
                    .await
                    .unwrap_or(None)
                    .map(Secret::String),
            ))
        }
        (SecretManager::GoogleSecretManager(client), SecretOperationContext::Google(_)) => client
            .get_secret_for_python(&request.secret_name)
            .await
            .map(PythonSecretRead::Value)
            .map_err(Error::from),
        _ => Err(Error::NativeBackendUnavailable),
    }
}

#[derive(Debug)]
pub(super) enum PythonMutationError {
    Unsupported,
    Vault(Box<litellm_secrets::hashicorp::PythonFailure>),
    CyberarkWrite {
        name: String,
        failure: Box<litellm_secrets::cyberark::PythonWriteFailure>,
    },
    CurrentMissing(String),
    ReplacementMissing(String),
    ReplacementMismatch,
}

pub(super) async fn write_python_provider(
    manager: &SecretManager,
    name: &str,
    value: &litellm_secrets::SecretValue,
) -> Result<serde_json::Value, PythonMutationError> {
    match manager {
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

pub(super) async fn delete_python_provider(
    manager: &SecretManager,
    name: &str,
) -> Result<serde_json::Value, PythonMutationError> {
    match manager {
        SecretManager::Cyberark(client) => {
            client
                .async_delete_secret(name, None)
                .await
                .map_err(|failure| PythonMutationError::CyberarkWrite {
                    name: name.to_owned(),
                    failure: Box::new(litellm_secrets::cyberark::PythonWriteFailure {
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

pub(super) async fn rotate_python_provider(
    manager: &SecretManager,
    current_name: &str,
    new_name: &str,
    value: &litellm_secrets::SecretValue,
) -> Result<serde_json::Value, PythonMutationError> {
    match manager {
        SecretManager::Cyberark(client) => {
            use litellm_secrets::cyberark::PythonRotationFailure;
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
fn write_success(name: &str) -> serde_json::Value {
    serde_json::json!({"status": "success", "message": format!("Secret {name} written successfully")})
}

pub(super) enum PythonMutationResponse {
    Value(serde_json::Value),
    Json(Vec<u8>),
}

pub(super) async fn write_python_provider_with_context(
    manager: &SecretManager,
    name: &str,
    value: &litellm_secrets::SecretValue,
    context: &litellm_secrets_types::SecretWriteContext,
) -> Result<PythonMutationResponse, PythonMutationError> {
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

pub(super) async fn delete_python_provider_with_context(
    manager: &SecretManager,
    name: &str,
    context: &SecretOperationContext,
) -> Result<PythonMutationResponse, PythonMutationError> {
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

pub(super) async fn rotate_python_provider_with_context(
    manager: &SecretManager,
    current_name: &str,
    new_name: &str,
    value: &litellm_secrets::SecretValue,
    context: &SecretOperationContext,
) -> Result<PythonMutationResponse, PythonMutationError> {
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
