use litellm_core_utils::settings::Lookup;
use litellm_secrets::Secret;
use litellm_secrets::cyberark::AuthenticationRetry;
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
                    .read_with_retry(
                        &request.secret_name,
                        &Default::default(),
                        AuthenticationRetry::Never,
                    )
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
    Vault(Box<super::vault::Failure>),
    CyberarkWrite {
        name: String,
        failure: Box<litellm_secrets::cyberark::WriteFailure>,
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
                .write_with_retry(name, value, &Default::default(), AuthenticationRetry::Never)
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
                    failure: Box::new(litellm_secrets::cyberark::WriteFailure {
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
            if client
                .read_fresh_with_retry(
                    current_name,
                    &Default::default(),
                    AuthenticationRetry::Never,
                )
                .await
                .ok()
                .flatten()
                .is_none()
            {
                return Err(PythonMutationError::CurrentMissing(current_name.to_owned()));
            }
            client
                .write_with_retry(
                    new_name,
                    value,
                    &Default::default(),
                    AuthenticationRetry::Never,
                )
                .await
                .map_err(|failure| PythonMutationError::CyberarkWrite {
                    name: new_name.to_owned(),
                    failure: Box::new(failure),
                })?;
            let actual = client
                .read_fresh_with_retry(new_name, &Default::default(), AuthenticationRetry::Never)
                .await
                .ok()
                .flatten()
                .ok_or_else(|| PythonMutationError::ReplacementMissing(new_name.to_owned()))?;
            if actual != *value {
                return Err(PythonMutationError::ReplacementMismatch);
            }
            if current_name != new_name {
                client.invalidate_cached_secret(current_name).await;
            }
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
        return super::vault::write(
            client,
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
        super::vault::delete(client, name, context)
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
        return super::vault::rotate(client, current_name, new_name, value, context)
            .await
            .map(PythonMutationResponse::Json)
            .map_err(|failure| PythonMutationError::Vault(Box::new(failure)));
    }
    rotate_python_provider(manager, current_name, new_name, value)
        .await
        .map(PythonMutationResponse::Value)
}
