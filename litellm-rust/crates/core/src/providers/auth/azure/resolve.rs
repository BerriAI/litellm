use std::sync::Arc;

use crate::error::AuthError;
use crate::providers::auth::secret::SecretValue;
use crate::providers::auth::token::{ResolvedCredential, TokenCallerHandle};

use super::native::{NativeAzureRequest, NativeAzureTokenAcquirer};
use super::types::{
    AzureAuthInputs, AzureCredentialType, AzureValueLookupHandle, ConfigValue, DEFAULT_AZURE_SCOPE,
    ProcessAzureValueLookup,
};

const AZURE_AD_TOKEN_ENV: &str = "AZURE_AD_TOKEN";
const AZURE_TENANT_ID_ENV: &str = "AZURE_TENANT_ID";
const AZURE_CLIENT_ID_ENV: &str = "AZURE_CLIENT_ID";
const AZURE_CLIENT_SECRET_ENV: &str = "AZURE_CLIENT_SECRET";
const AZURE_SCOPE_ENV: &str = "AZURE_SCOPE";
const AZURE_AUTHORITY_HOST_ENV: &str = "AZURE_AUTHORITY_HOST";
const AZURE_CREDENTIAL_ENV: &str = "AZURE_CREDENTIAL";
const AZURE_FEDERATED_TOKEN_FILE_ENV: &str = "AZURE_FEDERATED_TOKEN_FILE";

#[derive(Clone, Debug)]
pub(crate) enum AzureAuthPlan {
    Supplied(ResolvedCredential),
    Caller(TokenCallerHandle),
    Oidc {
        reference: String,
        tenant_id: String,
        client_id: String,
        scope: String,
        authority: Option<String>,
    },
    Native(NativeAzureRequest),
    Chain(Vec<NativeAzureRequest>),
    Missing,
}

pub(crate) struct AzureAuthService {
    native: NativeAzureTokenAcquirer,
    lookup: AzureValueLookupHandle,
}

impl Default for AzureAuthService {
    fn default() -> Self {
        Self {
            native: NativeAzureTokenAcquirer::default(),
            lookup: AzureValueLookupHandle::new(Arc::new(ProcessAzureValueLookup)),
        }
    }
}

impl AzureAuthService {
    pub(crate) async fn resolve(
        &self,
        inputs: &AzureAuthInputs,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Option<ResolvedCredential>, AuthError> {
        match select_auth_plan(inputs, env_lookup)? {
            AzureAuthPlan::Supplied(credential) => Ok(Some(credential)),
            AzureAuthPlan::Caller(caller) => {
                let credential = caller.acquire().await?;
                if credential.secret().expose().is_empty() {
                    return Err(AuthError::Caller(
                        "Azure AD token provider returned an empty token".to_string(),
                    ));
                }
                Ok(Some(credential))
            }
            AzureAuthPlan::Oidc {
                reference,
                tenant_id,
                client_id,
                scope,
                authority,
            } => {
                let assertion = self.lookup.resolve(&reference).await?.ok_or_else(|| {
                    AuthError::Acquisition(
                        "Azure OIDC reference did not resolve to a value".to_string(),
                    )
                })?;
                self.native
                    .acquire(NativeAzureRequest::ClientAssertion {
                        tenant_id,
                        client_id,
                        assertion,
                        assertion_identity: reference,
                        scope,
                        authority,
                    })
                    .await
                    .map(Some)
            }
            AzureAuthPlan::Native(request) => self.native.acquire(request).await.map(Some),
            AzureAuthPlan::Chain(requests) => {
                let mut failures = Vec::new();
                for request in requests {
                    match self.native.acquire(request).await {
                        Ok(credential) => return Ok(Some(credential)),
                        Err(error) => failures.push(error.to_string()),
                    }
                }
                Err(AuthError::Acquisition(failures.join("; ")))
            }
            AzureAuthPlan::Missing => Ok(None),
        }
    }
}

pub(crate) fn select_auth_plan(
    inputs: &AzureAuthInputs,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<AzureAuthPlan, AuthError> {
    let token = configured_secret(&inputs.azure_ad_token, AZURE_AD_TOKEN_ENV, env_lookup);
    let tenant_id = configured_string(&inputs.tenant_id, AZURE_TENANT_ID_ENV, env_lookup);
    let client_id = configured_string(&inputs.client_id, AZURE_CLIENT_ID_ENV, env_lookup);
    let client_secret =
        configured_secret(&inputs.client_secret, AZURE_CLIENT_SECRET_ENV, env_lookup);
    let scope = configured_string(&inputs.azure_scope, AZURE_SCOPE_ENV, env_lookup)
        .unwrap_or_else(|| DEFAULT_AZURE_SCOPE.to_string());
    let authority = configured_string(
        &inputs.azure_authority_host,
        AZURE_AUTHORITY_HOST_ENV,
        env_lookup,
    );
    let selector = configured_string(&inputs.azure_credential, AZURE_CREDENTIAL_ENV, env_lookup)
        .map(|value| value.parse::<AzureCredentialType>())
        .transpose()
        .map_err(|_| {
            AuthError::InvalidConfiguration("invalid Azure credential selector".to_string())
        })?;
    let federated_token_file = configured_string(
        &inputs.federated_token_file,
        AZURE_FEDERATED_TOKEN_FILE_ENV,
        env_lookup,
    );

    if inputs.azure_ad_token_provider.is_none()
        && let (Some(tenant_id), Some(client_id), Some(client_secret)) =
            (tenant_id.clone(), client_id.clone(), client_secret)
    {
        return Ok(AzureAuthPlan::Native(NativeAzureRequest::ClientSecret {
            tenant_id,
            client_id,
            client_secret,
            scope,
            authority,
        }));
    }

    if let (Some(reference), Some(tenant_id), Some(client_id)) =
        (oidc_reference(&token), tenant_id.clone(), client_id.clone())
    {
        return Ok(AzureAuthPlan::Oidc {
            reference,
            tenant_id,
            client_id,
            scope,
            authority,
        });
    }

    if let Some(caller) = &inputs.azure_ad_token_provider {
        return Ok(AzureAuthPlan::Caller(caller.clone()));
    }

    if let Some(token) = token {
        return Ok(AzureAuthPlan::Supplied(ResolvedCredential::AccessToken {
            token,
            expires_on: None,
        }));
    }

    if !inputs.enable_azure_ad_token_refresh && selector.is_none() {
        return Ok(AzureAuthPlan::Missing);
    }

    select_native_plan(
        selector,
        tenant_id,
        client_id,
        federated_token_file,
        scope,
        authority,
    )
}

fn select_native_plan(
    selector: Option<AzureCredentialType>,
    tenant_id: Option<String>,
    client_id: Option<String>,
    federated_token_file: Option<String>,
    scope: String,
    authority: Option<String>,
) -> Result<AzureAuthPlan, AuthError> {
    let selected = selector.unwrap_or_else(|| {
        if federated_token_file.is_some() {
            AzureCredentialType::DefaultAzureCredential
        } else if client_id.is_some() {
            AzureCredentialType::ManagedIdentityCredential
        } else {
            AzureCredentialType::DefaultAzureCredential
        }
    });

    match selected {
        AzureCredentialType::ClientSecretCredential => Err(AuthError::InvalidConfiguration(
            "ClientSecretCredential requires tenant_id, client_id, and client_secret".to_string(),
        )),
        AzureCredentialType::WorkloadIdentityCredential => Ok(AzureAuthPlan::Native(
            workload_request(tenant_id, client_id, federated_token_file, scope, authority)?,
        )),
        AzureCredentialType::ManagedIdentityCredential => {
            Ok(AzureAuthPlan::Native(NativeAzureRequest::ManagedIdentity {
                client_id,
                scope,
            }))
        }
        AzureCredentialType::DefaultAzureCredential => {
            let workload = match (tenant_id, client_id.clone(), federated_token_file) {
                (Some(tenant_id), Some(client_id), Some(token_file_path)) => {
                    Some(NativeAzureRequest::WorkloadIdentity {
                        tenant_id,
                        client_id,
                        token_file_path,
                        scope: scope.clone(),
                        authority,
                    })
                }
                _ => None,
            };
            Ok(AzureAuthPlan::Chain(
                workload
                    .into_iter()
                    .chain(std::iter::once(NativeAzureRequest::ManagedIdentity {
                        client_id,
                        scope: scope.clone(),
                    }))
                    .chain(std::iter::once(NativeAzureRequest::DeveloperTools {
                        scope,
                    }))
                    .collect(),
            ))
        }
        AzureCredentialType::DeploymentIdentityCredential => {
            let workload = match (tenant_id, client_id.clone(), federated_token_file) {
                (Some(tenant_id), Some(client_id), Some(token_file_path)) => {
                    Some(NativeAzureRequest::WorkloadIdentity {
                        tenant_id,
                        client_id,
                        token_file_path,
                        scope: scope.clone(),
                        authority,
                    })
                }
                _ => None,
            };
            let user_assigned = client_id.map(|client_id| NativeAzureRequest::ManagedIdentity {
                client_id: Some(client_id),
                scope: scope.clone(),
            });
            Ok(AzureAuthPlan::Chain(
                workload
                    .into_iter()
                    .chain(user_assigned)
                    .chain(std::iter::once(NativeAzureRequest::ManagedIdentity {
                        client_id: None,
                        scope,
                    }))
                    .collect(),
            ))
        }
    }
}

fn workload_request(
    tenant_id: Option<String>,
    client_id: Option<String>,
    token_file_path: Option<String>,
    scope: String,
    authority: Option<String>,
) -> Result<NativeAzureRequest, AuthError> {
    Ok(NativeAzureRequest::WorkloadIdentity {
        tenant_id: tenant_id.ok_or_else(|| {
            AuthError::InvalidConfiguration(
                "WorkloadIdentityCredential requires tenant_id".to_string(),
            )
        })?,
        client_id: client_id.ok_or_else(|| {
            AuthError::InvalidConfiguration(
                "WorkloadIdentityCredential requires client_id".to_string(),
            )
        })?,
        token_file_path: token_file_path.ok_or_else(|| {
            AuthError::InvalidConfiguration(
                "WorkloadIdentityCredential requires azure_federated_token_file".to_string(),
            )
        })?,
        scope,
        authority,
    })
}

fn configured_string(
    configured: &ConfigValue<String>,
    environment_name: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    configured
        .as_value()
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| env_lookup(environment_name).filter(|value| !value.is_empty()))
}

fn configured_secret(
    configured: &ConfigValue<SecretValue>,
    environment_name: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<SecretValue> {
    configured
        .as_value()
        .filter(|value| !value.expose().is_empty())
        .cloned()
        .or_else(|| {
            env_lookup(environment_name)
                .filter(|value| !value.is_empty())
                .map(SecretValue::new)
        })
}

fn oidc_reference(token: &Option<SecretValue>) -> Option<String> {
    token
        .as_ref()
        .map(SecretValue::expose)
        .filter(|value| value.starts_with("oidc/"))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{AzureAuthPlan, select_auth_plan};
    use crate::providers::auth::azure::AzureAuthInputs;

    #[test]
    fn null_and_empty_values_fall_back_to_environment() {
        let params = json!({"tenant_id": null, "client_id": "", "client_secret": null});
        let inputs = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();
        let plan = select_auth_plan(&inputs, &|name| match name {
            "AZURE_TENANT_ID" => Some("tenant".to_string()),
            "AZURE_CLIENT_ID" => Some("client".to_string()),
            "AZURE_CLIENT_SECRET" => Some("secret".to_string()),
            _ => None,
        })
        .unwrap();

        assert!(matches!(plan, AzureAuthPlan::Native(_)));
    }

    #[test]
    fn supplied_token_does_not_require_refresh() {
        let params = json!({"azure_ad_token": "token"});
        let inputs = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();

        assert!(matches!(
            select_auth_plan(&inputs, &|_| None).unwrap(),
            AzureAuthPlan::Supplied(_)
        ));
    }

    #[test]
    fn oidc_reference_is_deferred() {
        let params = json!({
            "azure_ad_token": "oidc/env/ASSERTION",
            "tenant_id": "tenant",
            "client_id": "client"
        });
        let inputs = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();

        assert!(matches!(
            select_auth_plan(&inputs, &|_| None).unwrap(),
            AzureAuthPlan::Oidc { .. }
        ));
    }
}
