use crate::AuthError;
use crate::auth::error::AuthConfigurationError;
use crate::auth::{
    CredentialFileRef, CredentialLookup, CredentialRef, ResolvedCredential, SecretValue,
    TokenProviderHandle,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::native::{NativeAzureRequest, NativeAzureTokenAcquirer};
use super::types::{AzureAuthInputs, AzureCredentialType, ConfigValue, DEFAULT_AZURE_SCOPE};

const AZURE_AD_TOKEN_ENV: &str = "AZURE_AD_TOKEN";
const AZURE_TENANT_ID_ENV: &str = "AZURE_TENANT_ID";
const AZURE_CLIENT_ID_ENV: &str = "AZURE_CLIENT_ID";
const AZURE_CLIENT_SECRET_ENV: &str = "AZURE_CLIENT_SECRET";
const AZURE_SCOPE_ENV: &str = "AZURE_SCOPE";
const AZURE_AUTHORITY_HOST_ENV: &str = "AZURE_AUTHORITY_HOST";
const AZURE_CREDENTIAL_ENV: &str = "AZURE_CREDENTIAL";
const AZURE_FEDERATED_TOKEN_FILE_ENV: &str = "AZURE_FEDERATED_TOKEN_FILE";

#[derive(Clone, Debug)]
pub(crate) enum AzureCredentialPlan {
    Supplied(ResolvedCredential),
    Caller(TokenProviderHandle),
    Oidc {
        reference: CredentialRef,
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
    native: Arc<dyn AzureTokenAcquirer>,
}

trait AzureTokenAcquirer: Send + Sync {
    fn acquire(
        &self,
        request: NativeAzureRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ResolvedCredential, AuthError>> + Send + '_>>;
}

impl AzureTokenAcquirer for NativeAzureTokenAcquirer {
    fn acquire(
        &self,
        request: NativeAzureRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ResolvedCredential, AuthError>> + Send + '_>> {
        Box::pin(NativeAzureTokenAcquirer::acquire(self, request))
    }
}

impl Default for AzureAuthService {
    fn default() -> Self {
        Self {
            native: Arc::new(NativeAzureTokenAcquirer::default()),
        }
    }
}

impl AzureAuthService {
    #[cfg(test)]
    fn with_acquirer(native: Arc<dyn AzureTokenAcquirer>) -> Self {
        Self { native }
    }

    pub(crate) async fn resolve(
        &self,
        inputs: &AzureAuthInputs,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Option<ResolvedCredential>, AuthError> {
        match select_auth_plan(inputs, env_lookup)? {
            AzureCredentialPlan::Supplied(credential) => Ok(Some(credential)),
            AzureCredentialPlan::Caller(caller) => {
                let credential = caller.acquire().await?;
                if credential.secret().expose().is_empty() {
                    return Err(AuthError::EmptyAzureToken);
                }
                Ok(Some(credential))
            }
            AzureCredentialPlan::Oidc {
                reference,
                tenant_id,
                client_id,
                scope,
                authority,
            } => {
                let assertion = resolve_reference(inputs, env_lookup, &reference)
                    .await?
                    .ok_or(AuthError::UnresolvedOidcReference)?;
                self.native
                    .acquire(NativeAzureRequest::ClientAssertion {
                        tenant_id,
                        client_id,
                        assertion,
                        assertion_identity: format!("{reference:?}"),
                        scope,
                        authority,
                    })
                    .await
                    .map(Some)
            }
            AzureCredentialPlan::Native(request) => self.native.acquire(request).await.map(Some),
            AzureCredentialPlan::Chain(requests) => {
                let mut failures = Vec::new();
                for request in requests {
                    match self.native.acquire(request).await {
                        Ok(credential) => return Ok(Some(credential)),
                        Err(error) => failures.push(error),
                    }
                }
                Err(AuthError::CredentialChain(failures))
            }
            AzureCredentialPlan::Missing => Ok(None),
        }
    }
}

pub(crate) fn select_auth_plan(
    inputs: &AzureAuthInputs,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<AzureCredentialPlan, AuthError> {
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
        .map_err(|_| AuthError::Configuration(AuthConfigurationError::InvalidAzureSelector))?;
    let federated_token_file = configured_string(
        &inputs.federated_token_file,
        AZURE_FEDERATED_TOKEN_FILE_ENV,
        env_lookup,
    );

    if inputs.azure_ad_token_provider.is_none()
        && let (Some(tenant_id), Some(client_id), Some(client_secret)) =
            (tenant_id.clone(), client_id.clone(), client_secret)
    {
        return Ok(AzureCredentialPlan::Native(
            NativeAzureRequest::ClientSecret {
                tenant_id,
                client_id,
                client_secret,
                scope,
                authority,
            },
        ));
    }

    if let (Some(reference), Some(tenant_id), Some(client_id)) = (
        oidc_reference(&token)?,
        tenant_id.clone(),
        client_id.clone(),
    ) {
        return Ok(AzureCredentialPlan::Oidc {
            reference,
            tenant_id,
            client_id,
            scope,
            authority,
        });
    }

    if let Some(caller) = &inputs.azure_ad_token_provider {
        return Ok(AzureCredentialPlan::Caller(caller.clone()));
    }

    if let Some(token) = token {
        return Ok(AzureCredentialPlan::Supplied(
            ResolvedCredential::AccessToken {
                token,
                expires_on: None,
            },
        ));
    }

    if !inputs.enable_azure_ad_token_refresh && selector.is_none() {
        return Ok(AzureCredentialPlan::Missing);
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
) -> Result<AzureCredentialPlan, AuthError> {
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
        AzureCredentialType::ClientSecretCredential => Err(AuthError::Configuration(
            AuthConfigurationError::MissingClientSecretFields,
        )),
        AzureCredentialType::WorkloadIdentityCredential => Ok(AzureCredentialPlan::Native(
            workload_request(tenant_id, client_id, federated_token_file, scope, authority)?,
        )),
        AzureCredentialType::ManagedIdentityCredential => Ok(AzureCredentialPlan::Native(
            NativeAzureRequest::ManagedIdentity { client_id, scope },
        )),
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
            Ok(AzureCredentialPlan::Chain(
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
            Ok(AzureCredentialPlan::Chain(
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
        tenant_id: tenant_id.ok_or(AuthError::Configuration(
            AuthConfigurationError::MissingWorkloadTenant,
        ))?,
        client_id: client_id.ok_or(AuthError::Configuration(
            AuthConfigurationError::MissingWorkloadClient,
        ))?,
        token_file_path: token_file_path.ok_or(AuthError::Configuration(
            AuthConfigurationError::MissingWorkloadTokenFile,
        ))?,
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

async fn resolve_reference(
    inputs: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    reference: &CredentialRef,
) -> Result<Option<SecretValue>, AuthError> {
    let lookup = match reference {
        CredentialRef::Explicit(secret) => return Ok(Some(secret.clone())),
        CredentialRef::Env(name) => env_lookup(name)
            .filter(|value| !value.is_empty())
            .map(SecretValue::new)
            .map_or(CredentialLookup::Missing, CredentialLookup::Found),
        CredentialRef::None => return Ok(None),
        CredentialRef::File(_) | CredentialRef::Request(_) | CredentialRef::Host(_) => {
            let resolver = inputs
                .credential_resolver
                .as_ref()
                .ok_or(AuthError::Configuration(
                    AuthConfigurationError::MissingHostResolver,
                ))?;
            resolver.resolve(reference).await?
        }
    };
    Ok(match lookup {
        CredentialLookup::Found(secret) => Some(secret),
        CredentialLookup::Missing | CredentialLookup::Declined => None,
    })
}

fn oidc_reference(token: &Option<SecretValue>) -> Result<Option<CredentialRef>, AuthError> {
    let Some(value) = token.as_ref().map(SecretValue::expose) else {
        return Ok(None);
    };
    if let Some(name) = value.strip_prefix("oidc/env/") {
        return non_empty_reference(name, "OIDC environment reference")
            .map(CredentialRef::Env)
            .map(Some);
    }
    if let Some(name) = value.strip_prefix("oidc/env_path/") {
        return non_empty_reference(name, "OIDC environment path reference")
            .map(|name| CredentialRef::File(CredentialFileRef::EnvironmentVariable(name)))
            .map(Some);
    }
    if let Some(path) = value.strip_prefix("oidc/file/") {
        let path = non_empty_reference(path, "OIDC file reference")?;
        return Ok(Some(CredentialRef::File(CredentialFileRef::Path(
            path.into(),
        ))));
    }
    if value.starts_with("oidc/") {
        return Err(AuthError::Configuration(
            AuthConfigurationError::UnsupportedOidcReference,
        ));
    }
    Ok(None)
}

fn non_empty_reference(value: &str, kind: &str) -> Result<String, AuthError> {
    if value.is_empty() {
        return Err(AuthError::Configuration(
            AuthConfigurationError::EmptyReference(kind.to_string()),
        ));
    }
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use super::{
        AzureAuthService, AzureCredentialPlan, AzureTokenAcquirer, oidc_reference,
        resolve_reference, select_auth_plan,
    };
    use crate::AuthError;
    use crate::auth::ResolvedCredential;
    use crate::auth::azure::AzureAuthInputs;
    use crate::auth::azure::native::NativeAzureRequest;
    use crate::auth::{
        CredentialFileRef, CredentialLookup, CredentialLookupFuture, CredentialRef,
        CredentialResolver, CredentialResolverHandle, SecretValue,
    };

    #[derive(Debug)]
    struct FileResolver;

    struct ChainAcquirer {
        requests: Mutex<Vec<&'static str>>,
        succeed_on: Option<&'static str>,
    }

    impl AzureTokenAcquirer for ChainAcquirer {
        fn acquire(
            &self,
            request: NativeAzureRequest,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<ResolvedCredential, AuthError>> + Send + '_>,
        > {
            let kind = match request {
                NativeAzureRequest::ManagedIdentity { .. } => "managed-identity",
                NativeAzureRequest::DeveloperTools { .. } => "developer-tools",
                NativeAzureRequest::WorkloadIdentity { .. } => "workload-identity",
                NativeAzureRequest::ClientSecret { .. } => "client-secret",
                NativeAzureRequest::ClientAssertion { .. } => "client-assertion",
            };
            self.requests.lock().unwrap().push(kind);
            Box::pin(async move {
                if self.succeed_on == Some(kind) {
                    Ok(ResolvedCredential::AccessToken {
                        token: SecretValue::new("chain-token"),
                        expires_on: None,
                    })
                } else {
                    Err(AuthError::AzureTokenAcquisition(format!("{kind} failed")))
                }
            })
        }
    }

    impl CredentialResolver for FileResolver {
        fn resolve<'a>(&'a self, reference: &'a CredentialRef) -> CredentialLookupFuture<'a> {
            Box::pin(async move {
                Ok(match reference {
                    CredentialRef::File(CredentialFileRef::Path(path))
                        if path == std::path::Path::new("/run/secrets/assertion") =>
                    {
                        CredentialLookup::Found(SecretValue::new("rotated-assertion"))
                    }
                    _ => CredentialLookup::Declined,
                })
            })
        }
    }

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

        assert!(matches!(plan, AzureCredentialPlan::Native(_)));
    }

    #[test]
    fn supplied_token_does_not_require_refresh() {
        let params = json!({"azure_ad_token": "token"});
        let inputs = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();

        assert!(matches!(
            select_auth_plan(&inputs, &|_| None).unwrap(),
            AzureCredentialPlan::Supplied(_)
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
            AzureCredentialPlan::Oidc {
                reference: CredentialRef::Env(name),
                ..
            } if name == "ASSERTION"
        ));
    }

    #[test]
    fn oidc_file_location_is_typed_before_resolution() {
        assert_eq!(
            oidc_reference(&Some(SecretValue::new("oidc/file//run/secrets/assertion"))).unwrap(),
            Some(CredentialRef::File(CredentialFileRef::Path(
                "/run/secrets/assertion".into()
            )))
        );
    }

    #[test]
    fn unsupported_oidc_reference_is_rejected_during_plan_creation() {
        let error = oidc_reference(&Some(SecretValue::new("oidc/vault/assertion")))
            .expect_err("unsupported backend must fail validation");

        assert!(error.to_string().contains("unsupported OIDC reference"));
    }

    #[tokio::test]
    async fn host_resolver_owns_file_access() {
        let inputs = AzureAuthInputs {
            credential_resolver: Some(CredentialResolverHandle::new(Arc::new(FileResolver))),
            ..AzureAuthInputs::default()
        };
        let reference =
            CredentialRef::File(CredentialFileRef::Path("/run/secrets/assertion".into()));

        let resolved = resolve_reference(&inputs, &|_| None, &reference)
            .await
            .unwrap();

        assert_eq!(resolved, Some(SecretValue::new("rotated-assertion")));
    }

    #[tokio::test]
    async fn default_chain_uses_declared_order_and_stops_after_success() {
        let acquirer = Arc::new(ChainAcquirer {
            requests: Mutex::new(Vec::new()),
            succeed_on: Some("developer-tools"),
        });
        let service = AzureAuthService::with_acquirer(acquirer.clone());
        let inputs = AzureAuthInputs {
            enable_azure_ad_token_refresh: true,
            ..Default::default()
        };

        let credential = service.resolve(&inputs, &|_| None).await.unwrap().unwrap();

        assert_eq!(credential.secret().expose(), "chain-token");
        assert_eq!(
            *acquirer.requests.lock().unwrap(),
            ["managed-identity", "developer-tools"]
        );
    }

    #[tokio::test]
    async fn chain_reports_each_acquisition_failure() {
        let acquirer = Arc::new(ChainAcquirer {
            requests: Mutex::new(Vec::new()),
            succeed_on: None,
        });
        let service = AzureAuthService::with_acquirer(acquirer);
        let inputs = AzureAuthInputs {
            enable_azure_ad_token_refresh: true,
            ..Default::default()
        };

        let error = service.resolve(&inputs, &|_| None).await.unwrap_err();

        assert!(matches!(error, AuthError::CredentialChain(errors) if errors.len() == 2));
    }
}
