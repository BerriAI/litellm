use crate::native::CredentialAttempt;
use litellm_auth_types::{
    CredentialFileRef, CredentialLookup, CredentialRef, InputSource, ResolvedCredential, Sourced,
    TokenProviderHandle,
};
use litellm_secret_value::SecretValue;

use litellm_auth_types::{Error, ErrorDetail};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::native::{NativeAzureRequest, NativeAzureTokenAcquirer, ValidatedAzureRequest};
use super::types::{AzureAuthInputs, AzureCredentialType, ConfigValue, DEFAULT_AZURE_SCOPE};

const AZURE_AD_TOKEN_ENV: &str = "AZURE_AD_TOKEN";
const AZURE_TENANT_ID_ENV: &str = "AZURE_TENANT_ID";
const AZURE_CLIENT_ID_ENV: &str = "AZURE_CLIENT_ID";
const AZURE_CLIENT_SECRET_ENV: &str = "AZURE_CLIENT_SECRET";
const AZURE_SCOPE_ENV: &str = "AZURE_SCOPE";
const AZURE_AUTHORITY_HOST_ENV: &str = "AZURE_AUTHORITY_HOST";
const AZURE_CREDENTIAL_ENV: &str = "AZURE_CREDENTIAL";
const AZURE_FEDERATED_TOKEN_FILE_ENV: &str = "AZURE_FEDERATED_TOKEN_FILE";

pub const SECRET_NAMES: &[&str] = &[
    AZURE_AD_TOKEN_ENV,
    AZURE_TENANT_ID_ENV,
    AZURE_CLIENT_ID_ENV,
    AZURE_CLIENT_SECRET_ENV,
    AZURE_SCOPE_ENV,
    AZURE_AUTHORITY_HOST_ENV,
    AZURE_CREDENTIAL_ENV,
    AZURE_FEDERATED_TOKEN_FILE_ENV,
];

#[derive(Clone, Debug)]
pub(crate) enum AzureCredentialPlan {
    Supplied(Sourced<ResolvedCredential>),
    Caller(TokenProviderHandle),
    Oidc {
        reference: Sourced<CredentialRef>,
        tenant_id: Sourced<String>,
        client_id: Sourced<String>,
        scope: Sourced<String>,
        authority: Option<Sourced<String>>,
    },
    Native(ValidatedAzureRequest),
    Chain(Vec<ValidatedAzureRequest>),
    Missing,
}

/// Rust counterpart to Python's `get_azure_ad_token`, not `BaseAzureLLM`.
pub struct AzureAuthService {
    native: Arc<dyn AzureTokenAcquirer>,
}

trait AzureTokenAcquirer: Send + Sync {
    fn acquire(
        &self,
        request: ValidatedAzureRequest,
    ) -> Pin<Box<dyn Future<Output = CredentialAttempt> + Send + '_>>;
}

impl AzureTokenAcquirer for NativeAzureTokenAcquirer {
    fn acquire(
        &self,
        request: ValidatedAzureRequest,
    ) -> Pin<Box<dyn Future<Output = CredentialAttempt> + Send + '_>> {
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

    pub async fn identity_token(
        &self,
        identity: &AzureIdentity,
        scope: &str,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Option<ResolvedCredential>, Error> {
        let plan = identity_plan(identity, scope, env_lookup)?;
        let inputs = AzureAuthInputs::default_credential_for_scope(scope);
        Ok(self
            .run_plan(plan, &inputs, env_lookup)
            .await?
            .map(Sourced::into_value))
    }

    pub async fn get_azure_ad_token(
        &self,
        inputs: &AzureAuthInputs,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Option<Sourced<ResolvedCredential>>, Error> {
        self.run_plan(select_auth_plan(inputs, env_lookup)?, inputs, env_lookup)
            .await
    }

    async fn run_plan(
        &self,
        plan: AzureCredentialPlan,
        inputs: &AzureAuthInputs,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Option<Sourced<ResolvedCredential>>, Error> {
        match plan {
            AzureCredentialPlan::Supplied(credential) => Ok(Some(credential)),
            AzureCredentialPlan::Caller(caller) => {
                let credential = caller.acquire().await?;
                if credential.secret().expose().is_empty() {
                    return Err(Error::EmptyCallerCredential {
                        caller: "Azure AD token provider",
                    });
                }
                Ok(Some(Sourced::new(credential, InputSource::Deployment)))
            }
            AzureCredentialPlan::Oidc {
                reference,
                tenant_id,
                client_id,
                scope,
                authority,
            } => {
                let assertion = resolve_reference(inputs, env_lookup, reference.value())
                    .await?
                    .ok_or(Error::CredentialAcquisition(ErrorDetail::Unresolved {
                        subject: "Azure OIDC reference",
                    }))?;
                let request = ValidatedAzureRequest::new(NativeAzureRequest::ClientAssertion {
                    tenant_id,
                    client_id,
                    assertion: Sourced::new(assertion, reference.source()),
                    assertion_identity: format!("{:?}", reference.value()),
                    scope,
                    authority,
                })?;
                let source = request.credential_source();
                self.native
                    .acquire(request)
                    .await
                    .into_result()
                    .map(|credential| Sourced::new(credential, source))
                    .map(Some)
            }
            AzureCredentialPlan::Native(request) => {
                let source = request.credential_source();
                self.native
                    .acquire(request)
                    .await
                    .into_result()
                    .map(|credential| Some(Sourced::new(credential, source)))
            }
            AzureCredentialPlan::Chain(requests) => {
                let mut failures = Vec::new();
                for request in requests {
                    let source = request.credential_source();
                    match self.native.acquire(request).await {
                        CredentialAttempt::Resolved(credential) => {
                            return Ok(Some(Sourced::new(credential, source)));
                        }
                        CredentialAttempt::Unavailable(error) => failures.push(error),
                        CredentialAttempt::Failed(error) => {
                            return Err(Error::CredentialChain {
                                last: Box::new(error),
                                earlier: failures,
                            });
                        }
                    }
                }
                match failures.pop() {
                    Some(last) => Err(Error::CredentialChain {
                        last: Box::new(last),
                        earlier: failures,
                    }),
                    None => Ok(None),
                }
            }
            AzureCredentialPlan::Missing => Ok(None),
        }
    }
}

/// The identity azure-identity builds for one explicit credential class, without LiteLLM's provider settings
/// (`AZURE_AD_TOKEN`, `AZURE_CREDENTIAL`, `AZURE_SCOPE`) that `get_azure_ad_token` also reads.
#[derive(Clone, Debug)]
pub enum AzureIdentity {
    ClientSecret {
        tenant_id: String,
        client_id: String,
        client_secret: SecretValue,
    },
    ManagedIdentity {
        client_id: String,
    },
    Default,
}

fn identity_plan(
    identity: &AzureIdentity,
    scope: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<AzureCredentialPlan, Error> {
    let deployment = |value: &str| Sourced::new(value.to_owned(), InputSource::Deployment);
    let environment = |name: &str| {
        env_lookup(name)
            .filter(|value| !value.trim().is_empty())
            .map(|value| Sourced::new(value, InputSource::Environment))
    };
    let scope = deployment(scope);
    let authority = environment(AZURE_AUTHORITY_HOST_ENV);
    match identity {
        AzureIdentity::ClientSecret {
            tenant_id,
            client_id,
            client_secret,
        } => Ok(AzureCredentialPlan::Native(ValidatedAzureRequest::new(
            NativeAzureRequest::ClientSecret {
                tenant_id: deployment(tenant_id),
                client_id: deployment(client_id),
                client_secret: Sourced::new(client_secret.clone(), InputSource::Deployment),
                scope,
                authority,
            },
        )?)),
        AzureIdentity::ManagedIdentity { client_id } => Ok(AzureCredentialPlan::Native(
            ValidatedAzureRequest::new(NativeAzureRequest::ManagedIdentity {
                client_id: Some(deployment(client_id)),
                scope,
                selection_source: InputSource::Deployment,
            })?,
        )),
        AzureIdentity::Default => {
            let tenant_id = environment(AZURE_TENANT_ID_ENV);
            let client_id = environment(AZURE_CLIENT_ID_ENV);
            let from_environment = match (
                tenant_id.clone(),
                client_id.clone(),
                environment(AZURE_CLIENT_SECRET_ENV),
            ) {
                (Some(tenant_id), Some(client_id), Some(client_secret)) => {
                    Some(NativeAzureRequest::ClientSecret {
                        tenant_id,
                        client_id,
                        client_secret: client_secret.map(SecretValue::new),
                        scope: scope.clone(),
                        authority: authority.clone(),
                    })
                }
                _ => None,
            };
            let probe_imds =
                env_lookup("IDENTITY_ENDPOINT").is_none() && env_lookup("MSI_ENDPOINT").is_none();
            let AzureCredentialPlan::Chain(chain) = select_native_plan(
                Some(Sourced::new(
                    AzureCredentialType::DefaultAzureCredential,
                    InputSource::Deployment,
                )),
                tenant_id,
                client_id,
                environment(AZURE_FEDERATED_TOKEN_FILE_ENV),
                scope,
                authority,
                InputSource::Environment,
            )?
            else {
                unreachable!("DefaultAzureCredential always plans a chain")
            };
            let environment_request = from_environment
                .map(ValidatedAzureRequest::new)
                .transpose()?;
            Ok(AzureCredentialPlan::Chain(
                environment_request
                    .into_iter()
                    .chain(chain)
                    .map(|request| request.with_imds_probe(probe_imds))
                    .collect(),
            ))
        }
    }
}

pub(crate) fn select_auth_plan(
    inputs: &AzureAuthInputs,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<AzureCredentialPlan, Error> {
    let token = configured_secret(&inputs.azure_ad_token, AZURE_AD_TOKEN_ENV, env_lookup);
    let tenant_id = configured_string(&inputs.tenant_id, AZURE_TENANT_ID_ENV, env_lookup);
    let client_id = configured_string(&inputs.client_id, AZURE_CLIENT_ID_ENV, env_lookup);
    let client_secret =
        configured_secret(&inputs.client_secret, AZURE_CLIENT_SECRET_ENV, env_lookup);
    let scope = configured_string(&inputs.azure_scope, AZURE_SCOPE_ENV, env_lookup)
        .unwrap_or_else(|| Sourced::new(DEFAULT_AZURE_SCOPE.to_string(), InputSource::Environment));
    let authority = configured_string(
        &inputs.azure_authority_host,
        AZURE_AUTHORITY_HOST_ENV,
        env_lookup,
    );
    let selector = configured_string(&inputs.azure_credential, AZURE_CREDENTIAL_ENV, env_lookup)
        .map(|value| {
            value
                .value()
                .parse::<AzureCredentialType>()
                .map(|selector| Sourced::new(selector, value.source()))
        })
        .transpose()
        .map_err(|error| {
            Error::InvalidConfiguration(ErrorDetail::failed(
                "Azure credential selector parsing",
                error,
            ))
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
        return Ok(AzureCredentialPlan::Native(ValidatedAzureRequest::new(
            NativeAzureRequest::ClientSecret {
                tenant_id,
                client_id,
                client_secret,
                scope,
                authority,
            },
        )?));
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
        return Ok(AzureCredentialPlan::Supplied(token.map(|token| {
            ResolvedCredential::AccessToken {
                token,
                expires_on: None,
            }
        })));
    }

    if !*inputs.enable_azure_ad_token_refresh.value() && selector.is_none() {
        return Ok(AzureCredentialPlan::Missing);
    }

    let probe_imds = selector
        .as_ref()
        .is_none_or(|selector| *selector.value() == AzureCredentialType::DefaultAzureCredential)
        && env_lookup("IDENTITY_ENDPOINT").is_none()
        && env_lookup("MSI_ENDPOINT").is_none();
    match select_native_plan(
        selector,
        tenant_id,
        client_id,
        federated_token_file,
        scope,
        authority,
        inputs.enable_azure_ad_token_refresh.source(),
    )? {
        AzureCredentialPlan::Chain(requests) => Ok(AzureCredentialPlan::Chain(
            requests
                .into_iter()
                .map(|request| request.with_imds_probe(probe_imds))
                .collect(),
        )),
        plan => Ok(plan),
    }
}

fn select_native_plan(
    selector: Option<Sourced<AzureCredentialType>>,
    tenant_id: Option<Sourced<String>>,
    client_id: Option<Sourced<String>>,
    federated_token_file: Option<Sourced<String>>,
    scope: Sourced<String>,
    authority: Option<Sourced<String>>,
    refresh_source: InputSource,
) -> Result<AzureCredentialPlan, Error> {
    let selected = selector.unwrap_or_else(|| {
        Sourced::new(
            {
                if federated_token_file.is_some() {
                    AzureCredentialType::DefaultAzureCredential
                } else if client_id.is_some() {
                    AzureCredentialType::ManagedIdentityCredential
                } else {
                    AzureCredentialType::DefaultAzureCredential
                }
            },
            refresh_source,
        )
    });
    let selection_source = selected.source();

    match selected.into_value() {
        AzureCredentialType::ClientSecretCredential => {
            Err(Error::InvalidConfiguration(ErrorDetail::MissingSettings {
                credential: "ClientSecretCredential",
                settings: "tenant_id, client_id, and client_secret",
            }))
        }
        AzureCredentialType::WorkloadIdentityCredential => {
            Ok(AzureCredentialPlan::Native(ValidatedAzureRequest::new(
                workload_request(tenant_id, client_id, federated_token_file, scope, authority)?,
            )?))
        }
        AzureCredentialType::ManagedIdentityCredential => Ok(AzureCredentialPlan::Native(
            ValidatedAzureRequest::new(NativeAzureRequest::ManagedIdentity {
                client_id,
                scope,
                selection_source,
            })?,
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
                        selection_source,
                    }))
                    .chain(std::iter::once(NativeAzureRequest::DeveloperTools {
                        scope,
                        selection_source,
                    }))
                    .map(ValidatedAzureRequest::new)
                    .collect::<Result<Vec<_>, _>>()?,
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
                selection_source,
            });
            Ok(AzureCredentialPlan::Chain(
                workload
                    .into_iter()
                    .chain(user_assigned)
                    .chain(std::iter::once(NativeAzureRequest::ManagedIdentity {
                        client_id: None,
                        scope,
                        selection_source,
                    }))
                    .map(ValidatedAzureRequest::new)
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
    }
}

fn workload_request(
    tenant_id: Option<Sourced<String>>,
    client_id: Option<Sourced<String>>,
    token_file_path: Option<Sourced<String>>,
    scope: Sourced<String>,
    authority: Option<Sourced<String>>,
) -> Result<NativeAzureRequest, Error> {
    Ok(NativeAzureRequest::WorkloadIdentity {
        tenant_id: tenant_id.ok_or(workload_requires("tenant_id"))?,
        client_id: client_id.ok_or(workload_requires("client_id"))?,
        token_file_path: token_file_path.ok_or(workload_requires("azure_federated_token_file"))?,
        scope,
        authority,
    })
}

fn workload_requires(settings: &'static str) -> Error {
    Error::InvalidConfiguration(ErrorDetail::MissingSettings {
        credential: "WorkloadIdentityCredential",
        settings,
    })
}

fn configured_string(
    configured: &ConfigValue<String>,
    environment_name: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<Sourced<String>> {
    configured
        .as_value()
        .filter(|value| !value.value().is_empty())
        .cloned()
        .or_else(|| {
            env_lookup(environment_name)
                .filter(|value| !value.is_empty())
                .map(|value| Sourced::new(value, InputSource::Environment))
        })
}

fn configured_secret(
    configured: &ConfigValue<SecretValue>,
    environment_name: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<Sourced<SecretValue>> {
    configured
        .as_value()
        .filter(|value| !value.value().expose().is_empty())
        .cloned()
        .or_else(|| {
            env_lookup(environment_name)
                .filter(|value| !value.is_empty())
                .map(|value| Sourced::new(SecretValue::new(value), InputSource::Environment))
        })
}

async fn resolve_reference(
    inputs: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    reference: &CredentialRef,
) -> Result<Option<SecretValue>, Error> {
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
                .ok_or(Error::InvalidConfiguration(ErrorDetail::MissingResolver))?;
            resolver.resolve(reference).await?
        }
    };
    Ok(match lookup {
        CredentialLookup::Found(secret) => Some(secret),
        CredentialLookup::Missing | CredentialLookup::Declined => None,
    })
}

fn oidc_reference(
    token: &Option<Sourced<SecretValue>>,
) -> Result<Option<Sourced<CredentialRef>>, Error> {
    let Some(token) = token.as_ref() else {
        return Ok(None);
    };
    let value = token.value().expose();
    if token.source() == InputSource::Request && value.starts_with("oidc/") {
        return Err(Error::InvalidConfiguration(
            ErrorDetail::RequestControlledReference { provider: "Azure" },
        ));
    }
    if let Some(name) = value.strip_prefix("oidc/env/") {
        return non_empty_reference(name, "OIDC environment reference")
            .map(CredentialRef::Env)
            .map(|reference| Sourced::new(reference, token.source()))
            .map(Some);
    }
    if let Some(name) = value.strip_prefix("oidc/env_path/") {
        return non_empty_reference(name, "OIDC environment path reference")
            .map(|name| CredentialRef::File(CredentialFileRef::EnvironmentVariable(name)))
            .map(|reference| Sourced::new(reference, token.source()))
            .map(Some);
    }
    if let Some(path) = value.strip_prefix("oidc/file/") {
        let path = non_empty_reference(path, "OIDC file reference")?;
        return Ok(Some(Sourced::new(
            CredentialRef::File(CredentialFileRef::Path(path.into())),
            token.source(),
        )));
    }
    if value.starts_with("oidc/") {
        return Err(Error::InvalidConfiguration(
            ErrorDetail::UnsupportedReference,
        ));
    }
    Ok(None)
}

fn non_empty_reference(value: &str, kind: &'static str) -> Result<String, Error> {
    if value.is_empty() {
        return Err(Error::InvalidConfiguration(ErrorDetail::Empty {
            subject: kind,
        }));
    }
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::future::Future;
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use super::{
        AzureAuthService, AzureCredentialPlan, AzureTokenAcquirer, SECRET_NAMES, oidc_reference,
        resolve_reference, select_auth_plan,
    };
    use crate::native::ValidatedAzureRequest;
    use crate::types::AzureAuthInputs;
    use litellm_auth_types::Error;
    use litellm_auth_types::ResolvedCredential;
    use litellm_auth_types::{
        CredentialFileRef, CredentialLookup, CredentialLookupFuture, CredentialRef,
        CredentialResolver, CredentialResolverHandle, InputSource, Sourced,
    };
    use litellm_secret_value::SecretValue;

    #[derive(Debug)]
    struct FileResolver;

    use crate::native::CredentialAttempt;

    struct ChainAcquirer {
        terminal_on: Option<&'static str>,
        requests: Mutex<Vec<&'static str>>,
        succeed_on: Option<&'static str>,
    }

    impl AzureTokenAcquirer for ChainAcquirer {
        fn acquire(
            &self,
            request: ValidatedAzureRequest,
        ) -> std::pin::Pin<Box<dyn Future<Output = CredentialAttempt> + Send + '_>> {
            let kind = request.kind();
            self.requests.lock().unwrap().push(kind);
            Box::pin(async move {
                if self.succeed_on == Some(kind) {
                    CredentialAttempt::Resolved(ResolvedCredential::AccessToken {
                        token: SecretValue::new("chain-token"),
                        expires_on: None,
                    })
                } else {
                    let error =
                        Error::CredentialAcquisition(litellm_auth_types::ErrorDetail::Unresolved {
                            subject: kind,
                        });
                    if self.terminal_on == Some(kind) {
                        CredentialAttempt::Failed(error)
                    } else {
                        CredentialAttempt::Unavailable(error)
                    }
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
    fn secret_names_cover_environment_reads() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(BTreeSet::<String>::new()));
        let recorded = seen.clone();
        let inputs = AzureAuthInputs::default();
        select_auth_plan(&inputs, &|name| {
            recorded.lock().unwrap().insert(name.to_string());
            None
        })
        .unwrap();
        assert!(
            seen.lock()
                .unwrap()
                .iter()
                .all(|name| SECRET_NAMES.contains(&name.as_str()))
        );
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
                reference,
                ..
            } if reference.value() == &CredentialRef::Env("ASSERTION".to_string())
        ));
    }

    #[test]
    fn oidc_file_location_is_typed_before_resolution() {
        assert_eq!(
            oidc_reference(&Some(Sourced::new(
                SecretValue::new("oidc/file//run/secrets/assertion"),
                InputSource::Deployment,
            )))
            .unwrap()
            .map(Sourced::into_value),
            Some(CredentialRef::File(CredentialFileRef::Path(
                "/run/secrets/assertion".into()
            )))
        );
    }

    #[test]
    fn unsupported_oidc_reference_is_rejected_during_plan_creation() {
        let error = oidc_reference(&Some(Sourced::new(
            SecretValue::new("oidc/vault/assertion"),
            InputSource::Deployment,
        )))
        .expect_err("unsupported backend must fail validation");

        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("unsupported OIDC reference")
        );
    }

    #[rstest::rstest]
    fn request_oidc_reference_is_rejected_before_lookup() {
        let params = json!({
            "azure_ad_token": "oidc/env/ASSERTION",
            "tenant_id": "tenant",
            "client_id": "client"
        });
        let sources = std::collections::BTreeMap::from([
            ("azure_ad_token".to_string(), InputSource::Request),
            ("tenant_id".to_string(), InputSource::Request),
            ("client_id".to_string(), InputSource::Request),
        ]);
        let inputs =
            AzureAuthInputs::from_sourced_optional_params(params.as_object().unwrap(), &sources)
                .unwrap();

        let error = select_auth_plan(&inputs, &|name| {
            assert_ne!(name, "ASSERTION");
            None
        })
        .unwrap_err();

        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("request-controlled Azure credential references are not allowed"),
            "{error:?}"
        );
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

    #[derive(Debug)]
    struct UnavailableImdsClient;

    impl azure_core::http::HttpClient for UnavailableImdsClient {
        fn execute_request<'life0, 'life1, 'async_trait>(
            &'life0 self,
            _request: &'life1 azure_core::http::Request,
        ) -> std::pin::Pin<
            Box<
                dyn Future<Output = azure_core::Result<azure_core::http::AsyncRawResponse>>
                    + Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(std::future::pending())
        }
    }

    struct OfflineChainAcquirer(super::NativeAzureTokenAcquirer);

    impl AzureTokenAcquirer for OfflineChainAcquirer {
        fn acquire(
            &self,
            request: ValidatedAzureRequest,
        ) -> std::pin::Pin<Box<dyn Future<Output = CredentialAttempt> + Send + '_>> {
            Box::pin(async move {
                if request.kind() == "developer-tools" {
                    return CredentialAttempt::Resolved(ResolvedCredential::AccessToken {
                        token: SecretValue::new("cli-token"),
                        expires_on: None,
                    });
                }
                self.0.acquire(request).await
            })
        }
    }

    #[rstest::rstest]
    #[case::default_off_azure("DefaultAzureCredential", None, true)]
    #[case::explicit_managed_identity("ManagedIdentityCredential", None, false)]
    #[case::app_service("DefaultAzureCredential", Some("http://app-service.test"), false)]
    #[tokio::test(start_paused = true)]
    async fn only_default_imds_discovery_is_bounded(
        #[case] selector: &str,
        #[case] identity_endpoint: Option<&str>,
        #[case] expect_cli: bool,
    ) {
        let native = super::NativeAzureTokenAcquirer::with_transport(
            4,
            azure_core::http::Transport::new(Arc::new(UnavailableImdsClient)),
        );
        let service = AzureAuthService::with_acquirer(Arc::new(OfflineChainAcquirer(native)));
        let inputs = AzureAuthInputs {
            azure_credential: crate::types::ConfigValue::Value(Sourced::new(
                selector.into(),
                InputSource::Deployment,
            )),
            enable_azure_ad_token_refresh: Sourced::new(true, InputSource::Deployment),
            ..Default::default()
        };
        let environment = |name: &str| {
            (name == "IDENTITY_ENDPOINT")
                .then_some(identity_endpoint)
                .flatten()
                .map(str::to_string)
        };
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            service.get_azure_ad_token(&inputs, &environment),
        )
        .await;

        assert_eq!(outcome.is_ok(), expect_cli);
        if expect_cli {
            assert_eq!(
                outcome.unwrap().unwrap().unwrap().value().secret().expose(),
                "cli-token"
            );
        }
    }

    #[rstest::rstest]
    #[case::litellm_provider_settings_are_ignored(
        &[("AZURE_AD_TOKEN", "provider-token"), ("AZURE_CREDENTIAL", "ClientSecretCredential")],
        &["managed-identity", "developer-tools"]
    )]
    #[case::environment_client_secret_comes_first(
        &[("AZURE_TENANT_ID", "tenant"), ("AZURE_CLIENT_ID", "client"), ("AZURE_CLIENT_SECRET", "secret")],
        &["client-secret", "managed-identity", "developer-tools"]
    )]
    #[case::aks_workload_identity(
        &[("AZURE_TENANT_ID", "tenant"), ("AZURE_CLIENT_ID", "client"), ("AZURE_FEDERATED_TOKEN_FILE", "/var/run/token")],
        &["workload-identity", "managed-identity", "developer-tools"]
    )]
    #[tokio::test]
    async fn the_default_identity_follows_azure_identity_defaults(
        #[case] environment: &[(&str, &str)],
        #[case] expected: &[&str],
    ) {
        let acquirer = Arc::new(ChainAcquirer {
            terminal_on: None,
            requests: Mutex::new(Vec::new()),
            succeed_on: Some("developer-tools"),
        });
        let service = AzureAuthService::with_acquirer(acquirer.clone());
        let lookup = |name: &str| {
            environment
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        };

        let credential = service
            .identity_token(
                &super::AzureIdentity::Default,
                "https://redis.azure.com/.default",
                &lookup,
            )
            .await
            .unwrap()
            .unwrap();

        assert_eq!(credential.secret().expose(), "chain-token");
        assert_eq!(*acquirer.requests.lock().unwrap(), expected);
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn default_chain_uses_declared_order_and_stops_after_success() {
        let acquirer = Arc::new(ChainAcquirer {
            terminal_on: None,
            requests: Mutex::new(Vec::new()),
            succeed_on: Some("developer-tools"),
        });
        let service = AzureAuthService::with_acquirer(acquirer.clone());
        let inputs = AzureAuthInputs {
            enable_azure_ad_token_refresh: Sourced::new(true, InputSource::Deployment),
            ..Default::default()
        };

        let credential = service
            .get_azure_ad_token(&inputs, &|_| None)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(credential.value().secret().expose(), "chain-token");
        assert_eq!(
            *acquirer.requests.lock().unwrap(),
            ["managed-identity", "developer-tools"]
        );
    }

    #[rstest::rstest]
    #[case::all_unavailable(false)]
    #[case::terminal_failure_after_unavailable(true)]
    #[tokio::test]
    async fn chain_reports_each_acquisition_failure(#[case] terminal: bool) {
        let acquirer = Arc::new(ChainAcquirer {
            terminal_on: terminal.then_some("developer-tools"),
            requests: Mutex::new(Vec::new()),
            succeed_on: None,
        });
        let service = AzureAuthService::with_acquirer(acquirer);
        let inputs = AzureAuthInputs {
            enable_azure_ad_token_refresh: Sourced::new(true, InputSource::Deployment),
            ..Default::default()
        };

        let error = service
            .get_azure_ad_token(&inputs, &|_| None)
            .await
            .unwrap_err();

        assert!(matches!(&error, Error::CredentialChain { earlier, .. } if earlier.len() == 1));
        let report = litellm_error::Report::new(&error).to_string();
        assert!(
            report.contains("managed-identity did not resolve"),
            "{report}"
        );
        assert!(
            report.contains("developer-tools did not resolve"),
            "{report}"
        );
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn a_configured_identity_failure_does_not_fall_back_to_a_developer_identity() {
        let acquirer = Arc::new(ChainAcquirer {
            terminal_on: Some("managed-identity"),
            requests: Mutex::new(Vec::new()),
            succeed_on: Some("developer-tools"),
        });
        let service = AzureAuthService::with_acquirer(acquirer.clone());
        let inputs = AzureAuthInputs {
            enable_azure_ad_token_refresh: Sourced::new(true, InputSource::Deployment),
            ..Default::default()
        };
        let error = service
            .get_azure_ad_token(&inputs, &|_| None)
            .await
            .unwrap_err();
        assert_eq!(*acquirer.requests.lock().unwrap(), ["managed-identity"]);
        assert!(matches!(error, Error::CredentialChain { earlier, .. } if earlier.is_empty()));
    }

    #[derive(Debug)]
    struct CallerToken(&'static str);

    impl litellm_auth_types::TokenProvider for CallerToken {
        fn acquire(&self) -> litellm_auth_types::TokenFuture<'_> {
            Box::pin(async move {
                Ok(ResolvedCredential::AccessToken {
                    token: SecretValue::new(self.0),
                    expires_on: None,
                })
            })
        }
    }

    fn caller_inputs(token: &'static str) -> AzureAuthInputs {
        let params = json!({"azure_ad_token": "static-token"});
        AzureAuthInputs {
            azure_ad_token_provider: Some(litellm_auth_types::TokenProviderHandle::new(Arc::new(
                CallerToken(token),
            ))),
            ..AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap()
        }
    }

    #[tokio::test]
    async fn caller_token_is_chosen_over_supplied_static_token() {
        let credential = AzureAuthService::default()
            .get_azure_ad_token(&caller_inputs("caller-token"), &|_| None)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(credential.value().secret().expose(), "caller-token");
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn empty_caller_token_is_rejected() {
        let error = AzureAuthService::default()
            .get_azure_ad_token(&caller_inputs(""), &|_| None)
            .await
            .unwrap_err();

        assert_eq!(
            error,
            Error::EmptyCallerCredential {
                caller: "Azure AD token provider"
            }
        );
    }
}
