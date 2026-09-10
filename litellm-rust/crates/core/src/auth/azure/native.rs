use crate::auth::error::AuthConfigurationError;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use azure_core::cloud::{CloudConfiguration, CustomConfiguration};
use azure_core::credentials::{Secret, TokenCredential};
use azure_core::http::ClientOptions;
use azure_identity::{
    ClientAssertion, ClientAssertionCredential, ClientAssertionCredentialOptions,
    ClientSecretCredential, ClientSecretCredentialOptions, DeveloperToolsCredential,
    ManagedIdentityCredential, ManagedIdentityCredentialOptions, UserAssignedId,
    WorkloadIdentityCredential, WorkloadIdentityCredentialOptions,
};
use sha2::{Digest, Sha256};

use crate::AuthError;
use crate::auth::secret::SecretValue;
use crate::auth::token::ResolvedCredential;
use crate::auth::{InputSource, Sourced};

use super::credential_provider_cache::{
    AzureCredentialProviderCache, AzureCredentialProviderCacheKey,
};

#[derive(Clone, Debug)]
pub(crate) enum NativeAzureRequest {
    ClientSecret {
        tenant_id: Sourced<String>,
        client_id: Sourced<String>,
        client_secret: Sourced<SecretValue>,
        scope: Sourced<String>,
        authority: Option<Sourced<String>>,
    },
    ClientAssertion {
        tenant_id: Sourced<String>,
        client_id: Sourced<String>,
        assertion: Sourced<SecretValue>,
        assertion_identity: String,
        scope: Sourced<String>,
        authority: Option<Sourced<String>>,
    },
    WorkloadIdentity {
        tenant_id: Sourced<String>,
        client_id: Sourced<String>,
        token_file_path: Sourced<String>,
        scope: Sourced<String>,
        authority: Option<Sourced<String>>,
    },
    ManagedIdentity {
        client_id: Option<Sourced<String>>,
        scope: Sourced<String>,
        selection_source: InputSource,
    },
    DeveloperTools {
        scope: Sourced<String>,
        selection_source: InputSource,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct ValidatedAzureRequest {
    request: NativeAzureRequest,
    credential_source: InputSource,
}

impl ValidatedAzureRequest {
    pub(crate) fn new(request: NativeAzureRequest) -> Result<Self, AuthError> {
        validate_authority(&request)?;
        let credential_source = validate_sources(&request)?;
        Ok(Self {
            request,
            credential_source,
        })
    }

    pub(crate) fn credential_source(&self) -> InputSource {
        self.credential_source
    }

    #[cfg(test)]
    pub(super) fn kind(&self) -> &'static str {
        match self.request {
            NativeAzureRequest::ClientSecret { .. } => "client-secret",
            NativeAzureRequest::ClientAssertion { .. } => "client-assertion",
            NativeAzureRequest::WorkloadIdentity { .. } => "workload-identity",
            NativeAzureRequest::ManagedIdentity { .. } => "managed-identity",
            NativeAzureRequest::DeveloperTools { .. } => "developer-tools",
        }
    }
}

pub(crate) struct NativeAzureTokenAcquirer {
    cache: AzureCredentialProviderCache,
    transport: Option<azure_core::http::Transport>,
}

impl Default for NativeAzureTokenAcquirer {
    fn default() -> Self {
        Self::new(64)
    }
}

impl NativeAzureTokenAcquirer {
    pub(crate) fn new(cache_capacity: u64) -> Self {
        Self {
            cache: AzureCredentialProviderCache::new(cache_capacity),
            transport: None,
        }
    }

    #[cfg(test)]
    pub(super) fn with_transport(
        cache_capacity: u64,
        transport: azure_core::http::Transport,
    ) -> Self {
        Self {
            cache: AzureCredentialProviderCache::new(cache_capacity),
            transport: Some(transport),
        }
    }

    pub(crate) async fn acquire(
        &self,
        request: ValidatedAzureRequest,
    ) -> Result<ResolvedCredential, AuthError> {
        let scope = request.request.scope().to_string();
        let key = request.request.cache_key();
        let transport = self.transport.clone();
        let credential = self
            .cache
            .get_or_create(
                key,
                async move { build_credential(request.request, transport) },
            )
            .await?;
        let token = credential
            .get_token(&[scope.as_str()], None)
            .await
            .map_err(|error| AuthError::AzureTokenAcquisition(error.to_string()))?;
        let expires_on = u64::try_from(token.expires_on.unix_timestamp())
            .ok()
            .map(|seconds| UNIX_EPOCH + Duration::from_secs(seconds));

        Ok(ResolvedCredential::AccessToken {
            token: SecretValue::new(token.token.secret()),
            expires_on,
        })
    }
}

impl NativeAzureRequest {
    fn scope(&self) -> &str {
        match self {
            Self::ClientSecret { scope, .. }
            | Self::ClientAssertion { scope, .. }
            | Self::WorkloadIdentity { scope, .. }
            | Self::ManagedIdentity { scope, .. }
            | Self::DeveloperTools { scope, .. } => scope.value(),
        }
    }

    fn cache_key(&self) -> AzureCredentialProviderCacheKey {
        match self {
            Self::ClientSecret {
                tenant_id,
                client_id,
                client_secret,
                scope,
                authority,
            } => AzureCredentialProviderCacheKey {
                mechanism: "client-secret",
                authority: authority
                    .as_ref()
                    .map(|value| value.value().clone())
                    .unwrap_or_default(),
                tenant_id: tenant_id.value().clone(),
                client_id: client_id.value().clone(),
                scope: scope.value().clone(),
                secret_identity: secret_digest(client_secret.value().expose()),
            },
            Self::ClientAssertion {
                tenant_id,
                client_id,
                assertion,
                assertion_identity,
                scope,
                authority,
            } => AzureCredentialProviderCacheKey {
                mechanism: "client-assertion",
                authority: authority
                    .as_ref()
                    .map(|value| value.value().clone())
                    .unwrap_or_default(),
                tenant_id: tenant_id.value().clone(),
                client_id: client_id.value().clone(),
                scope: scope.value().clone(),
                secret_identity: format!(
                    "{assertion_identity}:{}",
                    secret_digest(assertion.value().expose())
                ),
            },
            Self::WorkloadIdentity {
                tenant_id,
                client_id,
                token_file_path,
                scope,
                authority,
            } => AzureCredentialProviderCacheKey {
                mechanism: "workload-identity",
                authority: authority
                    .as_ref()
                    .map(|value| value.value().clone())
                    .unwrap_or_default(),
                tenant_id: tenant_id.value().clone(),
                client_id: client_id.value().clone(),
                scope: scope.value().clone(),
                secret_identity: token_file_path.value().clone(),
            },
            Self::ManagedIdentity {
                client_id, scope, ..
            } => AzureCredentialProviderCacheKey {
                mechanism: "managed-identity",
                authority: String::new(),
                tenant_id: String::new(),
                client_id: client_id
                    .as_ref()
                    .map(|value| value.value().clone())
                    .unwrap_or_default(),
                scope: scope.value().clone(),
                secret_identity: String::new(),
            },
            Self::DeveloperTools { scope, .. } => AzureCredentialProviderCacheKey {
                mechanism: "developer-tools",
                authority: String::new(),
                tenant_id: String::new(),
                client_id: String::new(),
                scope: scope.value().clone(),
                secret_identity: String::new(),
            },
        }
    }
}

fn validate_authority(request: &NativeAzureRequest) -> Result<(), AuthError> {
    let authority = match request {
        NativeAzureRequest::ClientSecret { authority, .. }
        | NativeAzureRequest::ClientAssertion { authority, .. }
        | NativeAzureRequest::WorkloadIdentity { authority, .. } => authority.as_ref(),
        NativeAzureRequest::ManagedIdentity { .. } | NativeAzureRequest::DeveloperTools { .. } => {
            None
        }
    };
    let Some(authority) = authority else {
        return Ok(());
    };
    let url = url::Url::parse(authority.value())
        .map_err(|_| AuthError::Configuration(AuthConfigurationError::InvalidAzureAuthority))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err(AuthError::Configuration(
            AuthConfigurationError::InvalidAzureAuthority,
        ));
    }
    Ok(())
}

fn validate_sources(request: &NativeAzureRequest) -> Result<InputSource, AuthError> {
    match request {
        NativeAzureRequest::ClientSecret {
            tenant_id,
            client_id,
            client_secret,
            scope,
            authority,
        } => {
            let identity_sources = [
                tenant_id.source(),
                client_id.source(),
                client_secret.source(),
            ];
            let request_identity = identity_sources.contains(&InputSource::Request);
            if request_identity
                && !identity_sources
                    .iter()
                    .all(|source| *source == InputSource::Request)
            {
                return mixed_sources();
            }
            if !request_identity && is_request_controlled(scope, authority.as_ref()) {
                return mixed_sources();
            }
            Ok(if request_identity {
                InputSource::Request
            } else {
                trusted_source(&identity_sources)
            })
        }
        NativeAzureRequest::ClientAssertion {
            tenant_id,
            client_id,
            assertion,
            scope,
            authority,
            ..
        } => trusted_only(&[
            tenant_id.source(),
            client_id.source(),
            assertion.source(),
            scope.source(),
            authority
                .as_ref()
                .map(Sourced::source)
                .unwrap_or(InputSource::Environment),
        ]),
        NativeAzureRequest::WorkloadIdentity {
            tenant_id,
            client_id,
            token_file_path,
            scope,
            authority,
        } => trusted_only(&[
            tenant_id.source(),
            client_id.source(),
            token_file_path.source(),
            scope.source(),
            authority
                .as_ref()
                .map(Sourced::source)
                .unwrap_or(InputSource::Environment),
        ]),
        NativeAzureRequest::ManagedIdentity {
            client_id,
            scope,
            selection_source,
        } => trusted_only(&[
            client_id
                .as_ref()
                .map(Sourced::source)
                .unwrap_or(InputSource::Environment),
            scope.source(),
            *selection_source,
        ]),
        NativeAzureRequest::DeveloperTools {
            scope,
            selection_source,
        } => trusted_only(&[scope.source(), *selection_source]),
    }
}

fn is_request_controlled<T>(value: &Sourced<T>, optional: Option<&Sourced<String>>) -> bool {
    value.source() == InputSource::Request
        || optional.is_some_and(|value| value.source() == InputSource::Request)
}

fn trusted_only(sources: &[InputSource]) -> Result<InputSource, AuthError> {
    if sources.contains(&InputSource::Request) {
        return mixed_sources();
    }
    Ok(trusted_source(sources))
}

fn trusted_source(sources: &[InputSource]) -> InputSource {
    if sources.contains(&InputSource::Deployment) {
        InputSource::Deployment
    } else {
        InputSource::Environment
    }
}

fn mixed_sources<T>() -> Result<T, AuthError> {
    Err(AuthError::Configuration(
        AuthConfigurationError::MixedAzureCredentialSources,
    ))
}

fn build_credential(
    request: NativeAzureRequest,
    transport: Option<azure_core::http::Transport>,
) -> Result<Arc<dyn TokenCredential>, AuthError> {
    match request {
        NativeAzureRequest::ClientSecret {
            tenant_id,
            client_id,
            client_secret,
            authority,
            ..
        } => ClientSecretCredential::new(
            tenant_id.value(),
            client_id.into_value(),
            Secret::new(client_secret.value().expose().to_string()),
            Some(ClientSecretCredentialOptions {
                client_options: client_options(authority.map(Sourced::into_value), transport),
            }),
        )
        .map(|credential| credential as Arc<dyn TokenCredential>),
        NativeAzureRequest::ClientAssertion {
            tenant_id,
            client_id,
            assertion,
            authority,
            ..
        } => ClientAssertionCredential::new(
            tenant_id.into_value(),
            client_id.into_value(),
            StaticAssertion(assertion.into_value()),
            Some(ClientAssertionCredentialOptions {
                client_options: client_options(authority.map(Sourced::into_value), transport),
            }),
        )
        .map(|credential| credential as Arc<dyn TokenCredential>),
        NativeAzureRequest::WorkloadIdentity {
            tenant_id,
            client_id,
            token_file_path,
            authority,
            ..
        } => WorkloadIdentityCredential::new(Some(WorkloadIdentityCredentialOptions {
            credential_options: azure_identity::ClientAssertionCredentialOptions {
                client_options: client_options(authority.map(Sourced::into_value), transport),
            },
            client_id: Some(client_id.into_value()),
            tenant_id: Some(tenant_id.into_value()),
            token_file_path: Some(token_file_path.into_value().into()),
        }))
        .map(|credential| credential as Arc<dyn TokenCredential>),
        NativeAzureRequest::ManagedIdentity { client_id, .. } => {
            ManagedIdentityCredential::new(Some(ManagedIdentityCredentialOptions {
                user_assigned_id: client_id
                    .map(Sourced::into_value)
                    .map(UserAssignedId::ClientId),
                client_options: client_options(None, transport),
            }))
            .map(|credential| credential as Arc<dyn TokenCredential>)
        }
        NativeAzureRequest::DeveloperTools { .. } => DeveloperToolsCredential::new(None)
            .map(|credential| credential as Arc<dyn TokenCredential>),
    }
    .map_err(|error| {
        AuthError::Configuration(AuthConfigurationError::AzureCredentialInitialization(
            error.to_string(),
        ))
    })
}

fn client_options(
    authority: Option<String>,
    transport: Option<azure_core::http::Transport>,
) -> ClientOptions {
    let cloud = authority.map(|authority_host| {
        let mut custom = CustomConfiguration::default();
        custom.authority_host = authority_host;
        Arc::new(CloudConfiguration::from(custom))
    });
    ClientOptions {
        cloud,
        transport,
        ..Default::default()
    }
}

fn secret_digest(secret: &str) -> String {
    format!("{:x}", Sha256::digest(secret.as_bytes()))
}

#[derive(Debug)]
struct StaticAssertion(SecretValue);

impl ClientAssertion for StaticAssertion {
    fn secret<'life0, 'life1, 'async_trait>(
        &'life0 self,
        _options: Option<azure_core::http::ClientMethodOptions<'life1>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = azure_core::Result<String>> + Send + 'async_trait>,
    >
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { Ok(self.0.expose().to_string()) })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use azure_core::http::headers::Headers;
    use azure_core::http::{AsyncRawResponse, HttpClient, Request, StatusCode, Transport};
    use azure_core::{Bytes, Result};

    use super::{NativeAzureRequest, NativeAzureTokenAcquirer, ValidatedAzureRequest};
    use crate::auth::{InputSource, SecretValue, Sourced};

    fn deployment<T>(value: T) -> Sourced<T> {
        Sourced::new(value, InputSource::Deployment)
    }

    fn sourced_client_secret(
        credential_source: InputSource,
        authority_source: InputSource,
        authority: &str,
    ) -> NativeAzureRequest {
        NativeAzureRequest::ClientSecret {
            tenant_id: Sourced::new("tenant".to_string(), credential_source),
            client_id: Sourced::new("client".to_string(), credential_source),
            client_secret: Sourced::new(SecretValue::new("secret"), credential_source),
            scope: Sourced::new("scope".to_string(), InputSource::Environment),
            authority: Some(Sourced::new(authority.to_string(), authority_source)),
        }
    }

    fn client_secret_request(
        tenant: &str,
        client: &str,
        secret: &str,
        scope: &str,
        authority: &str,
    ) -> ValidatedAzureRequest {
        ValidatedAzureRequest::new(NativeAzureRequest::ClientSecret {
            tenant_id: deployment(tenant.to_string()),
            client_id: deployment(client.to_string()),
            client_secret: deployment(SecretValue::new(secret)),
            scope: deployment(scope.to_string()),
            authority: Some(deployment(authority.to_string())),
        })
        .unwrap()
    }

    #[derive(Debug, Default)]
    struct RecordingTokenClient {
        requests: Mutex<Vec<(String, String)>>,
    }

    impl HttpClient for RecordingTokenClient {
        fn execute_request<'life0, 'life1, 'async_trait>(
            &'life0 self,
            request: &'life1 Request,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<AsyncRawResponse>> + Send + 'async_trait>,
        >
        where
            'life0: 'async_trait,
            'life1: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move {
                let body = Bytes::from(request.body());
                self.requests.lock().unwrap().push((
                    request.url().to_string(),
                    String::from_utf8(body.to_vec()).unwrap(),
                ));
                Ok(AsyncRawResponse::from_bytes(
                    StatusCode::Ok,
                    Headers::new(),
                    r#"{"token_type":"Bearer","expires_in":3600,"ext_expires_in":3600,"access_token":"native-token"}"#,
                ))
            })
        }
    }

    #[tokio::test]
    async fn client_secret_uses_sdk_protocol_and_reuses_cached_credential() {
        let transport = Arc::new(RecordingTokenClient::default());
        let acquirer =
            NativeAzureTokenAcquirer::with_transport(4, Transport::new(transport.clone()));
        let request = client_secret_request(
            "tenant",
            "client",
            "secret",
            "https://service.test/.default",
            "https://login.test",
        );

        let first = acquirer.acquire(request.clone()).await.unwrap();
        let second = acquirer.acquire(request).await.unwrap();

        assert_eq!(first.secret().expose(), "native-token");
        assert_eq!(second.secret().expose(), "native-token");
        let requests = transport.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].0, "https://login.test/tenant/oauth2/v2.0/token");
        assert!(requests[0].1.contains("client_id=client"));
        assert!(requests[0].1.contains("client_secret=secret"));
        assert!(
            requests[0]
                .1
                .contains("scope=https%3A%2F%2Fservice.test%2F.default")
        );
    }

    #[tokio::test]
    async fn credential_provider_cache_isolates_every_client_secret_identity_field() {
        let transport = Arc::new(RecordingTokenClient::default());
        let acquirer =
            NativeAzureTokenAcquirer::with_transport(16, Transport::new(transport.clone()));
        let request = client_secret_request;
        let base = request("tenant", "client", "secret", "scope", "https://login.test");
        let variants = [
            base.clone(),
            request(
                "other-tenant",
                "client",
                "secret",
                "scope",
                "https://login.test",
            ),
            request(
                "tenant",
                "other-client",
                "secret",
                "scope",
                "https://login.test",
            ),
            request(
                "tenant",
                "client",
                "other-secret",
                "scope",
                "https://login.test",
            ),
            request(
                "tenant",
                "client",
                "secret",
                "other-scope",
                "https://login.test",
            ),
            request(
                "tenant",
                "client",
                "secret",
                "scope",
                "https://other-login.test",
            ),
        ];

        acquirer.acquire(base.clone()).await.unwrap();
        acquirer.acquire(base).await.unwrap();
        for request in variants.into_iter().skip(1) {
            acquirer.acquire(request).await.unwrap();
        }

        assert_eq!(transport.requests.lock().unwrap().len(), 6);
    }

    #[test]
    fn request_authority_requires_request_owned_client_secret_identity() {
        let error = ValidatedAzureRequest::new(sourced_client_secret(
            InputSource::Deployment,
            InputSource::Request,
            "https://login.example",
        ))
        .unwrap_err();

        assert!(matches!(
            error,
            crate::AuthError::Configuration(
                crate::auth::error::AuthConfigurationError::MixedAzureCredentialSources
            )
        ));
    }

    #[test]
    fn request_owned_client_secret_identity_can_select_custom_authority() {
        let request = ValidatedAzureRequest::new(sourced_client_secret(
            InputSource::Request,
            InputSource::Request,
            "https://login.example",
        ))
        .unwrap();

        assert_eq!(request.credential_source(), InputSource::Request);
    }

    #[test]
    fn authority_is_restricted_to_an_https_origin() {
        for authority in [
            "http://login.example",
            "https://user@login.example",
            "https://login.example/tenant",
            "https://login.example?target=other",
        ] {
            let error = ValidatedAzureRequest::new(sourced_client_secret(
                InputSource::Deployment,
                InputSource::Deployment,
                authority,
            ))
            .unwrap_err();
            assert!(matches!(
                error,
                crate::AuthError::Configuration(
                    crate::auth::error::AuthConfigurationError::InvalidAzureAuthority
                )
            ));
        }
    }
}
