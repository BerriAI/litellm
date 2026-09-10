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

use super::credential_provider_cache::{
    AzureCredentialProviderCache, AzureCredentialProviderCacheKey,
};

#[derive(Clone, Debug)]
pub(crate) enum NativeAzureRequest {
    ClientSecret {
        tenant_id: String,
        client_id: String,
        client_secret: SecretValue,
        scope: String,
        authority: Option<String>,
    },
    ClientAssertion {
        tenant_id: String,
        client_id: String,
        assertion: SecretValue,
        assertion_identity: String,
        scope: String,
        authority: Option<String>,
    },
    WorkloadIdentity {
        tenant_id: String,
        client_id: String,
        token_file_path: String,
        scope: String,
        authority: Option<String>,
    },
    ManagedIdentity {
        client_id: Option<String>,
        scope: String,
    },
    DeveloperTools {
        scope: String,
    },
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
        request: NativeAzureRequest,
    ) -> Result<ResolvedCredential, AuthError> {
        let scope = request.scope().to_string();
        let key = request.cache_key();
        let transport = self.transport.clone();
        let credential = self
            .cache
            .get_or_create(key, async move { build_credential(request, transport) })
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
            | Self::DeveloperTools { scope } => scope,
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
                authority: authority.clone().unwrap_or_default(),
                tenant_id: tenant_id.clone(),
                client_id: client_id.clone(),
                scope: scope.clone(),
                secret_identity: secret_digest(client_secret.expose()),
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
                authority: authority.clone().unwrap_or_default(),
                tenant_id: tenant_id.clone(),
                client_id: client_id.clone(),
                scope: scope.clone(),
                secret_identity: format!(
                    "{assertion_identity}:{}",
                    secret_digest(assertion.expose())
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
                authority: authority.clone().unwrap_or_default(),
                tenant_id: tenant_id.clone(),
                client_id: client_id.clone(),
                scope: scope.clone(),
                secret_identity: token_file_path.clone(),
            },
            Self::ManagedIdentity { client_id, scope } => AzureCredentialProviderCacheKey {
                mechanism: "managed-identity",
                authority: String::new(),
                tenant_id: String::new(),
                client_id: client_id.clone().unwrap_or_default(),
                scope: scope.clone(),
                secret_identity: String::new(),
            },
            Self::DeveloperTools { scope } => AzureCredentialProviderCacheKey {
                mechanism: "developer-tools",
                authority: String::new(),
                tenant_id: String::new(),
                client_id: String::new(),
                scope: scope.clone(),
                secret_identity: String::new(),
            },
        }
    }
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
            &tenant_id,
            client_id,
            Secret::new(client_secret.expose().to_string()),
            Some(ClientSecretCredentialOptions {
                client_options: client_options(authority, transport),
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
            tenant_id,
            client_id,
            StaticAssertion(assertion),
            Some(ClientAssertionCredentialOptions {
                client_options: client_options(authority, transport),
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
                client_options: client_options(authority, transport),
            },
            client_id: Some(client_id),
            tenant_id: Some(tenant_id),
            token_file_path: Some(token_file_path.into()),
        }))
        .map(|credential| credential as Arc<dyn TokenCredential>),
        NativeAzureRequest::ManagedIdentity { client_id, .. } => {
            ManagedIdentityCredential::new(Some(ManagedIdentityCredentialOptions {
                user_assigned_id: client_id.map(UserAssignedId::ClientId),
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

    use super::{NativeAzureRequest, NativeAzureTokenAcquirer};
    use crate::auth::secret::SecretValue;

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
        let request = NativeAzureRequest::ClientSecret {
            tenant_id: "tenant".to_string(),
            client_id: "client".to_string(),
            client_secret: SecretValue::new("secret"),
            scope: "https://service.test/.default".to_string(),
            authority: Some("https://login.test".to_string()),
        };

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
        let request = |tenant: &str, client: &str, secret: &str, scope: &str, authority: &str| {
            NativeAzureRequest::ClientSecret {
                tenant_id: tenant.into(),
                client_id: client.into(),
                client_secret: SecretValue::new(secret),
                scope: scope.into(),
                authority: Some(authority.into()),
            }
        };
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
}
