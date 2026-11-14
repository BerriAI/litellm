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

use litellm_auth_types::{Error, ErrorDetail};
use litellm_auth_types::{InputSource, ResolvedCredential, Sourced};
use litellm_secret_value::SecretValue;

use super::credential_provider_cache::{
    AzureCredentialProviderCache, AzureCredentialProviderCacheKey,
};
use super::requests_transport::RequestsTransport;
use super::token_retry::TokenRetry;
use super::types::{AzureTokenRetry, TokenTransport};

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
    probe_imds: bool,
    token_retry: AzureTokenRetry,
    token_transport: TokenTransport,
}

impl ValidatedAzureRequest {
    pub(crate) fn new(request: NativeAzureRequest) -> Result<Self, Error> {
        validate_authority(&request)?;
        let credential_source = validate_sources(&request)?;
        Ok(Self {
            request,
            credential_source,
            probe_imds: false,
            token_retry: AzureTokenRetry::SdkDefault,
            token_transport: TokenTransport::default(),
        })
    }

    pub(crate) fn with_token_retry(self, token_retry: AzureTokenRetry) -> Self {
        Self {
            token_retry,
            ..self
        }
    }

    pub(crate) fn with_token_transport(self, token_transport: TokenTransport) -> Self {
        Self {
            token_transport,
            ..self
        }
    }

    pub(crate) fn with_imds_probe(self, enabled: bool) -> Self {
        Self {
            probe_imds: enabled
                && matches!(self.request, NativeAzureRequest::ManagedIdentity { .. }),
            ..self
        }
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

pub(crate) enum CredentialAttempt {
    Resolved(ResolvedCredential),
    Unavailable(Error),
    Failed(Error),
}

impl CredentialAttempt {
    pub(crate) fn into_result(self) -> Result<ResolvedCredential, Error> {
        match self {
            Self::Resolved(credential) => Ok(credential),
            Self::Unavailable(error) | Self::Failed(error) => Err(error),
        }
    }
}

pub(crate) struct NativeAzureTokenAcquirer {
    cache: AzureCredentialProviderCache,
    transport: Option<azure_core::http::Transport>,
    requests: Arc<RequestsTransport>,
    imds_available: tokio::sync::Mutex<ImdsAvailability>,
}

enum ImdsAvailability {
    Unknown,
    Available,
    Unavailable {
        retry_at: tokio::time::Instant,
        error: Error,
    },
}

impl Default for NativeAzureTokenAcquirer {
    fn default() -> Self {
        Self::new(64)
    }
}

impl NativeAzureTokenAcquirer {
    pub(crate) fn new(cache_capacity: u64) -> Self {
        Self::with_requests_transport(cache_capacity, RequestsTransport::process())
    }

    pub(super) fn with_requests_transport(
        cache_capacity: u64,
        requests: RequestsTransport,
    ) -> Self {
        Self {
            cache: AzureCredentialProviderCache::new(cache_capacity),
            transport: None,
            requests: Arc::new(requests),
            imds_available: tokio::sync::Mutex::new(ImdsAvailability::Unknown),
        }
    }

    #[cfg(test)]
    pub(super) fn with_transport(
        cache_capacity: u64,
        transport: azure_core::http::Transport,
    ) -> Self {
        Self {
            transport: Some(transport),
            ..Self::new(cache_capacity)
        }
    }

    pub(crate) async fn acquire(&self, request: ValidatedAzureRequest) -> CredentialAttempt {
        let discover_imds = request.probe_imds;
        if discover_imds && let Err(error) = self.discover_imds().await {
            return CredentialAttempt::Unavailable(error);
        }
        match self.acquire_token(request).await {
            Ok(credential) => CredentialAttempt::Resolved(credential),
            Err(error) => {
                let missing_identity = discover_imds && unassigned_identity(&error);
                if missing_identity {
                    CredentialAttempt::Unavailable(error)
                } else {
                    CredentialAttempt::Failed(error)
                }
            }
        }
    }

    async fn discover_imds(&self) -> Result<(), Error> {
        let mut availability = self.imds_available.lock().await;
        match &*availability {
            ImdsAvailability::Available => return Ok(()),
            ImdsAvailability::Unavailable { retry_at, error }
                if tokio::time::Instant::now() < *retry_at =>
            {
                return Err(error.clone());
            }
            ImdsAvailability::Unknown | ImdsAvailability::Unavailable { .. } => {}
        }
        let result = probe_imds(self.transport.clone().unwrap_or_default()).await;
        *availability = match &result {
            Ok(()) => ImdsAvailability::Available,
            Err(error) => ImdsAvailability::Unavailable {
                retry_at: tokio::time::Instant::now() + Duration::from_secs(30),
                error: error.clone(),
            },
        };
        result
    }

    async fn acquire_token(
        &self,
        request: ValidatedAzureRequest,
    ) -> Result<ResolvedCredential, Error> {
        let scope = request.request.scope().to_string();
        let key = request.cache_key();
        let transport = self
            .transport
            .clone()
            .or_else(|| match request.token_transport {
                TokenTransport::AzureSdk => None,
                TokenTransport::PythonRequests => Some(azure_core::http::Transport::new(
                    Arc::clone(&self.requests) as Arc<dyn azure_core::http::HttpClient>,
                )),
            });
        let credential = self
            .cache
            .get_or_create(key, async move {
                build_credential(request.request, request.token_retry, transport)
            })
            .await?;
        let token = credential
            .get_token(&[scope.as_str()], None)
            .await
            .map_err(|error| {
                Error::CredentialAcquisition(ErrorDetail::failed("Azure token request", error))
            })?;
        let expires_on = u64::try_from(token.expires_on.unix_timestamp())
            .ok()
            .map(|seconds| UNIX_EPOCH + Duration::from_secs(seconds));

        Ok(ResolvedCredential::AccessToken {
            token: SecretValue::new(token.token.secret()),
            expires_on,
        })
    }
}

fn unassigned_identity(error: &(dyn std::error::Error + 'static)) -> bool {
    std::iter::successors(Some(error), |cause| cause.source()).any(|cause| {
        cause
            .downcast_ref::<azure_core::Error>()
            .or_else(|| {
                cause
                    .downcast_ref::<litellm_error::Source>()?
                    .downcast_ref::<azure_core::Error>()
            })
            .is_some_and(|error| {
                error.http_status() == Some(azure_core::http::StatusCode::BadRequest)
            })
    })
}

async fn probe_imds(transport: azure_core::http::Transport) -> Result<(), Error> {
    use azure_core::http::{Context, Method, Request, Url};

    let endpoint =
        Url::parse("http://169.254.169.254/metadata/identity/oauth2/token?api-version=2019-08-01")
            .map_err(|error| {
                Error::InvalidConfiguration(ErrorDetail::failed("Azure IMDS probe endpoint", error))
            })?;
    let mut request = Request::new(endpoint, Method::Get);
    let response = tokio::time::timeout(
        Duration::from_secs(1),
        transport.send(&Context::new(), &mut request),
    )
    .await
    .map_err(|error| {
        Error::CredentialAcquisition(ErrorDetail::failed("Azure IMDS availability probe", error))
    })?
    .map_err(|error| {
        Error::CredentialAcquisition(ErrorDetail::failed("Azure IMDS availability probe", error))
    })?;
    // Headerless IMDS discovery expects 400, checked 2026-11-01
    // https://learn.microsoft.com/en-us/azure/virtual-machines/instance-metadata-service#errors-and-debugging
    if response.status() != azure_core::http::StatusCode::BadRequest {
        return Err(Error::CredentialAcquisition(
            ErrorDetail::UnexpectedStatus {
                operation: "Azure IMDS availability probe",
                status: u16::from(response.status()),
            },
        ));
    }
    Ok(())
}

impl ValidatedAzureRequest {
    fn cache_key(&self) -> AzureCredentialProviderCacheKey {
        AzureCredentialProviderCacheKey {
            token_transport: self.token_transport,
            ..self.request.cache_key(self.token_retry)
        }
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

    fn cache_key(&self, token_retry: AzureTokenRetry) -> AzureCredentialProviderCacheKey {
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
                token_retry,
                secret_identity: secret_digest(client_secret.value().expose()),
                token_transport: TokenTransport::default(),
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
                token_retry,
                secret_identity: format!(
                    "{assertion_identity}:{}",
                    secret_digest(assertion.value().expose())
                ),
                token_transport: TokenTransport::default(),
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
                token_retry,
                secret_identity: token_file_path.value().clone(),
                token_transport: TokenTransport::default(),
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
                token_retry,
                secret_identity: String::new(),
                token_transport: TokenTransport::default(),
            },
            Self::DeveloperTools { scope, .. } => AzureCredentialProviderCacheKey {
                mechanism: "developer-tools",
                authority: String::new(),
                tenant_id: String::new(),
                client_id: String::new(),
                scope: scope.value().clone(),
                token_retry,
                secret_identity: String::new(),
                token_transport: TokenTransport::default(),
            },
        }
    }
}

fn validate_authority(request: &NativeAzureRequest) -> Result<(), Error> {
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
    let url = url::Url::parse(authority.value()).map_err(|error| {
        Error::InvalidConfiguration(ErrorDetail::failed("Azure authority parsing", error))
    })?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err(Error::InvalidConfiguration(ErrorDetail::InvalidType {
            field: "Azure authority".into(),
            expected: "an HTTPS origin without credentials, query, or fragment",
        }));
    }
    Ok(())
}

fn validate_sources(request: &NativeAzureRequest) -> Result<InputSource, Error> {
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

fn trusted_only(sources: &[InputSource]) -> Result<InputSource, Error> {
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

fn mixed_sources<T>() -> Result<T, Error> {
    Err(Error::InvalidConfiguration(
        ErrorDetail::MixedInputSources { provider: "Azure" },
    ))
}

fn build_credential(
    request: NativeAzureRequest,
    token_retry: AzureTokenRetry,
    transport: Option<azure_core::http::Transport>,
) -> Result<Arc<dyn TokenCredential>, Error> {
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
                client_options: client_options(
                    authority.map(Sourced::into_value),
                    transport,
                    TokenRetry::exchange(token_retry),
                ),
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
                client_options: client_options(
                    authority.map(Sourced::into_value),
                    transport,
                    TokenRetry::assertion(token_retry),
                ),
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
                client_options: client_options(
                    authority.map(Sourced::into_value),
                    transport,
                    TokenRetry::exchange(token_retry),
                ),
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
                client_options: client_options(None, transport, TokenRetry::SdkDefault),
            }))
            .map(|credential| credential as Arc<dyn TokenCredential>)
        }
        NativeAzureRequest::DeveloperTools { .. } => DeveloperToolsCredential::new(None)
            .map(|credential| credential as Arc<dyn TokenCredential>),
    }
    .map_err(|error| {
        Error::InvalidConfiguration(ErrorDetail::failed(
            "Azure credential initialization",
            error,
        ))
    })
}

fn client_options(
    authority: Option<String>,
    transport: Option<azure_core::http::Transport>,
    retry: TokenRetry,
) -> ClientOptions {
    let cloud = authority.map(|authority_host| {
        let mut custom = CustomConfiguration::default();
        custom.authority_host = authority_host;
        Arc::new(CloudConfiguration::from(custom))
    });
    let (retry, per_call_policies) = retry.into_options();
    ClientOptions {
        cloud,
        transport,
        retry,
        per_call_policies,
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
    use std::time::Duration;

    use azure_core::http::headers::Headers;
    use azure_core::http::{AsyncRawResponse, HttpClient, Request, StatusCode, Transport};
    use azure_core::{Bytes, Result};

    use super::{
        AzureTokenRetry, NativeAzureRequest, NativeAzureTokenAcquirer, ValidatedAzureRequest,
    };
    use litellm_auth_types::{InputSource, Sourced};
    use litellm_secret_value::SecretValue;

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

    #[derive(Debug)]
    struct FailingTokenEndpoint {
        status: StatusCode,
        retry_after: Option<&'static str>,
        posts: std::sync::atomic::AtomicUsize,
    }

    impl HttpClient for FailingTokenEndpoint {
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
                assert_eq!(request.method(), azure_core::http::Method::Post);
                self.posts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut headers = Headers::new();
                if let Some(retry_after) = self.retry_after {
                    headers.insert(azure_core::http::headers::RETRY_AFTER, retry_after);
                }
                Ok(AsyncRawResponse::from_bytes(
                    self.status,
                    headers,
                    r#"{"error":"temporarily_unavailable","error_description":"AADSTS90033: A transient error has occurred."}"#,
                ))
            })
        }
    }

    fn client_secret_token_request() -> ValidatedAzureRequest {
        client_secret_request(
            "tenant",
            "client",
            "secret",
            "https://service.test/.default",
            "https://login.test",
        )
    }

    fn workload_identity_token_request() -> ValidatedAzureRequest {
        let token_file = std::env::temp_dir().join(format!(
            "litellm-auth-azure-federated-token-{}",
            std::process::id()
        ));
        std::fs::write(&token_file, "federated-assertion").unwrap();
        ValidatedAzureRequest::new(NativeAzureRequest::WorkloadIdentity {
            tenant_id: deployment("tenant".to_string()),
            client_id: deployment("client".to_string()),
            token_file_path: deployment(token_file.to_string_lossy().into_owned()),
            scope: deployment("https://service.test/.default".to_string()),
            authority: Some(deployment("https://login.test".to_string())),
        })
        .unwrap()
    }

    fn client_assertion_token_request() -> ValidatedAzureRequest {
        ValidatedAzureRequest::new(NativeAzureRequest::ClientAssertion {
            tenant_id: deployment("tenant".to_string()),
            client_id: deployment("client".to_string()),
            assertion: deployment(SecretValue::new("oidc-assertion")),
            assertion_identity: "oidc".to_string(),
            scope: deployment("https://service.test/.default".to_string()),
            authority: Some(deployment("https://login.test".to_string())),
        })
        .unwrap()
    }

    // Python's counts and delays come from azure-core 1.41.0 RetryPolicy defaults (retry_status=3,
    // retry_backoff_factor=0.8, Retry-After honoured) that azure-identity 1.25.2 uses for token POSTs,
    // checked 2026-11-13.
    // OIDC is minted by litellm's own HTTPHandler in Python, which posts once.
    #[rstest::rstest]
    #[case::client_secret_500_retries_three_times_with_backoff(
        client_secret_token_request,
        StatusCode::InternalServerError,
        None,
        4,
        Duration::from_millis(4800),
        "AADSTS90033"
    )]
    #[case::workload_identity_500_retries_three_times_with_backoff(
        workload_identity_token_request,
        StatusCode::InternalServerError,
        None,
        4,
        Duration::from_millis(4800),
        "AADSTS90033"
    )]
    #[case::client_secret_503_waits_retry_after_instead_of_backoff(
        client_secret_token_request,
        StatusCode::ServiceUnavailable,
        Some("1"),
        4,
        Duration::from_secs(3),
        "AADSTS90033"
    )]
    #[case::client_secret_429_zero_retry_after_keeps_backoff(
        client_secret_token_request,
        StatusCode::TooManyRequests,
        Some("0"),
        4,
        Duration::from_millis(4800),
        "AADSTS90033"
    )]
    #[case::client_secret_401_invalid_client_posts_once(
        client_secret_token_request,
        StatusCode::Unauthorized,
        None,
        1,
        Duration::ZERO,
        "AADSTS90033"
    )]
    #[case::client_secret_500_waits_retry_after(
        client_secret_token_request,
        StatusCode::InternalServerError,
        Some("2"),
        4,
        Duration::from_secs(6),
        "AADSTS90033"
    )]
    #[case::client_secret_401_with_retry_after_is_retried(
        client_secret_token_request,
        StatusCode::Unauthorized,
        Some("2"),
        4,
        Duration::from_secs(6),
        "AADSTS90033"
    )]
    #[case::client_secret_malformed_retry_after_fails_after_one_post(
        client_secret_token_request,
        StatusCode::InternalServerError,
        Some("soon"),
        1,
        Duration::ZERO,
        "Invalid HTTP date"
    )]
    #[case::client_secret_infinite_retry_after_fails_after_one_post(
        client_secret_token_request,
        StatusCode::ServiceUnavailable,
        Some("inf"),
        1,
        Duration::ZERO,
        "timestamp out of range for platform time_t"
    )]
    #[case::client_assertion_503_posts_once(
        client_assertion_token_request,
        StatusCode::ServiceUnavailable,
        None,
        1,
        Duration::ZERO,
        "AADSTS90033"
    )]
    #[case::client_secret_400_posts_once(
        client_secret_token_request,
        StatusCode::BadRequest,
        None,
        1,
        Duration::ZERO,
        "AADSTS90033"
    )]
    #[tokio::test(start_paused = true)]
    async fn entra_token_endpoint_failures_retry_like_python(
        #[case] request: fn() -> ValidatedAzureRequest,
        #[case] status: StatusCode,
        #[case] retry_after: Option<&'static str>,
        #[case] expected_posts: usize,
        #[case] expected_wait: Duration,
        #[case] expected_reason: &str,
    ) {
        let endpoint = Arc::new(FailingTokenEndpoint {
            status,
            retry_after,
            posts: std::sync::atomic::AtomicUsize::new(0),
        });
        let acquirer =
            NativeAzureTokenAcquirer::with_transport(4, Transport::new(endpoint.clone()));
        let started = tokio::time::Instant::now();

        let error = acquirer
            .acquire(request().with_token_retry(AzureTokenRetry::Python))
            .await
            .into_result()
            .unwrap_err();

        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains(expected_reason),
            "{error:?}"
        );
        assert_eq!(
            endpoint.posts.load(std::sync::atomic::Ordering::SeqCst),
            expected_posts
        );
        assert_eq!(started.elapsed(), expected_wait);
    }

    #[rstest::rstest]
    #[case::exhausted_server_error(StatusCode::InternalServerError, None, Some(500))]
    #[case::rejection(StatusCode::Unauthorized, None, Some(401))]
    #[case::malformed_retry_after(StatusCode::InternalServerError, Some("soon"), None)]
    #[tokio::test(start_paused = true)]
    async fn a_failed_token_post_keeps_what_python_reports_from_it(
        #[case] status: StatusCode,
        #[case] retry_after: Option<&'static str>,
        #[case] answered: Option<u16>,
    ) {
        let endpoint = Arc::new(FailingTokenEndpoint {
            status,
            retry_after,
            posts: std::sync::atomic::AtomicUsize::new(0),
        });
        let acquirer = NativeAzureTokenAcquirer::with_transport(4, Transport::new(endpoint));

        let error = acquirer
            .acquire(client_secret_token_request().with_token_retry(AzureTokenRetry::Python))
            .await
            .into_result()
            .unwrap_err();

        match (
            litellm_error::find::<crate::EntraTokenFailure>(&error),
            answered,
        ) {
            (Some(crate::EntraTokenFailure::Response(response)), Some(expected)) => {
                assert_eq!(response.status, expected);
                assert!(
                    String::from_utf8_lossy(&response.body).contains("AADSTS90033"),
                    "{response:?}"
                );
            }
            (Some(crate::EntraTokenFailure::InvalidRetryAfter(response)), None) => {
                assert_eq!(
                    response
                        .headers
                        .iter()
                        .find(|(name, _)| name == "retry-after"),
                    Some(&("retry-after".to_string(), "soon".to_string()))
                );
            }
            (failure, _) => panic!("{failure:?} from {error:?}"),
        }
    }

    fn key_vault_style_inputs(token_retry: AzureTokenRetry) -> crate::AzureAuthInputs {
        let configured = |value: &str| crate::ConfigValue::Value(deployment(value.to_string()));
        crate::AzureAuthInputs {
            tenant_id: configured("tenant"),
            client_id: configured("client"),
            client_secret: crate::ConfigValue::Value(deployment(SecretValue::new("secret"))),
            azure_scope: configured("https://service.test/.default"),
            azure_authority_host: configured("https://login.test"),
            token_retry,
            ..crate::AzureAuthInputs::default()
        }
    }

    #[rstest::rstest]
    #[case::other_consumers_keep_the_sdk_default(AzureTokenRetry::SdkDefault, false)]
    #[case::a_consumer_asking_for_python_gets_azure_identity_retries(AzureTokenRetry::Python, true)]
    #[tokio::test(start_paused = true)]
    async fn only_a_consumer_that_asks_retries_token_posts_like_python(
        #[case] token_retry: AzureTokenRetry,
        #[case] python: bool,
    ) {
        let endpoint = Arc::new(FailingTokenEndpoint {
            status: StatusCode::Unauthorized,
            retry_after: Some("2"),
            posts: std::sync::atomic::AtomicUsize::new(0),
        });
        let service = crate::AzureAuthService::with_acquirer(Arc::new(
            NativeAzureTokenAcquirer::with_transport(4, Transport::new(endpoint.clone())),
        ));

        let error = service
            .get_azure_ad_token(&key_vault_style_inputs(token_retry), &|_| None)
            .await
            .unwrap_err();

        let posts = endpoint.posts.load(std::sync::atomic::Ordering::SeqCst);
        let classified = litellm_error::find::<crate::EntraTokenFailure>(&error).is_some();
        assert_eq!(
            (classified, posts == 4),
            (python, python),
            "{posts} posts, {error:?}"
        );
    }

    #[derive(Debug)]
    struct AvailableImdsClient {
        probe_status: StatusCode,
        probe_delay: Duration,
        recover_after: Option<usize>,
        response_status: Option<StatusCode>,
        probes: std::sync::atomic::AtomicUsize,
    }

    impl Default for AvailableImdsClient {
        fn default() -> Self {
            Self {
                probe_status: StatusCode::BadRequest,
                probe_delay: Duration::ZERO,
                recover_after: None,
                response_status: None,
                probes: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl HttpClient for AvailableImdsClient {
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
                if request
                    .headers()
                    .get_optional_str(&azure_core::http::headers::HeaderName::from_static(
                        "metadata",
                    ))
                    .is_none()
                {
                    let probe = self
                        .probes
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if self.recover_after.is_some_and(|failures| probe >= failures) {
                        return Ok(AsyncRawResponse::from_bytes(
                            StatusCode::BadRequest,
                            Headers::new(),
                            "metadata header required",
                        ));
                    }
                    tokio::time::sleep(self.probe_delay).await;
                    return Ok(AsyncRawResponse::from_bytes(
                        self.probe_status,
                        Headers::new(),
                        "metadata header required",
                    ));
                }
                if let Some(status) = self.response_status {
                    return Ok(AsyncRawResponse::from_bytes(
                        status,
                        Headers::new(),
                        "identity endpoint failure",
                    ));
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let expires = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + 3600;
                Ok(AsyncRawResponse::from_bytes(
                    StatusCode::Ok,
                    Headers::new(),
                    serde_json::json!({
                        "access_token": "managed-token", "expires_on": expires.to_string(),
                        "token_type": "Bearer", "resource": "https://service.test",
                    })
                    .to_string(),
                ))
            })
        }
    }

    #[rstest::rstest]
    #[tokio::test(start_paused = true)]
    async fn available_imds_retains_full_sdk_token_acquisition() {
        let acquirer = NativeAzureTokenAcquirer::with_transport(
            4,
            Transport::new(Arc::new(AvailableImdsClient {
                probe_status: StatusCode::BadRequest,
                response_status: None,
                ..Default::default()
            })),
        );
        let request = ValidatedAzureRequest::new(NativeAzureRequest::ManagedIdentity {
            client_id: None,
            scope: deployment("https://service.test/.default".into()),
            selection_source: InputSource::Deployment,
        })
        .unwrap()
        .with_imds_probe(true);
        let credential = acquirer.acquire(request).await.into_result().unwrap();
        assert_eq!(credential.secret().expose(), "managed-token");
    }

    #[rstest::rstest]
    #[case::another_metadata_service(StatusCode::NotFound)]
    #[case::token_required(StatusCode::Unauthorized)]
    #[case::proxy_rejection(StatusCode::Forbidden)]
    #[case::unavailable(StatusCode::ServiceUnavailable)]
    #[tokio::test(start_paused = true)]
    async fn unrelated_metadata_responses_do_not_start_managed_identity_acquisition(
        #[case] status: StatusCode,
    ) {
        let client = Arc::new(AvailableImdsClient {
            probe_status: status,
            ..Default::default()
        });
        let acquirer = NativeAzureTokenAcquirer::with_transport(4, Transport::new(client.clone()));
        let request = ValidatedAzureRequest::new(NativeAzureRequest::ManagedIdentity {
            client_id: None,
            scope: deployment("https://service.test/.default".into()),
            selection_source: InputSource::Deployment,
        })
        .unwrap()
        .with_imds_probe(true);
        assert!(matches!(
            acquirer.acquire(request.clone()).await,
            super::CredentialAttempt::Unavailable(_)
        ));
        assert!(matches!(
            acquirer.acquire(request).await,
            super::CredentialAttempt::Unavailable(_)
        ));
        assert_eq!(client.probes.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[rstest::rstest]
    #[tokio::test(start_paused = true)]
    async fn an_imds_timeout_is_not_repeated_on_the_next_request() {
        let client = Arc::new(AvailableImdsClient {
            probe_delay: Duration::from_secs(10),
            ..Default::default()
        });
        let acquirer = NativeAzureTokenAcquirer::with_transport(4, Transport::new(client.clone()));
        let request = ValidatedAzureRequest::new(NativeAzureRequest::ManagedIdentity {
            client_id: None,
            scope: deployment("https://service.test/.default".into()),
            selection_source: InputSource::Deployment,
        })
        .unwrap()
        .with_imds_probe(true);
        assert!(matches!(
            acquirer.acquire(request.clone()).await,
            super::CredentialAttempt::Unavailable(_)
        ));
        let second = tokio::time::timeout(Duration::from_millis(10), acquirer.acquire(request))
            .await
            .unwrap();
        assert!(matches!(second, super::CredentialAttempt::Unavailable(_)));
        assert_eq!(client.probes.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[rstest::fixture]
    fn imds_request() -> ValidatedAzureRequest {
        ValidatedAzureRequest::new(NativeAzureRequest::ManagedIdentity {
            client_id: None,
            scope: deployment("https://service.test/.default".into()),
            selection_source: InputSource::Deployment,
        })
        .unwrap()
        .with_imds_probe(true)
    }

    #[rstest::rstest]
    #[case::timeout(StatusCode::BadRequest, Duration::from_secs(10))]
    #[case::service_unavailable(StatusCode::ServiceUnavailable, Duration::ZERO)]
    #[tokio::test(start_paused = true)]
    async fn failed_imds_discovery_recovers_without_restarting_the_acquirer(
        imds_request: ValidatedAzureRequest,
        #[case] status: StatusCode,
        #[case] delay: Duration,
    ) {
        let client = Arc::new(AvailableImdsClient {
            probe_status: status,
            probe_delay: delay,
            recover_after: Some(1),
            ..Default::default()
        });
        let acquirer = NativeAzureTokenAcquirer::with_transport(4, Transport::new(client.clone()));
        let (first, second) = tokio::join!(
            acquirer.acquire(imds_request.clone()),
            acquirer.acquire(imds_request.clone()),
        );
        assert!(matches!(first, super::CredentialAttempt::Unavailable(_)));
        assert!(matches!(second, super::CredentialAttempt::Unavailable(_)));
        assert_eq!(client.probes.load(std::sync::atomic::Ordering::SeqCst), 1);

        tokio::time::advance(Duration::from_secs(30)).await;
        let (first, second) = tokio::join!(
            acquirer.acquire(imds_request.clone()),
            acquirer.acquire(imds_request.clone()),
        );
        assert_eq!(
            first.into_result().unwrap().secret().expose(),
            "managed-token"
        );
        assert_eq!(
            second.into_result().unwrap().secret().expose(),
            "managed-token"
        );
        assert_eq!(client.probes.load(std::sync::atomic::Ordering::SeqCst), 2);

        tokio::time::advance(Duration::from_secs(60)).await;
        assert_eq!(
            acquirer
                .acquire(imds_request)
                .await
                .into_result()
                .unwrap()
                .secret()
                .expose(),
            "managed-token",
        );
        assert_eq!(client.probes.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[rstest::rstest]
    #[tokio::test(start_paused = true)]
    async fn a_persistent_imds_outage_reuses_each_failed_probe(
        imds_request: ValidatedAzureRequest,
    ) {
        let client = Arc::new(AvailableImdsClient {
            probe_delay: Duration::from_secs(10),
            ..Default::default()
        });
        let acquirer = NativeAzureTokenAcquirer::with_transport(4, Transport::new(client.clone()));
        assert!(matches!(
            acquirer.acquire(imds_request.clone()).await,
            super::CredentialAttempt::Unavailable(_),
        ));
        tokio::time::advance(Duration::from_secs(30)).await;
        let (first, second) = tokio::join!(
            acquirer.acquire(imds_request.clone()),
            acquirer.acquire(imds_request),
        );
        assert!(matches!(first, super::CredentialAttempt::Unavailable(_)));
        assert!(matches!(second, super::CredentialAttempt::Unavailable(_)));
        assert_eq!(client.probes.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[rstest::rstest]
    #[case::default_identity_not_assigned(true, StatusCode::BadRequest, true)]
    #[case::explicit_identity_not_assigned(false, StatusCode::BadRequest, false)]
    #[case::default_identity_forbidden(true, StatusCode::Forbidden, false)]
    #[case::default_identity_service_failure(true, StatusCode::InternalServerError, false)]
    #[tokio::test(start_paused = true)]
    async fn only_unavailable_default_identities_allow_fallback(
        #[case] discovery: bool,
        #[case] status: StatusCode,
        #[case] unavailable: bool,
    ) {
        let acquirer = NativeAzureTokenAcquirer::with_transport(
            4,
            Transport::new(Arc::new(AvailableImdsClient {
                probe_status: StatusCode::BadRequest,
                response_status: Some(status),
                ..Default::default()
            })),
        );
        let request = ValidatedAzureRequest::new(NativeAzureRequest::ManagedIdentity {
            client_id: None,
            scope: deployment("https://service.test/.default".into()),
            selection_source: InputSource::Deployment,
        })
        .unwrap()
        .with_imds_probe(discovery);
        let outcome = acquirer.acquire(request).await;
        let is_unavailable = matches!(outcome, super::CredentialAttempt::Unavailable(_));
        let error = outcome.into_result().unwrap_err();
        assert_eq!(
            is_unavailable,
            unavailable,
            "{}",
            litellm_error::Report::new(&error)
        );
    }

    #[rstest::rstest]
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

        let first = acquirer
            .acquire(request.clone())
            .await
            .into_result()
            .unwrap();
        let second = acquirer.acquire(request).await.into_result().unwrap();

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

    #[rstest::rstest]
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

        acquirer.acquire(base.clone()).await.into_result().unwrap();
        acquirer.acquire(base).await.into_result().unwrap();
        for request in variants.into_iter().skip(1) {
            acquirer.acquire(request).await.into_result().unwrap();
        }

        assert_eq!(transport.requests.lock().unwrap().len(), 6);
    }

    #[rstest::rstest]
    fn request_authority_requires_request_owned_client_secret_identity() {
        let error = ValidatedAzureRequest::new(sourced_client_secret(
            InputSource::Deployment,
            InputSource::Request,
            "https://login.example",
        ))
        .unwrap_err();

        assert!(
            litellm_error::Report::new(&error).to_string().contains(
                "request-controlled Azure auth inputs cannot be combined with host credentials"
            ),
            "{error:?}"
        );
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

    #[rstest::rstest]
    #[case::http("http://login.example")]
    #[case::userinfo("https://user@login.example")]
    #[case::path("https://login.example/tenant")]
    #[case::query("https://login.example?target=other")]
    fn authority_is_restricted_to_an_https_origin(#[case] authority: &str) {
        let error = ValidatedAzureRequest::new(sourced_client_secret(
            InputSource::Deployment,
            InputSource::Deployment,
            authority,
        ))
        .unwrap_err();
        assert!(
            litellm_error::Report::new(&error).to_string().contains(
                "Azure authority must be an HTTPS origin without credentials, query, or fragment"
            ),
            "{error:?}"
        );
    }

    #[rstest::rstest]
    #[tokio::test(start_paused = true)]
    #[expect(
        clippy::disallowed_methods,
        clippy::disallowed_types,
        reason = "pins what azure_core's own reqwest transport reports, without the sandbox's proxy"
    )]
    async fn a_refused_token_endpoint_keeps_its_host_port_and_os_error() {
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let acquirer =
            NativeAzureTokenAcquirer::with_transport(4, Transport::new(Arc::new(client)));
        let request = client_secret_request(
            "tenant",
            "client",
            "secret",
            "https://service.test/.default",
            "https://127.0.0.1:1",
        )
        .with_token_retry(AzureTokenRetry::Python);

        let error = acquirer.acquire(request).await.into_result().unwrap_err();

        let Some(crate::EntraTokenFailure::Transport { host, port, source }) =
            litellm_error::find::<crate::EntraTokenFailure>(&error)
        else {
            panic!("{error:?}");
        };
        assert_eq!((host.as_str(), *port), ("127.0.0.1", 1));
        let errno = std::iter::successors(
            Some(source as &(dyn std::error::Error + 'static)),
            |cause| cause.source(),
        )
        .find_map(|cause| cause.downcast_ref::<std::io::Error>()?.raw_os_error());
        assert_eq!(errno, Some(111));
    }
}
