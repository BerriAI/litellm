use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;

use reqwest::header::HeaderName;

use super::OperationFuture;

#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[redacted]")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialProvenance {
    CallerSupplied,
    ForwardedHeader(HeaderName),
    EnvironmentVariable(String),
    ExternalProvider,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenLease {
    pub token: SecretString,
    pub expires_at: Instant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenCredential {
    KnownExpiry(TokenLease),
    NoStore(SecretString),
}

pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

pub trait TokenProvider: Send + Sync {
    fn token(&self) -> OperationFuture<'_, TokenCredential>;
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum AzureCredentialMethod {
    ClientSecret,
    ClientCertificate,
    SystemManagedIdentity,
    UserManagedIdentity,
    WorkloadIdentity,
    ExternalOidc,
    DefaultChain,
    DeveloperCli,
    DeploymentEnvironment,
    UsernamePassword,
    ConfiguredClass,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum GoogleCredentialMethod {
    ServiceAccount,
    AuthorizedUser,
    ApplicationDefault,
    MetadataServer,
    WorkloadIdentityFile,
    WorkloadIdentityUrl,
    WorkloadIdentityExecutable,
    AwsWorkloadIdentity,
    ImpersonatedServiceAccount,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CloudCredentialMethod {
    Azure(AzureCredentialMethod),
    Google(GoogleCredentialMethod),
    CallerToken,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CredentialIdentity {
    pub method: CloudCredentialMethod,
    pub principal: Option<String>,
    pub tenant: Option<String>,
    pub audience: Option<String>,
    pub scopes: Vec<String>,
}

pub struct ExpiryAwareTokenProvider {
    inner: Arc<dyn TokenProvider>,
    clock: Arc<dyn Clock>,
    cached: Mutex<Option<TokenLease>>,
}

impl ExpiryAwareTokenProvider {
    pub fn new(inner: Arc<dyn TokenProvider>, clock: Arc<dyn Clock>) -> Self {
        Self {
            inner,
            clock,
            cached: Mutex::new(None),
        }
    }
}

impl TokenProvider for ExpiryAwareTokenProvider {
    fn token(&self) -> OperationFuture<'_, TokenCredential> {
        Box::pin(async move {
            let mut cached = self.cached.lock().await;
            if let Some(lease) = cached.as_ref()
                && lease.expires_at > self.clock.now()
            {
                return Ok(TokenCredential::KnownExpiry(lease.clone()));
            }
            match self.inner.token().await? {
                TokenCredential::KnownExpiry(lease) if lease.expires_at > self.clock.now() => {
                    *cached = Some(lease.clone());
                    Ok(TokenCredential::KnownExpiry(lease))
                }
                TokenCredential::KnownExpiry(_) => Err(crate::Error::Auth(
                    "credential provider returned an expired token".into(),
                )),
                TokenCredential::NoStore(token) => {
                    *cached = None;
                    Ok(TokenCredential::NoStore(token))
                }
            }
        })
    }
}
