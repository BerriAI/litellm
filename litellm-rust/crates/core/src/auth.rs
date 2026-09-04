use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Method, Url};
use sha2::{Digest, Sha256};
use thiserror::Error as ThisError;
use tokio::sync::{Mutex, RwLock};

use crate::Error;

pub trait Environment: Send + Sync {
    fn get(&self, name: &str) -> Option<String>;
}

impl<F> Environment for F
where
    F: Fn(&str) -> Option<String> + Send + Sync,
{
    fn get(&self, name: &str) -> Option<String> {
        self(name)
    }
}

pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
}

#[derive(Debug)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthErrorKind {
    MissingCredential,
    InvalidConfiguration,
    CredentialUnavailable,
    ExternalProviderFailed,
    InvalidHeader,
    SigningFailed,
    UnsupportedMode,
    CrossOriginReplay,
}

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
#[error("{message}")]
pub struct AuthError {
    pub kind: AuthErrorKind,
    pub code: &'static str,
    message: String,
}

impl AuthError {
    pub fn new(kind: AuthErrorKind, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            code,
            message: message.into(),
        }
    }
}

impl From<AuthError> for Error {
    fn from(error: AuthError) -> Self {
        match error.kind {
            AuthErrorKind::UnsupportedMode => Error::Unsupported(error.code),
            AuthErrorKind::CrossOriginReplay => Error::InvalidResponse(error.to_string()),
            _ => Error::Auth(error.to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenLease {
    pub token: SecretString,
    pub expires_at: Instant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenCredential {
    Cached(TokenLease),
    NoStore(SecretString),
}

#[async_trait]
pub trait TokenProvider: Send + Sync {
    async fn token(&self) -> Result<TokenCredential, AuthError>;
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TokenCacheKey([u8; 32]);

impl TokenCacheKey {
    pub fn fingerprint<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> Self {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update(part.len().to_le_bytes());
            hasher.update(part);
        }
        Self(hasher.finalize().into())
    }
}

impl fmt::Debug for TokenCacheKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TokenCacheKey([redacted])")
    }
}

struct TokenEntry {
    lease: RwLock<Option<TokenLease>>,
    refresh: Arc<Mutex<()>>,
}

impl TokenEntry {
    fn new() -> Self {
        Self {
            lease: RwLock::new(None),
            refresh: Arc::new(Mutex::new(())),
        }
    }
}

pub struct TokenCache {
    entries: Mutex<HashMap<TokenCacheKey, Arc<TokenEntry>>>,
    refresh_before: Duration,
    clock: Arc<dyn Clock>,
}

impl fmt::Debug for TokenCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TokenCache")
            .field("refresh_before", &self.refresh_before)
            .finish_non_exhaustive()
    }
}

impl TokenCache {
    pub fn new(refresh_before: Duration, clock: Arc<dyn Clock>) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            refresh_before,
            clock,
        }
    }

    async fn entry(&self, key: TokenCacheKey) -> Arc<TokenEntry> {
        let mut entries = self.entries.lock().await;
        entries
            .entry(key)
            .or_insert_with(|| Arc::new(TokenEntry::new()))
            .clone()
    }

    pub async fn token(
        self: &Arc<Self>,
        key: TokenCacheKey,
        provider: Arc<dyn TokenProvider>,
    ) -> Result<SecretString, AuthError> {
        let entry = self.entry(key).await;
        let now = self.clock.now();
        let cached = entry.lease.read().await.clone();
        if let Some(lease) = cached.as_ref() {
            if lease.expires_at.saturating_duration_since(now) > self.refresh_before {
                return Ok(lease.token.clone());
            }
            if lease.expires_at > now {
                self.refresh_stale(entry, provider);
                return Ok(lease.token.clone());
            }
        }
        self.refresh_invalid(entry, provider).await
    }

    fn refresh_stale(self: &Arc<Self>, entry: Arc<TokenEntry>, provider: Arc<dyn TokenProvider>) {
        let Ok(guard) = entry.refresh.clone().try_lock_owned() else {
            return;
        };
        let cache = self.clone();
        tokio::spawn(async move {
            let _guard = guard;
            if let Ok(TokenCredential::Cached(lease)) = provider.token().await
                && lease.expires_at > cache.clock.now()
            {
                *entry.lease.write().await = Some(lease);
            }
        });
    }

    async fn refresh_invalid(
        &self,
        entry: Arc<TokenEntry>,
        provider: Arc<dyn TokenProvider>,
    ) -> Result<SecretString, AuthError> {
        let _guard = entry.refresh.lock().await;
        let now = self.clock.now();
        if let Some(lease) = entry.lease.read().await.as_ref()
            && lease.expires_at > now
        {
            return Ok(lease.token.clone());
        }
        let lease = match provider.token().await? {
            TokenCredential::Cached(lease) => lease,
            TokenCredential::NoStore(token) => return Ok(token),
        };
        if lease.expires_at <= self.clock.now() {
            return Err(AuthError::new(
                AuthErrorKind::CredentialUnavailable,
                "expired_token",
                "credential provider returned an expired token",
            ));
        }
        let token = lease.token.clone();
        *entry.lease.write().await = Some(lease);
        Ok(token)
    }

    pub async fn invalidate(&self, key: &TokenCacheKey) {
        if let Some(entry) = self.entries.lock().await.get(key).cloned() {
            *entry.lease.write().await = None;
        }
    }
}

#[derive(Clone)]
pub struct AuthRuntime {
    pub environment: Arc<dyn Environment>,
    pub clock: Arc<dyn Clock>,
    pub tokens: Arc<TokenCache>,
}

pub struct AuthInput<'a> {
    pub explicit_api_key: Option<&'a str>,
    pub forwarded_headers: &'a HeaderMap,
    pub external_token_provider: Option<Arc<dyn TokenProvider>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayPolicy {
    Never,
    SameOrigin,
}

pub struct AuthorizeRequest<'a> {
    pub method: &'a Method,
    pub url: &'a Url,
    pub headers: HeaderMap,
    pub serialized_body: Option<&'a [u8]>,
}

#[async_trait]
pub trait RequestAuthorizer: Send + Sync {
    async fn authorize(&self, request: AuthorizeRequest<'_>) -> Result<HeaderMap, AuthError>;
}

#[async_trait]
pub trait Authenticator: Send + Sync {
    async fn authenticate(
        &self,
        input: AuthInput<'_>,
        runtime: &AuthRuntime,
    ) -> Result<AuthSession, AuthError>;
}

pub trait AuthenticatedProvider: Sync {
    fn authenticator(&self) -> &dyn Authenticator;
}

pub struct AuthSession {
    authorizer: Arc<dyn RequestAuthorizer>,
    replay: ReplayPolicy,
    origin: Option<(String, String, u16)>,
}

impl AuthSession {
    pub fn new(authorizer: Arc<dyn RequestAuthorizer>, replay: ReplayPolicy) -> Self {
        Self {
            authorizer,
            replay,
            origin: None,
        }
    }

    pub fn bind(mut self, url: &Url) -> Self {
        self.origin = request_origin(url);
        self
    }

    pub async fn authorize_primary(
        &self,
        request: AuthorizeRequest<'_>,
    ) -> Result<HeaderMap, AuthError> {
        self.authorizer.authorize(request).await
    }

    pub async fn authorize_follow_up(
        &self,
        request: AuthorizeRequest<'_>,
    ) -> Result<HeaderMap, AuthError> {
        if self.replay != ReplayPolicy::SameOrigin || self.origin != request_origin(request.url) {
            return Err(AuthError::new(
                AuthErrorKind::CrossOriginReplay,
                "cross_origin_auth_replay",
                "refusing to replay provider credentials to another origin",
            ));
        }
        self.authorizer.authorize(request).await
    }
}

fn request_origin(url: &Url) -> Option<(String, String, u16)> {
    Some((
        url.scheme().to_string(),
        url.host_str()?.to_ascii_lowercase(),
        url.port_or_known_default()?,
    ))
}

pub struct StaticHeaderAuthorizer {
    name: HeaderName,
    value: SecretString,
    conflicts: Vec<HeaderName>,
}

impl StaticHeaderAuthorizer {
    pub fn new(name: HeaderName, value: SecretString, conflicts: Vec<HeaderName>) -> Self {
        Self {
            name,
            value,
            conflicts,
        }
    }
}

#[async_trait]
impl RequestAuthorizer for StaticHeaderAuthorizer {
    async fn authorize(&self, request: AuthorizeRequest<'_>) -> Result<HeaderMap, AuthError> {
        let mut headers = request.headers;
        for name in &self.conflicts {
            headers.remove(name);
        }
        let mut value = HeaderValue::from_str(self.value.expose()).map_err(|_| {
            AuthError::new(
                AuthErrorKind::InvalidHeader,
                "invalid_credential_header",
                "credential contains bytes that cannot be used in an HTTP header",
            )
        })?;
        value.set_sensitive(true);
        headers.insert(self.name.clone(), value);
        Ok(headers)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct TestClock(StdMutex<Instant>);

    impl Clock for TestClock {
        fn now(&self) -> Instant {
            *self.0.lock().unwrap()
        }
    }

    struct CountingProvider {
        calls: AtomicUsize,
        clock: Arc<TestClock>,
    }

    #[async_trait]
    impl TokenProvider for CountingProvider {
        async fn token(&self) -> Result<TokenCredential, AuthError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(TokenCredential::Cached(TokenLease {
                token: SecretString::new(format!("token-{call}")),
                expires_at: self.clock.now() + Duration::from_secs(60),
            }))
        }
    }

    struct NoStoreProvider(AtomicUsize);

    #[async_trait]
    impl TokenProvider for NoStoreProvider {
        async fn token(&self) -> Result<TokenCredential, AuthError> {
            let call = self.0.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(TokenCredential::NoStore(SecretString::new(format!(
                "token-{call}"
            ))))
        }
    }

    #[tokio::test]
    async fn invalid_requests_share_one_refresh() {
        let clock = Arc::new(TestClock(StdMutex::new(Instant::now())));
        let cache = Arc::new(TokenCache::new(Duration::from_secs(10), clock.clone()));
        let provider = Arc::new(CountingProvider {
            calls: AtomicUsize::new(0),
            clock,
        });
        let key = TokenCacheKey::fingerprint([b"provider".as_slice(), b"principal".as_slice()]);
        let requests = (0..8).map(|_| {
            let cache = cache.clone();
            let provider = provider.clone();
            let key = key.clone();
            tokio::spawn(async move { cache.token(key, provider).await.unwrap() })
        });
        for request in requests {
            assert_eq!(request.await.unwrap().expose(), "token-1");
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn no_store_credentials_are_resolved_for_each_request() {
        let clock = Arc::new(TestClock(StdMutex::new(Instant::now())));
        let cache = Arc::new(TokenCache::new(Duration::from_secs(10), clock));
        let provider = Arc::new(NoStoreProvider(AtomicUsize::new(0)));
        let key = TokenCacheKey::fingerprint([b"provider".as_slice()]);

        assert_eq!(
            cache
                .token(key.clone(), provider.clone())
                .await
                .unwrap()
                .expose(),
            "token-1"
        );
        assert_eq!(
            cache.token(key, provider.clone()).await.unwrap().expose(),
            "token-2"
        );
        assert_eq!(provider.0.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn debug_output_redacts_secrets_and_cache_keys() {
        assert_eq!(format!("{:?}", SecretString::new("secret")), "[redacted]");
        let key = TokenCacheKey::fingerprint([b"secret".as_slice()]);
        assert!(!format!("{key:?}").contains("secret"));
    }
}
