use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use reqwest::Request;
use reqwest::header::{HeaderName, HeaderValue};

use super::{AuthError, AuthErrorKind, AuthFuture, AuthPreparation, Clock, RequestAuthorizer};

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
pub enum AuthHeaderKind {
    Bearer,
    Header(&'static str),
}

impl AuthHeaderKind {
    pub fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "authorization",
            Self::Header(name) => name,
        }
    }

    pub fn header_value(self, secret: &SecretString) -> SecretString {
        match self {
            Self::Bearer => SecretString::new(format!("Bearer {}", secret.expose())),
            Self::Header(_) => secret.clone(),
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

pub trait TokenProvider: Send + Sync {
    fn token(&self) -> AuthFuture<'_, TokenCredential>;
}

#[derive(Clone)]
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

    fn preparation(&self) -> Result<AuthPreparation, AuthError> {
        let mut value = HeaderValue::from_str(self.value.expose()).map_err(|_| {
            AuthError::new(
                AuthErrorKind::InvalidHeader,
                "invalid_credential_header",
                "credential cannot be used in an HTTP header",
            )
        })?;
        value.set_sensitive(true);
        Ok(AuthPreparation {
            authorizer: Arc::new(self.clone()),
            headers: [(self.name.clone(), value)].into_iter().collect(),
            remove_headers: self.conflicts.clone(),
        })
    }

    fn apply(&self, mut request: Request) -> Result<Request, AuthError> {
        let mut value = HeaderValue::from_str(self.value.expose()).map_err(|_| {
            AuthError::new(
                AuthErrorKind::InvalidHeader,
                "invalid_credential_header",
                "credential cannot be used in an HTTP header",
            )
        })?;
        value.set_sensitive(true);
        for name in &self.conflicts {
            request.headers_mut().remove(name);
        }
        request.headers_mut().insert(self.name.clone(), value);
        Ok(request)
    }
}

impl RequestAuthorizer for StaticHeaderAuthorizer {
    fn prepare(&self) -> AuthFuture<'_, Option<AuthPreparation>> {
        Box::pin(async move { self.preparation().map(Some) })
    }

    fn authorize(&self, request: Request) -> AuthFuture<'_, Request> {
        Box::pin(async move { self.apply(request) })
    }
}

pub struct BearerTokenAuthorizer {
    provider: Arc<dyn TokenProvider>,
    clock: Arc<dyn Clock>,
    conflicts: Vec<HeaderName>,
}

impl BearerTokenAuthorizer {
    pub fn new(
        provider: Arc<dyn TokenProvider>,
        clock: Arc<dyn Clock>,
        conflicts: Vec<HeaderName>,
    ) -> Self {
        Self {
            provider,
            clock,
            conflicts,
        }
    }
}

impl BearerTokenAuthorizer {
    fn credential(&self) -> AuthFuture<'_, StaticHeaderAuthorizer> {
        Box::pin(async move {
            let token = match self.provider.token().await? {
                TokenCredential::Cached(lease) if lease.expires_at > self.clock.now() => {
                    lease.token
                }
                TokenCredential::Cached(_) => {
                    return Err(AuthError::new(
                        AuthErrorKind::CredentialUnavailable,
                        "expired_token",
                        "credential provider returned an expired token",
                    ));
                }
                TokenCredential::NoStore(token) => token,
            };
            if token.expose().trim().is_empty() {
                return Err(AuthError::new(
                    AuthErrorKind::CredentialUnavailable,
                    "empty_token",
                    "credential provider returned an empty token",
                ));
            }
            Ok(StaticHeaderAuthorizer::new(
                reqwest::header::AUTHORIZATION,
                AuthHeaderKind::Bearer.header_value(&token),
                self.conflicts.clone(),
            ))
        })
    }
}

impl RequestAuthorizer for BearerTokenAuthorizer {
    fn prepare(&self) -> AuthFuture<'_, Option<AuthPreparation>> {
        Box::pin(async move { self.credential().await?.preparation().map(Some) })
    }

    fn authorize(&self, request: Request) -> AuthFuture<'_, Request> {
        Box::pin(async move { self.credential().await?.apply(request) })
    }
}
