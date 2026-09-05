mod session;
mod token;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;

pub use session::{
    AuthHttpClient, AuthOperation, AuthPreparation, AuthSession, ReplayPolicy, RequestAuthorizer,
};
pub use token::{
    AuthHeaderKind, BearerTokenAuthorizer, SecretString, StaticHeaderAuthorizer, TokenCredential,
    TokenLease, TokenProvider,
};

pub type AuthFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, AuthError>> + Send + 'a>>;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct AuthError {
    pub kind: AuthErrorKind,
    pub code: &'static str,
    message: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthErrorKind {
    InvalidConfiguration,
    CredentialUnavailable,
    InvalidHeader,
    SigningFailed,
    ForbiddenDestination,
    DeadlineExceeded,
}

impl AuthError {
    pub const fn new(kind: AuthErrorKind, code: &'static str, message: &'static str) -> Self {
        Self {
            kind,
            code,
            message,
        }
    }
}

impl From<AuthError> for crate::Error {
    fn from(error: AuthError) -> Self {
        match error.kind {
            AuthErrorKind::ForbiddenDestination => Self::InvalidResponse(error.to_string()),
            AuthErrorKind::DeadlineExceeded => Self::Network(error.to_string()),
            _ => Self::Auth(error.to_string()),
        }
    }
}

pub enum AuthPreflight<T> {
    Ready(T),
    Declined(&'static str),
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

pub struct AuthRuntime {
    pub http: AuthHttpClient,
    pub clock: Arc<dyn Clock>,
    pub credentials: Arc<dyn CredentialResolver>,
}

pub struct AuthBinding {
    authorizer: Arc<dyn RequestAuthorizer>,
}

impl AuthBinding {
    pub fn new(authorizer: Arc<dyn RequestAuthorizer>) -> Self {
        Self { authorizer }
    }

    pub fn bind(self, url: &reqwest::Url, replay: ReplayPolicy) -> Result<AuthSession, AuthError> {
        AuthSession::new(self.authorizer, url, replay)
    }
}

pub trait AuthAdapter: Send + Sync {
    type Config: Send;

    fn build(&self, config: Self::Config, runtime: Arc<AuthRuntime>)
    -> AuthFuture<'_, AuthBinding>;
}

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub enum CredentialSpec {
    Header {
        name: reqwest::header::HeaderName,
        value: SecretString,
    },
}

pub trait CredentialResolver: Send + Sync {
    fn resolve(
        &self,
        credential: CredentialSpec,
        runtime: Arc<AuthRuntime>,
    ) -> AuthFuture<'_, AuthBinding>;
}
