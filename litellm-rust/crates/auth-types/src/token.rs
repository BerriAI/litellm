use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::SystemTime;

use veil::Redact;

use crate::{Error, SecretValue};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedCredential {
    Static(SecretValue),
    AccessToken {
        token: SecretValue,
        expires_on: Option<SystemTime>,
    },
}

impl ResolvedCredential {
    pub fn secret(&self) -> &SecretValue {
        match self {
            Self::Static(secret) | Self::AccessToken { token: secret, .. } => secret,
        }
    }
}

pub type TokenFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ResolvedCredential, Error>> + Send + 'a>>;

pub trait TokenProvider: std::fmt::Debug + Send + Sync {
    fn acquire(&self) -> TokenFuture<'_>;
}

#[derive(Clone, Redact)]
pub struct TokenProviderHandle(#[redact(with = "[REDACTED]")] Arc<dyn TokenProvider>);

impl TokenProviderHandle {
    pub fn new(caller: Arc<dyn TokenProvider>) -> Self {
        Self(caller)
    }

    pub fn from_callback<F, Fut>(acquire: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ResolvedCredential, Error>> + Send + 'static,
    {
        Self::new(Arc::new(CallbackTokenProvider(acquire)))
    }

    pub async fn acquire(&self) -> Result<ResolvedCredential, Error> {
        self.0.acquire().await
    }
}

struct CallbackTokenProvider<F>(F);

impl<F> std::fmt::Debug for CallbackTokenProvider<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CallbackTokenProvider")
    }
}

impl<F, Fut> TokenProvider for CallbackTokenProvider<F>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = Result<ResolvedCredential, Error>> + Send + 'static,
{
    fn acquire(&self) -> TokenFuture<'_> {
        Box::pin((self.0)())
    }
}
