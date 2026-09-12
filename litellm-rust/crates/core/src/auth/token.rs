use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::SystemTime;

use veil::Redact;

use crate::AuthError;

use super::secret::SecretValue;

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
    Pin<Box<dyn Future<Output = Result<ResolvedCredential, AuthError>> + Send + 'a>>;

pub trait TokenProvider: std::fmt::Debug + Send + Sync {
    fn acquire(&self) -> TokenFuture<'_>;
}

#[derive(Clone, Redact)]
pub struct TokenProviderHandle(#[redact(with = "[REDACTED]")] Arc<dyn TokenProvider>);

impl TokenProviderHandle {
    pub fn new(caller: Arc<dyn TokenProvider>) -> Self {
        Self(caller)
    }

    pub async fn acquire(&self) -> Result<ResolvedCredential, AuthError> {
        self.0.acquire().await
    }
}
