use std::{future::Future, pin::Pin};

use litellm_auth_gcp::{CredentialSource, GcpAuth};
use litellm_secret_value::SecretValue;

use litellm_cache::Error;

pub trait TokenSource: Send + Sync + 'static {
    fn bearer_token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>>;
}

pub struct GcsTokenSource {
    auth: GcpAuth,
    credentials: Option<String>,
}

impl GcsTokenSource {
    pub fn new(path_service_account: Option<String>) -> Self {
        Self {
            auth: GcpAuth::default(),
            credentials: path_service_account.filter(|path| !path.trim().is_empty()),
        }
    }
}

impl TokenSource for GcsTokenSource {
    fn bearer_token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move {
            self.auth
                .access_token(match &self.credentials {
                    Some(path) => CredentialSource::Trusted(SecretValue::new(path.clone())),
                    None => CredentialSource::application_default(&|name| std::env::var(name).ok()),
                })
                .await
                .map(|token| token.expose().to_owned())
                .map_err(Error::unavailable)
        })
    }
}

pub struct StaticTokenSource(pub String);

impl TokenSource for StaticTokenSource {
    fn bearer_token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move { Ok(self.0.clone()) })
    }
}
