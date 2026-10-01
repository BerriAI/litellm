use std::{future::Future, pin::Pin};

use litellm_auth_gcp::{VertexAuth, VertexConfig};
use litellm_auth_types::{InputSource, SecretValue, Sourced};
use litellm_cache::Error;

pub trait TokenSource: Send + Sync + 'static {
    fn bearer_token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>>;
}

pub struct GcpTokenSource {
    auth: VertexAuth,
    config: VertexConfig,
}

impl GcpTokenSource {
    pub fn new(path_service_account: Option<String>) -> Self {
        let credentials = path_service_account
            .map(|path| Sourced::new(SecretValue::new(path), InputSource::Deployment));
        Self {
            auth: VertexAuth::default(),
            config: VertexConfig::new(credentials, None, None),
        }
    }
}

impl TokenSource for GcpTokenSource {
    fn bearer_token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move {
            self.auth
                .access_token(&self.config, &|name| std::env::var(name).ok())
                .await
                .map_err(|_| Error::Unavailable)
        })
    }
}

pub struct StaticTokenSource(pub String);

impl TokenSource for StaticTokenSource {
    fn bearer_token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move { Ok(self.0.clone()) })
    }
}
