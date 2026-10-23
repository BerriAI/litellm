use std::{future::Future, pin::Pin};

use litellm_auth_gcp::{VertexAuth, VertexConfig};
use litellm_auth_types::{InputSource, SecretValue, Sourced};
use litellm_cache::Error;
use litellm_layered::SharedLookup;

pub trait TokenSource: Send + Sync + 'static {
    fn bearer_token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>>;
}

pub struct GcpTokenSource {
    auth: VertexAuth,
    config: VertexConfig,
    env: SharedLookup,
}

impl GcpTokenSource {
    pub fn new(path_service_account: Option<String>, env: SharedLookup) -> Self {
        let credentials = path_service_account
            .map(|path| Sourced::new(SecretValue::new(path), InputSource::Deployment));
        Self {
            auth: VertexAuth::default(),
            config: VertexConfig::new(credentials, None, None),
            env,
        }
    }
}

impl TokenSource for GcpTokenSource {
    fn bearer_token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async move {
            self.auth
                .access_token(&self.config, &|name| self.env.get(name))
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
