use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_http::HttpClientPool;

#[derive(Clone)]
pub struct CoreResources {
    pub pool: Arc<HttpClientPool>,
    pub auth: Arc<AuthServices>,
}

impl CoreResources {
    pub fn new(pool: Arc<HttpClientPool>) -> Self {
        let settings = litellm_http::HttpSettings::from_layers([
            litellm_http::HttpSettingsLayer::from_environment(&|key: &str| std::env::var(key).ok()),
        ]);
        let resolution = litellm_http::Resolution::from(&settings);
        let copilot = pool
            .client(&resolution.config, litellm_http::ClientVariant::NoRedirect)
            .map(litellm_auth_copilot::CopilotAuthService::with_no_redirect_http)
            .unwrap_or_default();
        Self {
            pool,
            auth: Arc::new(AuthServices {
                copilot,
                ..Default::default()
            }),
        }
    }
}
