use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_http::{HttpClientConfig, HttpClientPool, media::UrlPolicy};
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use litellm_secrets::source::SecretSource;

#[derive(Clone)]
pub struct CoreResources {
    pub pool: Arc<HttpClientPool>,
    pub auth: Arc<AuthServices>,
}

impl CoreResources {
    pub fn new(pool: Arc<HttpClientPool>) -> Self {
        Self {
            pool,
            auth: Arc::new(AuthServices::default()),
        }
    }

    pub fn ocr_client(
        &self,
        config: &HttpClientConfig,
        url_policy: UrlPolicy,
        settings: OcrSettings,
        secrets: Arc<dyn SecretSource>,
    ) -> Result<OcrClient, litellm_http::Error> {
        OcrClient::new(
            &self.pool,
            config,
            url_policy,
            self.auth.clone(),
            settings,
            secrets,
        )
    }
}
