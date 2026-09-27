use std::sync::Arc;

use litellm_http::{Client, ClientVariant, HttpClientConfig, media::UrlPolicy};
use litellm_llms::base_llm::ocr::{handler::OcrClient, settings::OcrSettings};
use litellm_secrets::source::SecretSource;

use crate::resources::CoreResources;

#[derive(Clone)]
pub struct CoreClient {
    resources: CoreResources,
    http: HttpClientConfig,
    secrets: Arc<dyn SecretSource>,
    url_policy: UrlPolicy,
    ocr_settings: OcrSettings,
}

impl CoreClient {
    pub fn new(
        resources: CoreResources,
        http: HttpClientConfig,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            resources,
            http,
            secrets,
            url_policy: UrlPolicy::default(),
            ocr_settings: OcrSettings::default(),
        }
    }

    pub fn with_url_policy(self, url_policy: UrlPolicy) -> Self {
        Self { url_policy, ..self }
    }

    pub fn with_ocr_settings(self, ocr_settings: OcrSettings) -> Self {
        Self {
            ocr_settings,
            ..self
        }
    }

    pub fn resources(&self) -> &CoreResources {
        &self.resources
    }

    pub fn http_config(&self) -> &HttpClientConfig {
        &self.http
    }

    pub fn secret_source(&self) -> &Arc<dyn SecretSource> {
        &self.secrets
    }

    pub(crate) fn provider_http(&self) -> Result<Client, litellm_http::Error> {
        self.resources
            .pool
            .client(&self.http, ClientVariant::Provider)
    }

    pub(crate) fn ocr_client(&self) -> Result<OcrClient, litellm_http::Error> {
        OcrClient::new(
            &self.resources.pool,
            &self.http,
            self.url_policy.clone(),
            self.resources.auth.clone(),
            self.ocr_settings.clone(),
            self.secrets.clone(),
        )
    }
}
