mod generic;

pub use generic::GenericGuardrailApi;

use guardrails_core::{Guardrail, ProviderConfig, ProviderError};
use guardrails_http::HttpClient;
use std::sync::Arc;

fn validate_url(url: &str) -> Result<(), ProviderError> {
    let parsed = url::Url::parse(url).map_err(|e| ProviderError::InvalidConfig {
        message: format!("invalid api_base URL: {e}"),
    })?;
    match parsed.scheme() {
        "http" | "https" => Ok(()),
        scheme => Err(ProviderError::InvalidConfig {
            message: format!("api_base must use http or https scheme, got: {scheme}"),
        }),
    }
}

pub fn build(
    config: ProviderConfig,
    http: &Arc<HttpClient>,
) -> Result<Box<dyn Guardrail>, ProviderError> {
    match config {
        ProviderConfig::GenericGuardrailApi(cfg) => {
            validate_url(&cfg.api_base)?;
            Ok(Box::new(GenericGuardrailApi::new(cfg, http.clone())))
        }
        _ => Err(ProviderError::InvalidConfig {
            message: "provider not yet implemented".to_owned(),
        }),
    }
}
