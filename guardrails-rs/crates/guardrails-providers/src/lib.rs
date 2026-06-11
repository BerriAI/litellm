mod generic;

pub use generic::GenericGuardrailApi;

use guardrails_core::{Guardrail, ProviderConfig, ProviderError};
use guardrails_http::HttpClient;
use std::sync::Arc;

pub fn build(
    config: ProviderConfig,
    http: &Arc<HttpClient>,
) -> Result<Box<dyn Guardrail>, ProviderError> {
    match config {
        ProviderConfig::GenericGuardrailApi(cfg) => {
            Ok(Box::new(GenericGuardrailApi::new(cfg, http.clone())))
        }
        _ => Err(ProviderError::InvalidConfig {
            message: "provider not yet implemented".to_owned(),
        }),
    }
}
