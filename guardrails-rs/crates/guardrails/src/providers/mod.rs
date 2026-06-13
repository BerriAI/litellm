mod azure;
mod bedrock;
mod generic;
mod lakera;
mod openai_moderation;
mod presidio;

pub use azure::{AzurePromptShield, AzureTextModeration};
pub use bedrock::BedrockGuardrail;
pub use generic::GenericGuardrailApi;
pub use lakera::LakeraV2;
pub use openai_moderation::OpenaiModeration;
pub use presidio::Presidio;

use crate::{Guardrail, HttpClient, ProviderConfig, ProviderError};
use std::sync::Arc;
use std::time::Instant;

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

pub(crate) fn map_send_error(start: Instant) -> impl Fn(reqwest::Error) -> ProviderError {
    move |e| {
        if e.is_timeout() {
            ProviderError::Timeout {
                ms: start.elapsed().as_millis() as u64,
            }
        } else {
            ProviderError::Network {
                message: e.to_string(),
            }
        }
    }
}

pub(crate) async fn read_success_json(
    resp: reqwest::Response,
) -> Result<serde_json::Value, ProviderError> {
    let status = resp.status().as_u16();
    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(ProviderError::Upstream { status, body });
    }
    resp.json()
        .await
        .map_err(|e| ProviderError::InvalidResponse {
            message: e.to_string(),
        })
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
        ProviderConfig::OpenaiModeration(cfg) => {
            if let Some(base) = &cfg.api_base {
                validate_url(base)?;
            }
            Ok(Box::new(OpenaiModeration::new(cfg, http.clone())))
        }
        ProviderConfig::AzurePromptShield(cfg) => {
            validate_url(&cfg.api_base)?;
            Ok(Box::new(AzurePromptShield::new(cfg, http.clone())))
        }
        ProviderConfig::AzureTextModeration(cfg) => {
            validate_url(&cfg.api_base)?;
            Ok(Box::new(AzureTextModeration::new(cfg, http.clone())))
        }
        ProviderConfig::Presidio(cfg) => Ok(Box::new(Presidio::new(cfg, http.clone()))),
        ProviderConfig::LakeraV2(cfg) => {
            if let Some(base) = &cfg.api_base {
                validate_url(base)?;
            }
            Ok(Box::new(LakeraV2::new(cfg, http.clone())))
        }
        ProviderConfig::Bedrock(cfg) => Ok(Box::new(BedrockGuardrail::new(cfg, http.clone()))),
    }
}
