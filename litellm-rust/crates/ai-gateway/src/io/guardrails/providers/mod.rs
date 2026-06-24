//! Network-backed guardrail providers and the dispatch that builds them.

mod azure;
mod bedrock;
mod generic;
mod lakera;
mod local_pii;
mod openai_moderation;
mod presidio;

pub use azure::{AzurePromptShield, AzureTextModeration};
pub use bedrock::BedrockGuardrail;
pub use generic::GenericGuardrailApi;
pub use lakera::LakeraV2;
pub use local_pii::LocalPii;
pub use openai_moderation::OpenaiModeration;
pub use presidio::Presidio;

use std::time::Instant;

use litellm_core::guardrails::{ProviderConfig, ProviderError};

use super::http;
use super::provider::Guardrail;

pub(crate) fn validate_url(url: &str) -> Result<(), ProviderError> {
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

pub(crate) fn read_success_json(
    resp: reqwest::blocking::Response,
) -> Result<serde_json::Value, ProviderError> {
    let status = resp.status().as_u16();
    if !resp.status().is_success() {
        let body = resp.text().unwrap_or_default();
        return Err(ProviderError::Upstream {
            status,
            body: http::truncate_body(&body),
        });
    }
    resp.json().map_err(|e| ProviderError::InvalidResponse {
        message: e.to_string(),
    })
}

/// Build the provider for a fully-resolved config.
pub fn build(config: ProviderConfig) -> Result<Box<dyn Guardrail>, ProviderError> {
    match config {
        ProviderConfig::GenericGuardrailApi(cfg) => {
            validate_url(&cfg.api_base)?;
            Ok(Box::new(GenericGuardrailApi::new(cfg)))
        }
        ProviderConfig::OpenaiModeration(cfg) => {
            if let Some(base) = &cfg.api_base {
                validate_url(base)?;
            }
            Ok(Box::new(OpenaiModeration::new(cfg)))
        }
        ProviderConfig::AzurePromptShield(cfg) => {
            validate_url(&cfg.api_base)?;
            Ok(Box::new(AzurePromptShield::new(cfg)))
        }
        ProviderConfig::AzureTextModeration(cfg) => {
            validate_url(&cfg.api_base)?;
            Ok(Box::new(AzureTextModeration::new(cfg)))
        }
        ProviderConfig::Presidio(cfg) => Ok(Box::new(Presidio::new(cfg))),
        ProviderConfig::LakeraV2(cfg) => {
            if let Some(base) = &cfg.api_base {
                validate_url(base)?;
            }
            Ok(Box::new(LakeraV2::new(cfg)))
        }
        ProviderConfig::Bedrock(cfg) => {
            if let Some(endpoint) = &cfg.aws_bedrock_runtime_endpoint {
                validate_url(endpoint)?;
            }
            Ok(Box::new(BedrockGuardrail::new(cfg)))
        }
        ProviderConfig::LocalPii(cfg) => Ok(Box::new(LocalPii::new(cfg))),
    }
}
