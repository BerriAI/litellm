//! Network-backed guardrail providers and the dispatch that builds them.

mod local_pii;
mod openai_moderation;

pub use local_pii::LocalPii;
pub use openai_moderation::OpenaiModeration;

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
        ProviderConfig::OpenaiModeration(cfg) => {
            if let Some(base) = &cfg.api_base {
                validate_url(base)?;
            }
            Ok(Box::new(OpenaiModeration::new(cfg)))
        }
        ProviderConfig::LocalPii(cfg) => Ok(Box::new(LocalPii::new(cfg))),
        other => Err(ProviderError::InvalidConfig {
            message: format!(
                "guardrail provider not yet supported by the Rust engine: {}",
                provider_label(&other)
            ),
        }),
    }
}

fn provider_label(config: &ProviderConfig) -> &'static str {
    match config {
        ProviderConfig::GenericGuardrailApi(_) => "generic_guardrail_api",
        ProviderConfig::OpenaiModeration(_) => "openai_moderation",
        ProviderConfig::AzurePromptShield(_) => "azure/prompt_shield",
        ProviderConfig::AzureTextModeration(_) => "azure/text_moderations",
        ProviderConfig::Presidio(_) => "presidio",
        ProviderConfig::LakeraV2(_) => "lakera_v2",
        ProviderConfig::Bedrock(_) => "bedrock",
        ProviderConfig::LocalPii(_) => "local_pii",
    }
}
