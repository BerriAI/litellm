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

use std::net::IpAddr;
use std::time::Instant;

use litellm_core::guardrails::{ProviderConfig, ProviderError};

use super::http;
use super::provider::Guardrail;

fn is_link_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_link_local(),
        // IPv6 link-local: fe80::/10
        IpAddr::V6(v6) => (v6.segments()[0] & 0xffc0) == 0xfe80,
    }
}

pub(crate) fn validate_url(url: &str) -> Result<(), ProviderError> {
    let parsed = url::Url::parse(url).map_err(|e| ProviderError::InvalidConfig {
        message: format!("invalid api_base URL: {e}"),
    })?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(ProviderError::InvalidConfig {
                message: format!("api_base must use http or https scheme, got: {scheme}"),
            })
        }
    }
    // Defense-in-depth: reject literal link-local addresses (e.g. the cloud
    // metadata endpoint 169.254.169.254). Loopback and RFC-1918 are allowed
    // because guardrail services are commonly self-hosted on internal networks.
    // This is a best-effort check on literal IPs; it does not resolve hostnames.
    if let Some(host) = parsed.host_str() {
        if let Ok(ip) = host.parse::<IpAddr>() {
            if is_link_local(ip) {
                return Err(ProviderError::InvalidConfig {
                    message: format!("api_base must not target a link-local address: {host}"),
                });
            }
        }
    }
    Ok(())
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
        ProviderConfig::Presidio(cfg) => {
            // Validate the normalized bases (presidio accepts scheme-less hosts
            // and prepends http://), consistent with the other providers.
            validate_url(&presidio::normalize_base(&cfg.presidio_analyzer_api_base))?;
            if let Some(base) = &cfg.presidio_anonymizer_api_base {
                validate_url(&presidio::normalize_base(base))?;
            }
            Ok(Box::new(Presidio::new(cfg)))
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_core::guardrails::PresidioConfig;

    #[test]
    fn validate_url_requires_http_scheme() {
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://localhost:5002").is_ok());
        assert!(validate_url("file:///etc/passwd").is_err());
        assert!(validate_url("ftp://example.com").is_err());
    }

    #[test]
    fn validate_url_rejects_link_local_but_allows_internal() {
        // Cloud metadata / link-local is rejected.
        assert!(validate_url("http://169.254.169.254/latest/meta-data/").is_err());
        // Loopback and RFC-1918 are allowed (guardrails are often self-hosted).
        assert!(validate_url("http://127.0.0.1:5002").is_ok());
        assert!(validate_url("http://10.0.0.5:8080").is_ok());
        assert!(validate_url("http://192.168.1.10:5002").is_ok());
    }

    #[test]
    fn build_presidio_validates_analyzer_base() {
        let bad: PresidioConfig = serde_json::from_value(
            serde_json::json!({"presidio_analyzer_api_base": "169.254.169.254"}),
        )
        .unwrap();
        assert!(build(ProviderConfig::Presidio(bad)).is_err());

        let ok: PresidioConfig = serde_json::from_value(
            serde_json::json!({"presidio_analyzer_api_base": "http://localhost:5002"}),
        )
        .unwrap();
        assert!(build(ProviderConfig::Presidio(ok)).is_ok());
    }
}
