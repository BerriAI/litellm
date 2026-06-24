//! Build a fully-resolved [`ProviderConfig`] from the raw guardrail params the
//! proxy already holds. This is the host edge: it parses the params dict into
//! typed structs, resolves `os.environ/` secrets and env fallbacks, and reads
//! any referenced files. When a config uses a feature the Rust engine does not
//! support, it returns [`Unsupported`] so the caller falls back to Python.

use litellm_core::guardrails::{LocalPiiConfig, OpenaiModerationConfig, ProviderConfig};
use serde::Deserialize;
use serde_json::Value;

/// The Rust engine cannot handle this config; the caller must use the Python
/// implementation. The message is for logs only.
#[derive(Debug)]
pub struct Unsupported(pub String);

fn unsupported(reason: impl Into<String>) -> Unsupported {
    Unsupported(reason.into())
}

/// Resolve a config value: `os.environ/NAME` reads the env var (absent => None);
/// any other non-empty value passes through; empty/whitespace is treated as
/// absent.
fn resolve_secret(value: Option<&str>) -> Option<String> {
    let value = value?;
    if let Some(name) = value.strip_prefix("os.environ/") {
        return std::env::var(name).ok().filter(|v| !v.trim().is_empty());
    }
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| value.to_owned())
}

fn parse_params<T: for<'de> Deserialize<'de>>(params: &Value) -> Result<T, Unsupported> {
    serde_json::from_value(params.clone())
        .map_err(|e| unsupported(format!("could not parse guardrail params: {e}")))
}

pub fn build_config(guardrail_type: &str, params: &Value) -> Result<ProviderConfig, Unsupported> {
    match guardrail_type {
        "openai_moderation" => Ok(ProviderConfig::OpenaiModeration(openai_moderation(params)?)),
        "local_pii" => Ok(ProviderConfig::LocalPii(local_pii(params)?)),
        other => Err(unsupported(format!(
            "guardrail type not supported by the Rust engine: {other}"
        ))),
    }
}

#[derive(Deserialize, Default)]
struct OpenAiParams {
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_base: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

fn openai_moderation(params: &Value) -> Result<OpenaiModerationConfig, Unsupported> {
    let raw: OpenAiParams = parse_params(params)?;
    let api_key = resolve_secret(raw.api_key.as_deref()).or_else(|| {
        std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
    });
    Ok(OpenaiModerationConfig {
        api_key,
        api_base: resolve_secret(raw.api_base.as_deref()),
        model: raw.model,
    })
}

fn local_pii(params: &Value) -> Result<LocalPiiConfig, Unsupported> {
    parse_params(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_falls_back_to_env_key() {
        // SAFETY: single-threaded test, restores nothing it didn't set.
        std::env::set_var("OPENAI_API_KEY", "sk-from-env");
        let cfg = openai_moderation(&serde_json::json!({})).unwrap();
        assert_eq!(cfg.api_key.as_deref(), Some("sk-from-env"));
        std::env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn explicit_key_wins_over_env() {
        let cfg = openai_moderation(&serde_json::json!({"api_key": "sk-explicit"})).unwrap();
        assert_eq!(cfg.api_key.as_deref(), Some("sk-explicit"));
    }

    #[test]
    fn os_environ_prefix_is_resolved() {
        std::env::set_var("MY_MOD_KEY", "sk-resolved");
        let cfg =
            openai_moderation(&serde_json::json!({"api_key": "os.environ/MY_MOD_KEY"})).unwrap();
        assert_eq!(cfg.api_key.as_deref(), Some("sk-resolved"));
        std::env::remove_var("MY_MOD_KEY");
    }

    #[test]
    fn unknown_type_is_unsupported() {
        assert!(build_config("lakera_v2", &serde_json::json!({})).is_err());
    }
}
