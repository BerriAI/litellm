//! `model_list` data types, mirroring Python's deployment dict. Deserialize-ready
//! so a deployment can be loaded straight from the proxy config's `model_list`.

use serde::Deserialize;

use crate::error::CoreError;
use crate::CoreResult;

/// Per-deployment call parameters, mirroring Python's `litellm_params`.
#[derive(Clone, Debug, Deserialize)]
pub struct LiteLLMParams {
    /// Provider model, e.g. `gpt-realtime` or `openai/gpt-realtime`.
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_base: Option<String>,
    #[serde(default)]
    pub custom_llm_provider: Option<String>,
}

/// One entry of the `model_list`, mirroring Python's deployment dict.
#[derive(Clone, Debug, Deserialize)]
pub struct Deployment {
    /// Public alias clients request, e.g. `gpt-realtime`.
    pub model_name: String,
    pub litellm_params: LiteLLMParams,
}

pub fn resolve_deployment_provider(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> CoreResult<(String, String)> {
    let model = model.trim();
    if model.is_empty() {
        return Err(CoreError::InvalidProvider(
            "deployment model is empty".to_string(),
        ));
    }

    if let Some(provider) = custom_llm_provider
        .map(str::trim)
        .filter(|provider| !provider.is_empty())
    {
        let resolved_model = model
            .strip_prefix(&format!("{provider}/"))
            .unwrap_or(model)
            .to_string();
        return Ok((resolved_model, provider.to_string()));
    }

    model
        .split_once('/')
        .filter(|(provider, rest)| !provider.is_empty() && !rest.is_empty())
        .map(|(provider, rest)| (rest.to_string(), provider.to_string()))
        .ok_or_else(|| {
            CoreError::InvalidProvider(format!(
                "deployment model '{model}' must be prefixed with a provider (e.g. \
                 'mistral/mistral-ocr-latest') or set 'custom_llm_provider'"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_from_model_list_entry() {
        let entry = r#"{
            "model_name": "gpt-realtime",
            "litellm_params": {"model": "openai/gpt-realtime", "api_base": "https://x"}
        }"#;
        let deployment: Deployment = serde_json::from_str(entry).expect("valid entry");
        assert_eq!(deployment.model_name, "gpt-realtime");
        assert_eq!(deployment.litellm_params.model, "openai/gpt-realtime");
        assert_eq!(deployment.litellm_params.api_key, None);
        assert_eq!(
            deployment.litellm_params.api_base.as_deref(),
            Some("https://x")
        );
        assert_eq!(deployment.litellm_params.custom_llm_provider, None);
    }

    #[test]
    fn deserializes_explicit_custom_llm_provider() {
        let entry = r#"{
            "model_name": "rust-ocr-mistral",
            "litellm_params": {"model": "mistral-ocr-latest", "custom_llm_provider": "mistral"}
        }"#;
        let deployment: Deployment = serde_json::from_str(entry).expect("valid entry");
        assert_eq!(
            deployment.litellm_params.custom_llm_provider.as_deref(),
            Some("mistral")
        );
    }

    #[test]
    fn resolves_provider_from_prefix_when_no_explicit_provider() {
        assert_eq!(
            resolve_deployment_provider("mistral/mistral-ocr-latest", None).expect("splits"),
            ("mistral-ocr-latest".to_string(), "mistral".to_string())
        );
        assert_eq!(
            resolve_deployment_provider("azure_ai/doc-intelligence/prebuilt-layout", None)
                .expect("splits"),
            (
                "doc-intelligence/prebuilt-layout".to_string(),
                "azure_ai".to_string()
            )
        );
    }

    #[test]
    fn explicit_provider_wins_and_strips_matching_prefix() {
        assert_eq!(
            resolve_deployment_provider("mistral/mistral-ocr-latest", Some("mistral"))
                .expect("resolves"),
            ("mistral-ocr-latest".to_string(), "mistral".to_string())
        );
        assert_eq!(
            resolve_deployment_provider("mistral-ocr-latest", Some("mistral")).expect("resolves"),
            ("mistral-ocr-latest".to_string(), "mistral".to_string())
        );
    }

    #[test]
    fn explicit_provider_keeps_unrelated_prefix() {
        assert_eq!(
            resolve_deployment_provider("openai/some-model", Some("mistral")).expect("resolves"),
            ("openai/some-model".to_string(), "mistral".to_string())
        );
    }

    #[test]
    fn blank_explicit_provider_falls_back_to_prefix() {
        assert_eq!(
            resolve_deployment_provider("mistral/mistral-ocr-latest", Some("   ")).expect("splits"),
            ("mistral-ocr-latest".to_string(), "mistral".to_string())
        );
    }

    #[test]
    fn rejects_model_without_resolvable_provider() {
        assert!(matches!(
            resolve_deployment_provider("mistral-ocr-latest", None),
            Err(CoreError::InvalidProvider(_))
        ));
        assert!(matches!(
            resolve_deployment_provider("mistral/", None),
            Err(CoreError::InvalidProvider(_))
        ));
        assert!(matches!(
            resolve_deployment_provider("   ", Some("mistral")),
            Err(CoreError::InvalidProvider(_))
        ));
    }
}
