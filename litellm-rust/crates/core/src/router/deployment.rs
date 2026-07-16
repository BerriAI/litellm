//! `model_list` data types, mirroring Python's deployment dict. Deserialize-ready
//! so a deployment can be loaded straight from the proxy config's `model_list`.

use serde::Deserialize;

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
}
