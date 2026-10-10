use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_http::request::string_headers as shared_string_headers;
pub(super) use litellm_http::request::truncate_error_body;
use litellm_llms::{
    anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    azure_ai::messages::transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG,
    base_llm::messages::transformation::BaseMessagesConfig,
    bedrock::messages::invoke_transformations::anthropic_claude3_transformation::BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
    deepseek::messages::transformation::DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG,
    vertex_ai::messages::transformation::VERTEX_ANTHROPIC_MESSAGES_CONFIG,
};
use serde_json::{Map, Value};

use super::Error;

const HEADER_CONTEXT: &str = "messages";

#[derive(Clone, Copy)]
pub(crate) struct MessagesProvider {
    provider: LlmProviders,
    config: &'static dyn BaseMessagesConfig,
}

impl MessagesProvider {
    pub(crate) fn as_str(self) -> &'static str {
        self.provider.into()
    }

    pub(crate) fn config(self) -> &'static dyn BaseMessagesConfig {
        self.config
    }
}

/// Python's `get_provider_anthropic_messages_config`: Vertex AI serves only its Claude
/// partner models on this route.
pub(crate) fn messages_provider(provider: LlmProviders, model: &str) -> Option<MessagesProvider> {
    let config: &'static dyn BaseMessagesConfig = match provider {
        LlmProviders::Anthropic => &ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::AzureAi => &AZURE_ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::Bedrock => &BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::Deepseek => &DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::VertexAi if model.to_ascii_lowercase().contains("claude") => {
            &VERTEX_ANTHROPIC_MESSAGES_CONFIG
        }
        _ => return None,
    };
    Some(MessagesProvider { provider, config })
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use rstest::rstest;

    use super::{messages_provider, string_headers, truncate_error_body};
    use crate::Error;
    use litellm_core_utils::get_llm_provider_logic::LlmProviders;

    #[rstest]
    #[case::anthropic("anthropic")]
    #[case::azure_ai("azure_ai")]
    #[case::bedrock("bedrock")]
    #[case::deepseek("deepseek")]
    #[case::vertex_ai("vertex_ai")]
    fn provider_keeps_its_python_name(#[case] name: &str) {
        let provider =
            messages_provider(name.parse::<LlmProviders>().unwrap(), "claude-sonnet-4-5").unwrap();
        assert_eq!(provider.as_str(), name);
    }

    #[rstest]
    #[case::openai(LlmProviders::Openai, "gpt-5")]
    #[case::vertex_gemini(LlmProviders::VertexAi, "gemini-2.5-pro")]
    fn provider_without_a_messages_config_is_rejected(
        #[case] provider: LlmProviders,
        #[case] model: &str,
    ) {
        assert!(messages_provider(provider, model).is_none());
    }

    #[test]
    fn truncate_error_body_caps_long_payloads() {
        let body = "x".repeat(400);
        let truncated = truncate_error_body(&body);
        assert!(truncated.ends_with("... (truncated)"));
        let prefix_chars = truncated
            .strip_suffix("... (truncated)")
            .expect("truncated marker present")
            .chars()
            .count();
        assert_eq!(prefix_chars, 256);
    }

    #[test]
    fn string_headers_rejects_non_string_values() {
        let headers = json!({"x-count": 3}).as_object().unwrap().clone();
        let err = string_headers(Some(headers)).expect_err("non-string header rejected");
        assert_eq!(
            err,
            Error::Headers(litellm_http::request::HeaderError {
                context: "messages",
                name: "x-count".to_string(),
                actual: "number",
            })
        );
    }
}
