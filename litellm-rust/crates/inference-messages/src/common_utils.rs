use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_http::request::string_headers as shared_string_headers;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MessagesProvider {
    Anthropic,
    AzureAi,
    Bedrock,
    Deepseek,
    VertexAi,
}

impl MessagesProvider {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => LlmProviders::Anthropic,
            Self::AzureAi => LlmProviders::AzureAi,
            Self::Bedrock => LlmProviders::Bedrock,
            Self::Deepseek => LlmProviders::Deepseek,
            Self::VertexAi => LlmProviders::VertexAi,
        }
        .into()
    }

    pub(crate) fn config(self) -> &'static dyn BaseMessagesConfig {
        match self {
            Self::Anthropic => &ANTHROPIC_MESSAGES_CONFIG,
            Self::AzureAi => &AZURE_ANTHROPIC_MESSAGES_CONFIG,
            Self::Bedrock => &BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
            Self::Deepseek => &DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG,
            Self::VertexAi => &VERTEX_ANTHROPIC_MESSAGES_CONFIG,
        }
    }
}

/// Python's `get_provider_anthropic_messages_config`: Vertex AI serves only its Claude
/// partner models on this route.
pub(crate) fn messages_provider(provider: LlmProviders, model: &str) -> Option<MessagesProvider> {
    match provider {
        LlmProviders::Anthropic => Some(MessagesProvider::Anthropic),
        LlmProviders::AzureAi => Some(MessagesProvider::AzureAi),
        LlmProviders::Bedrock => Some(MessagesProvider::Bedrock),
        LlmProviders::Deepseek => Some(MessagesProvider::Deepseek),
        LlmProviders::VertexAi if model.to_ascii_lowercase().contains("claude") => {
            Some(MessagesProvider::VertexAi)
        }
        _ => None,
    }
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

    use super::{MessagesProvider, messages_provider, string_headers};
    use crate::Error;
    use litellm_core_utils::get_llm_provider_logic::LlmProviders;

    #[rstest]
    #[case::anthropic("anthropic", MessagesProvider::Anthropic)]
    #[case::azure_ai("azure_ai", MessagesProvider::AzureAi)]
    #[case::bedrock("bedrock", MessagesProvider::Bedrock)]
    #[case::deepseek("deepseek", MessagesProvider::Deepseek)]
    #[case::vertex_ai("vertex_ai", MessagesProvider::VertexAi)]
    fn provider_round_trips_through_its_python_name(
        #[case] name: &str,
        #[case] provider: MessagesProvider,
    ) {
        assert_eq!(
            messages_provider(name.parse::<LlmProviders>().unwrap(), "claude-sonnet-4-5"),
            Some(provider)
        );
        assert_eq!(provider.as_str(), name);
    }

    #[rstest]
    #[case::openai(LlmProviders::Openai, "gpt-5")]
    #[case::vertex_gemini(LlmProviders::VertexAi, "gemini-2.5-pro")]
    fn provider_without_a_messages_config_is_rejected(
        #[case] provider: LlmProviders,
        #[case] model: &str,
    ) {
        assert_eq!(messages_provider(provider, model), None);
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
