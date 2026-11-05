use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_http::request::string_headers as shared_string_headers;
use litellm_llms::tencent::messages::transformation::TENCENT_MESSAGES_CONFIG;
use litellm_llms::{
    anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    azure_ai::messages::transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG,
    base_llm::messages::transformation::BaseMessagesConfig,
    bedrock::messages::invoke_transformations::anthropic_claude_transformation::BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
    deepseek::messages::transformation::DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG,
    edenai::messages::transformation::EDENAI_MESSAGES_CONFIG,
    minimax::messages::transformation::MINIMAX_MESSAGES_CONFIG,
    vertex_ai::messages::transformation::VERTEX_ANTHROPIC_MESSAGES_CONFIG,
};
use litellm_llms::{
    bedrock::claude_platform::messages::transformation::BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG,
    bedrock_mantle::messages::transformation::BEDROCK_MANTLE_MESSAGES_CONFIG,
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

    /// Python's `MESSAGES_SERVICE_TIER_PROVIDERS`: everyone else has `service_tier` dropped.
    pub(crate) fn accepts_service_tier(self) -> bool {
        matches!(
            self.provider,
            LlmProviders::Anthropic | LlmProviders::AzureAi | LlmProviders::BedrockMantle
        )
    }
}

/// Python's `get_provider_anthropic_messages_config`: Vertex AI serves only its Claude
/// partner models on this route.
pub(crate) fn messages_provider(provider: LlmProviders, model: &str) -> Option<MessagesProvider> {
    use litellm_llms::openrouter::messages::transformation::OPENROUTER_MESSAGES_CONFIG;

    let config: &'static dyn BaseMessagesConfig = match provider {
        LlmProviders::Tencent => &TENCENT_MESSAGES_CONFIG,
        LlmProviders::Anthropic => &ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::AzureAi => &AZURE_ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::Bedrock if model.starts_with("claude_platform/") => {
            &BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
        }
        LlmProviders::Bedrock if model.starts_with("mantle/") => &BEDROCK_MANTLE_MESSAGES_CONFIG,
        LlmProviders::Bedrock => &BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::BedrockMantle => &BEDROCK_MANTLE_MESSAGES_CONFIG,
        LlmProviders::Deepseek => &DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::Edenai => &EDENAI_MESSAGES_CONFIG,
        LlmProviders::Minimax => &MINIMAX_MESSAGES_CONFIG,
        LlmProviders::VertexAi if model.to_ascii_lowercase().contains("claude") => {
            &VERTEX_ANTHROPIC_MESSAGES_CONFIG
        }
        LlmProviders::Openrouter => &OPENROUTER_MESSAGES_CONFIG,
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

    use super::{messages_provider, string_headers};
    use crate::Error;
    use litellm_core_utils::get_llm_provider_logic::LlmProviders;

    #[rstest]
    #[case::tencent(LlmProviders::Tencent, "tencent")]
    #[case::anthropic(LlmProviders::Anthropic, "anthropic")]
    #[case::azure_ai(LlmProviders::AzureAi, "azure_ai")]
    #[case::bedrock(LlmProviders::Bedrock, "bedrock")]
    #[case::deepseek(LlmProviders::Deepseek, "deepseek")]
    #[case::edenai(LlmProviders::Edenai, "edenai")]
    #[case::minimax(LlmProviders::Minimax, "minimax")]
    #[case::vertex_ai(LlmProviders::VertexAi, "vertex_ai")]
    #[case::openrouter(LlmProviders::Openrouter, "openrouter")]
    fn provider_keeps_its_python_name(#[case] provider: LlmProviders, #[case] name: &str) {
        let provider = messages_provider(provider, "claude-sonnet-4-5").unwrap();
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

/// Python's `_merge_messages_header_sources`: later sources win per header name, except
/// `anthropic-beta` (any casing), whose values from every source are unioned, deduplicated and
/// sorted into one lowercase `anthropic-beta` entry.
pub fn merge_messages_header_sources(
    sources: impl IntoIterator<Item = Map<String, Value>>,
) -> Map<String, Value> {
    let mut merged = Map::new();
    let mut betas = std::collections::BTreeSet::new();
    for (name, value) in sources.into_iter().flatten() {
        if !name.eq_ignore_ascii_case("anthropic-beta") {
            merged.insert(name, value);
            continue;
        }
        let values: Vec<&str> = match &value {
            Value::String(text) => vec![text.as_str()],
            Value::Array(entries) => entries.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        betas.extend(
            values
                .iter()
                .flat_map(|value| value.split(','))
                .map(str::trim)
                .filter(|beta| !beta.is_empty())
                .map(str::to_owned),
        );
    }
    if !betas.is_empty() {
        merged.insert(
            "anthropic-beta".to_owned(),
            Value::String(betas.into_iter().collect::<Vec<_>>().join(",")),
        );
    }
    merged
}

#[cfg(test)]
mod merge_tests {
    use serde_json::{Map, Value, json};

    use super::merge_messages_header_sources;

    fn map(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn later_sources_win_except_anthropic_beta_which_is_unioned_and_sorted() {
        let merged = merge_messages_header_sources([
            map(json!({"x-a": "1", "anthropic-beta": "b-2, a-1"})),
            map(json!({"x-a": "2", "Anthropic-Beta": ["c-3", 7, "a-1"]})),
            map(json!({"anthropic-beta": 5})),
        ]);
        assert_eq!(
            Value::Object(merged),
            json!({"x-a": "2", "anthropic-beta": "a-1,b-2,c-3"})
        );
    }

    #[test]
    fn no_beta_values_leave_no_beta_header() {
        let merged = merge_messages_header_sources([map(json!({"anthropic-beta": ""}))]);
        assert!(merged.is_empty());
    }
}
