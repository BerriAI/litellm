use std::time::Duration;

use bytes::Bytes;
use futures_util::stream::BoxStream;
use litellm_llms::anthropic::common_utils::AnthropicModelCapabilities;
use litellm_types::{
    llms::anthropic_messages::{
        anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
    },
    utils::ProviderSpecificHeaders,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::Error;

pub struct MessagesCall {
    pub body: AnthropicMessagesRequest,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub provider_specific_header: Option<ProviderSpecificHeaders>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}

pub fn messages_body(body: Map<String, Value>) -> Result<AnthropicMessagesRequest, Error> {
    serde_json::from_value(Value::Object(body)).map_err(|e| Error::RequestDecoding(e.into()))
}

pub enum MessagesResponse {
    Message(Box<AnthropicMessagesResponse>),
    Stream {
        headers: Vec<(String, String)>,
        chunks: BoxStream<'static, Result<Bytes, Error>>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessagesShaping {
    #[serde(default)]
    pub capabilities: AnthropicModelCapabilities,
    #[serde(default)]
    pub drop_params: bool,
    #[serde(default)]
    pub reasoning_auto_summary: bool,
    #[serde(default)]
    pub additional_drop_params: Vec<String>,
}

#[cfg(test)]
mod tests {
    use litellm_llms::anthropic::common_utils::SupportedEffortTiers;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::super::common_utils::MessagesProvider;
    use super::*;

    #[rstest]
    #[case::anthropic("anthropic", Some(MessagesProvider::Anthropic))]
    #[case::azure_ai("azure_ai", Some(MessagesProvider::AzureAi))]
    #[case::case_sensitive("Anthropic", None)]
    #[case::unsupported("openai", None)]
    fn provider_parses_from_its_name(
        #[case] name: &str,
        #[case] expected: Option<MessagesProvider>,
    ) {
        assert_eq!(name.parse::<MessagesProvider>().ok(), expected);
        if let Some(provider) = expected {
            assert_eq!(provider.as_str(), name);
        }
    }

    #[rstest]
    #[case::nothing_projected(json!({}), MessagesShaping::default())]
    #[case::only_drop_params(
        json!({"drop_params": true}),
        MessagesShaping { drop_params: true, ..MessagesShaping::default() },
    )]
    #[case::only_reasoning_auto_summary(
        json!({"reasoning_auto_summary": true}),
        MessagesShaping { reasoning_auto_summary: true, ..MessagesShaping::default() },
    )]
    #[case::only_additional_drop_params(
        json!({"additional_drop_params": ["tools[*].input_examples"]}),
        MessagesShaping {
            additional_drop_params: vec!["tools[*].input_examples".to_string()],
            ..MessagesShaping::default()
        },
    )]
    #[case::partial_capabilities(
        json!({"capabilities": {"supports_reasoning": true}}),
        MessagesShaping {
            capabilities: AnthropicModelCapabilities {
                supports_reasoning: true,
                ..AnthropicModelCapabilities::default()
            },
            ..MessagesShaping::default()
        },
    )]
    #[case::everything_the_python_host_projects(
        json!({
            "capabilities": {
                "supports_reasoning": true,
                "supports_adaptive_thinking": true,
                "thinking_always_on": false,
                "supports_legacy_thinking": false,
                "supports_output_config": true,
                "supports_sampling_params": false,
                "supports_speed": true,
                "effort_tiers": {"minimal": false, "low": true, "medium": true, "high": true, "xhigh": true, "max": false}
            },
            "drop_params": true,
            "reasoning_auto_summary": true,
            "additional_drop_params": ["metadata.user_id", "thinking"]
        }),
        MessagesShaping {
            capabilities: AnthropicModelCapabilities {
                supports_reasoning: true,
                supports_adaptive_thinking: true,
                thinking_always_on: false,
                supports_legacy_thinking: false,
                supports_output_config: true,
                supports_sampling_params: false,
                supports_speed: true,
                effort_tiers: SupportedEffortTiers {
                    minimal: false,
                    low: true,
                    medium: true,
                    high: true,
                    xhigh: true,
                    max: false,
                },
            },
            drop_params: true,
            reasoning_auto_summary: true,
            additional_drop_params: vec!["metadata.user_id".to_string(), "thinking".to_string()],
        },
    )]
    fn shaping_deserializes_with_defaults_for_absent_fields(
        #[case] projected: Value,
        #[case] expected: MessagesShaping,
    ) {
        let shaping: MessagesShaping = serde_json::from_value(projected).unwrap();
        assert_eq!(shaping, expected);
    }
}
