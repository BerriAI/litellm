use std::time::Duration;

use litellm_llms::{
    anthropic::common_utils::AnthropicModelCapabilities, base_llm::auth::ValidatedEnvironment,
};
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde::{Deserialize, Serialize};

use super::common_utils::MessagesProvider;

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

pub(crate) struct ProviderMessagesRequest {
    pub(crate) provider: MessagesProvider,
    pub(crate) url: String,
    pub(crate) body: AnthropicMessagesRequest,
    /// The forwarded, default and feature headers plus how the call authenticates; the
    /// credential itself is applied when the request is sent.
    pub(crate) environment: ValidatedEnvironment,
    pub(crate) timeout: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use litellm_llms::anthropic::common_utils::SupportedEffortTiers;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

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
