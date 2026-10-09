use std::time::Duration;

use bytes::Bytes;
use litellm_host::call::CallOutput;
use litellm_llms::base_llm::{
    litellm_params::LitellmParams, messages::context::MessagesModelCapabilities,
};
use litellm_llms_types::{
    formats::messages::{MessagesRequest, MessagesResponse},
    headers::ProviderSpecificHeaders,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::Error;

pub struct MessagesCall {
    pub body: MessagesRequest,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub litellm_params: LitellmParams,
    pub extra_headers: Option<Map<String, Value>>,
    pub provider_specific_header: Option<ProviderSpecificHeaders>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}

pub fn messages_body(body: Map<String, Value>) -> Result<MessagesRequest, Error> {
    serde_json::from_value(Value::Object(body)).map_err(invalid_request)
}

/// The caller's litellm params, projected by a host from the keys [`LitellmParams::fields`]
/// names. A key present with a value of the wrong type is a request error, as it is for
/// Python's `GenericLiteLLMParams(**kwargs)`.
pub fn litellm_params(fields: Map<String, Value>) -> Result<LitellmParams, Error> {
    serde_json::from_value(Value::Object(fields))
        .map_err(|err| Error::InvalidRequest(format!("invalid litellm params: {err}").into()))
}

pub(super) fn invalid_request(err: serde_json::Error) -> Error {
    Error::InvalidRequest(format!("invalid Anthropic messages request: {err}").into())
}

pub type MessagesCallResponse =
    CallOutput<Box<MessagesResponse>, super::route::MessagesStreamHead, Bytes, Error>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessagesShaping {
    #[serde(default)]
    pub capabilities: MessagesModelCapabilities,
    #[serde(flatten)]
    pub settings: MessagesSettings,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessagesSettings {
    #[serde(default)]
    pub drop_params: bool,
    #[serde(default)]
    pub reasoning_auto_summary: bool,
    #[serde(default)]
    pub additional_drop_params: Vec<String>,
}

#[cfg(test)]
mod tests {
    use litellm_llms::base_llm::messages::context::SupportedEffortTiers;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest]
    #[case::nothing_projected(json!({}), MessagesShaping::default())]
    #[case::only_drop_params(
        json!({"drop_params": true}),
        MessagesShaping {
            settings: MessagesSettings { drop_params: true, ..MessagesSettings::default() },
            ..MessagesShaping::default()
        },
    )]
    #[case::only_reasoning_auto_summary(
        json!({"reasoning_auto_summary": true}),
        MessagesShaping {
            settings: MessagesSettings { reasoning_auto_summary: true, ..MessagesSettings::default() },
            ..MessagesShaping::default()
        },
    )]
    #[case::only_additional_drop_params(
        json!({"additional_drop_params": ["tools[*].input_examples"]}),
        MessagesShaping {
            settings: MessagesSettings {
                additional_drop_params: vec!["tools[*].input_examples".to_string()],
                ..MessagesSettings::default()
            },
            ..MessagesShaping::default()
        },
    )]
    #[case::partial_capabilities(
        json!({"capabilities": {"supports_reasoning": true}}),
        MessagesShaping {
            capabilities: MessagesModelCapabilities {
                supports_reasoning: true,
                ..MessagesModelCapabilities::default()
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
            settings: MessagesSettings {
                drop_params: true,
                reasoning_auto_summary: true,
                additional_drop_params: vec!["metadata.user_id".to_string(), "thinking".to_string()],
            },
            capabilities: MessagesModelCapabilities {
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
        },
    )]
    fn shaping_deserializes_with_defaults_for_absent_fields(
        #[case] projected: Value,
        #[case] expected: MessagesShaping,
    ) {
        let shaping: MessagesShaping = serde_json::from_value(projected).unwrap();
        assert_eq!(shaping, expected);
        let serialized = serde_json::to_value(&shaping).unwrap();
        assert_eq!(
            serialized["drop_params"],
            json!(expected.settings.drop_params)
        );
        assert_eq!(
            serialized["reasoning_auto_summary"],
            json!(expected.settings.reasoning_auto_summary)
        );
        assert_eq!(
            serialized["additional_drop_params"],
            json!(expected.settings.additional_drop_params)
        );
        assert_eq!(
            serialized["capabilities"],
            serde_json::to_value(expected.capabilities).unwrap()
        );
    }
}
