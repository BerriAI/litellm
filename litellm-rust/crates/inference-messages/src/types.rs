use std::time::Duration;

use bytes::Bytes;
use litellm_host::call::CallOutput;
use litellm_llms::{
    base_llm::messages::context::MessagesModelCapabilities,
    bedrock::{
        messages::connection::BedrockMessagesConnection,
        request_metadata::BedrockRequestMetadataInput,
    },
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
    pub extra_headers: Option<Map<String, Value>>,
    pub provider_specific_header: Option<ProviderSpecificHeaders>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}

pub fn messages_body(body: Map<String, Value>) -> Result<MessagesRequest, Error> {
    serde_json::from_value(Value::Object(body)).map_err(invalid_request)
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bedrock_connection: Option<BedrockMessagesConnection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bedrock_request_metadata: Option<BedrockRequestMetadataInput>,
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
            ..MessagesShaping::default()
        },
    )]
    #[case::bedrock_inputs(
        json!({
            "bedrock_connection": {
                "api_base": "https://projected.test",
                "region": "us-east-2",
                "model_id": "override/model",
                "workspace_id": "trusted-project"
            },
            "bedrock_request_metadata": {
                "allowed_fields": ["user_api_key_alias", "spend_logs_metadata"],
                "sources": [{"identity": [["user_api_key_alias", "prod-key"]], "spend_logs": []}]
            }
        }),
        MessagesShaping {
            bedrock_connection: Some(BedrockMessagesConnection {
                api_base: Some("https://projected.test".to_string()),
                region: Some("us-east-2".to_string()),
                model_id: Some("override/model".to_string()),
                workspace_id: Some("trusted-project".to_string()),
            }),
            bedrock_request_metadata: Some(BedrockRequestMetadataInput {
                allowed_fields: vec!["user_api_key_alias".to_string(), "spend_logs_metadata".to_string()],
                sources: vec![litellm_llms::bedrock::request_metadata::BedrockMetadataSource {
                    identity: vec![("user_api_key_alias".to_string(), "prod-key".to_string())],
                    spend_logs: vec![],
                }],
            }),
            ..MessagesShaping::default()
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
        if expected.bedrock_connection.is_none() {
            assert_eq!(serialized.get("bedrock_connection"), None);
        }
        if expected.bedrock_request_metadata.is_none() {
            assert_eq!(serialized.get("bedrock_request_metadata"), None);
        }
    }

    #[rstest]
    fn default_shaping_serializes_without_bedrock_keys() {
        let serialized = serde_json::to_value(MessagesShaping::default()).unwrap();
        assert_eq!(serialized.get("bedrock_connection"), None);
        assert_eq!(serialized.get("bedrock_request_metadata"), None);
    }
}
