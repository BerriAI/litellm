use serde_json::{Map, Value};

use super::{
    Citation, ContentBlock, ContextManagementResponse, MessagesContainer, MessagesUsage,
    StopDetails,
};
use crate::recognized::Recognized;
use crate::serde_compat::{Nullable, deserialize_present};

#[macro_rules_attribute::apply(wire_type)]
pub struct MessagesStreamMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: super::MessageType,
    pub role: super::MessageRole,
    pub model: String,
    pub content: Vec<Recognized<ContentBlock>>,
    pub stop_reason: Option<super::StopReason>,
    pub stop_sequence: Option<String>,
    pub usage: MessagesUsage,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub safeguard_results: Option<Recognized<Vec<Map<String, Value>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesContentBlockDelta {
    TextDelta {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    InputJsonDelta {
        partial_json: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    #[serde(rename = "citations_delta")]
    Citations {
        citation: Box<Recognized<Citation>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ThinkingDelta {
        thinking: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    SignatureDelta {
        signature: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    CompactionDelta {
        content: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct MessagesContentBlock {
    #[serde(rename = "type")]
    pub block_type: super::ContentBlockType,
    #[serde(flatten)]
    pub payload: super::ContentBlockPayload,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub tool_use_id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cache_control: Option<Recognized<super::CacheControl>>,
}

impl std::ops::Deref for MessagesContentBlock {
    type Target = super::ContentBlockPayload;

    fn deref(&self) -> &Self::Target {
        &self.payload
    }
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MessagesDelta {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub stop_reason: Option<Nullable<super::StopReason>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub stop_sequence: Option<Nullable<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub stop_details: Option<Recognized<StopDetails>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub container: Option<Recognized<MessagesContainer>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub safeguard_results: Option<Recognized<Vec<Map<String, Value>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MessagesStreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_present"
    )]
    pub details: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesStreamEvent {
    MessageStart {
        message: Box<MessagesStreamMessage>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ContentBlockStart {
        index: u64,
        content_block: Box<MessagesContentBlock>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ContentBlockDelta {
        index: u64,
        delta: MessagesContentBlockDelta,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ContentBlockStop {
        index: u64,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    MessageDelta {
        delta: Box<MessagesDelta>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Box<MessagesUsage>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        context_management: Option<Box<Recognized<ContextManagementResponse>>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    MessageStop {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Box<MessagesUsage>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Ping {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Error {
        error: MessagesStreamError,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;
    use crate::formats::messages::{
        ContentBlockPayload, ContentBlockType, MessageRole, MessageType, StopReason,
    };

    #[fixture]
    fn stream_message() -> MessagesStreamMessage {
        MessagesStreamMessage {
            id: "msg_1".into(),
            message_type: MessageType::Message,
            role: MessageRole::Assistant,
            model: "model".into(),
            content: vec![],
            stop_reason: None,
            stop_sequence: None,
            usage: MessagesUsage {
                input_tokens: Some(Nullable::Value(3)),
                output_tokens: Some(Nullable::Value(0)),
                ..Default::default()
            },
            safeguard_results: None,
            extra: Map::new(),
        }
    }

    #[rstest]
    fn response_as_sse_message_start_contract(stream_message: MessagesStreamMessage) {
        let event = MessagesStreamEvent::MessageStart {
            message: Box::new(stream_message),
            extra: Map::new(),
        };
        let wire = json!({"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"model","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}});
        assert_eq!(serde_json::to_value(&event).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<MessagesStreamEvent>(wire).unwrap(),
            event
        );
    }

    #[rstest]
    #[case::text(ContentBlockType::Text, ContentBlockPayload { text: Some(Nullable::Value(String::new())), ..Default::default() }, json!({"type":"text","text":""}))]
    #[case::tool_use(ContentBlockType::ToolUse, ContentBlockPayload { id: Some(Nullable::Value("toolu_1".into())), name: Some(Nullable::Value("get_weather".into())), input: Some(Recognized::Known(Map::new())), ..Default::default() }, json!({"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}))]
    fn response_as_sse_content_block_start_contract(
        #[case] block_type: ContentBlockType,
        #[case] payload: ContentBlockPayload,
        #[case] block_wire: Value,
    ) {
        let event = MessagesStreamEvent::ContentBlockStart {
            index: 0,
            content_block: Box::new(MessagesContentBlock {
                block_type,
                payload,
                tool_use_id: None,
                cache_control: None,
            }),
            extra: Map::new(),
        };
        let wire = json!({"type":"content_block_start","index":0,"content_block":block_wire});
        assert_eq!(serde_json::to_value(&event).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<MessagesStreamEvent>(wire).unwrap(),
            event
        );
    }

    #[rstest]
    #[case::text(MessagesContentBlockDelta::TextDelta { text: "hello".into(), extra: Map::new() }, json!({"type":"text_delta","text":"hello"}))]
    #[case::tool_json(MessagesContentBlockDelta::InputJsonDelta { partial_json: "{\"city\":\"NYC\"}".into(), extra: Map::new() }, json!({"type":"input_json_delta","partial_json":"{\"city\":\"NYC\"}"}))]
    #[case::thinking(MessagesContentBlockDelta::ThinkingDelta { thinking: "let me think".into(), extra: Map::new() }, json!({"type":"thinking_delta","thinking":"let me think"}))]
    #[case::signature(MessagesContentBlockDelta::SignatureDelta { signature: "sig-abc123".into(), extra: Map::new() }, json!({"type":"signature_delta","signature":"sig-abc123"}))]
    fn response_as_sse_content_block_delta_contract(
        #[case] delta: MessagesContentBlockDelta,
        #[case] delta_wire: Value,
    ) {
        let event = MessagesStreamEvent::ContentBlockDelta {
            index: 1,
            delta,
            extra: Map::new(),
        };
        let wire = json!({"type":"content_block_delta","index":1,"delta":delta_wire});
        assert_eq!(serde_json::to_value(&event).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<MessagesStreamEvent>(wire).unwrap(),
            event
        );
    }

    #[rstest]
    fn response_as_sse_message_delta_contract() {
        let event = MessagesStreamEvent::MessageDelta {
            delta: Box::new(MessagesDelta {
                stop_reason: Some(Nullable::Value(StopReason::EndTurn)),
                ..Default::default()
            }),
            usage: Some(Box::new(MessagesUsage {
                input_tokens: Some(Nullable::Value(3)),
                output_tokens: Some(Nullable::Value(2)),
                ..Default::default()
            })),
            context_management: None,
            extra: Map::new(),
        };
        let wire = json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":3,"output_tokens":2}});
        assert_eq!(serde_json::to_value(&event).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<MessagesStreamEvent>(wire).unwrap(),
            event
        );
    }

    #[rstest]
    #[case::omitted(None, json!({"type":"api_error","message":"incomplete"}))]
    #[case::null(Some(Value::Null), json!({"type":"api_error","message":"incomplete","details":null}))]
    #[case::empty(Some(json!({})), json!({"type":"api_error","message":"incomplete","details":{}}))]
    fn stream_error_details_preserve_presence(
        #[case] details: Option<Value>,
        #[case] error_wire: Value,
    ) {
        let event = MessagesStreamEvent::Error {
            error: MessagesStreamError {
                error_type: "api_error".into(),
                message: "incomplete".into(),
                details,
                extra: Map::new(),
            },
            extra: Map::new(),
        };
        let wire = json!({"type":"error","error":error_wire});
        assert_eq!(serde_json::to_value(&event).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<MessagesStreamEvent>(wire).unwrap(),
            event
        );
    }

    #[rstest]
    #[case::null_message(json!({"type":"api_error","message":null}))]
    #[case::missing_message(json!({"type":"api_error"}))]
    #[case::numeric_type(json!({"type":17,"message":"incomplete"}))]
    fn stream_error_rejects_malformed_required_fields(#[case] error: Value) {
        assert!(
            serde_json::from_value::<MessagesStreamEvent>(json!({"type":"error","error":error}))
                .is_err()
        );
    }
}
