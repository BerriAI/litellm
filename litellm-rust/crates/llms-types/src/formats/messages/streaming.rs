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
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        usage: Option<Nullable<Box<MessagesUsage>>>,
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
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        usage: Option<Nullable<Box<MessagesUsage>>>,
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
    use rstest::rstest;
    use serde_json::json;

    use super::*;

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
