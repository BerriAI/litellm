use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessagesStreamUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessagesStreamMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<Value>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: MessagesStreamUsage,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesContentBlockDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(rename = "citations_delta")]
    Citations {
        citation: Value,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    CompactionDelta {
        content: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessagesContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessagesDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessagesStreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesStreamEvent {
    MessageStart {
        message: MessagesStreamMessage,
    },
    ContentBlockStart {
        index: u64,
        content_block: MessagesContentBlock,
    },
    ContentBlockDelta {
        index: u64,
        delta: MessagesContentBlockDelta,
    },
    ContentBlockStop {
        index: u64,
    },
    MessageDelta {
        delta: MessagesDelta,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<MessagesStreamUsage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_management: Option<Value>,
    },
    MessageStop {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<MessagesStreamUsage>,
    },
    Ping,
    Error {
        error: MessagesStreamError,
    },
}

#[cfg(test)]
mod tests {
    use super::MessagesStreamEvent;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::start(json!({
        "type": "message_start",
        "message": {
            "id": "msg", "type": "message", "role": "assistant", "model": "test-model",
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 3, "future_usage": {"count": 9}},
            "future_message": [1, 2]
        }
    }))]
    #[case::block(json!({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "future", "payload": {"keep": true}}
    }))]
    #[case::delta(json!({
        "type": "message_delta", "delta": {"stop_reason": "end_turn", "future_delta": 42},
        "usage": {"output_tokens": 7, "future_usage": true}
    }))]
    #[case::error(json!({
        "type": "error", "error": {"type": "future_error", "message": "failed", "future": 42}
    }))]
    fn stream_events_preserve_extensible_fields(#[case] wire: Value) {
        let event: MessagesStreamEvent = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(event).unwrap(), wire);
    }

    #[rstest]
    #[case::invalid_index(json!({"type": "content_block_stop", "index": "zero"}))]
    #[case::missing_delta(json!({"type": "content_block_delta", "index": 0}))]
    #[case::unknown_event(json!({"type": "future_event"}))]
    fn malformed_or_unrecognized_events_remain_rejected(#[case] wire: Value) {
        assert!(serde_json::from_value::<MessagesStreamEvent>(wire).is_err());
    }
}
