use serde_json::{Map, Value};

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
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

#[macro_rules_attribute::apply(wire_type)]
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

#[macro_rules_attribute::apply(wire_type)]
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

#[macro_rules_attribute::apply(wire_type)]
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

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
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

#[macro_rules_attribute::apply(wire_type)]
pub struct MessagesStreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
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
