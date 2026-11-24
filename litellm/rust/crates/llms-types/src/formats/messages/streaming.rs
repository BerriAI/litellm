use serde_json::{Map, Value};

use crate::{
    as_sent::AsSent,
    formats::messages::{MessagesErrorType, MessagesUsage},
    recognized::Recognized,
};

/// Only the fields Anthropic always sends are named, and a host may leave any of them out.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MessagesStreamMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub message_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<Value>>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<AsSent<MessagesUsage>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct MessagesStreamError {
    #[serde(rename = "type")]
    pub error_type: MessagesErrorType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// An event this crate has no variant for, or one of a known type that does not read as its
/// variant, kept as it was sent.
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct MessagesOtherStreamEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

impl AsRef<str> for MessagesOtherStreamEvent {
    fn as_ref(&self) -> &str {
        &self.event_type
    }
}

/// One event of a Messages stream, read off whatever host sent it: a delta type this crate does
/// not name rides in `Recognized::Unrecognized`, and an event it cannot read in `Other`.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(strum::AsRefStr)]
#[serde(tag = "type", rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
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
        delta: Recognized<MessagesContentBlockDelta>,
    },
    ContentBlockStop {
        index: u64,
    },
    MessageDelta {
        delta: MessagesDelta,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<AsSent<MessagesUsage>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_management: Option<Value>,
    },
    MessageStop {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<AsSent<MessagesUsage>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Ping,
    Error {
        error: MessagesStreamError,
    },
    #[serde(untagged)]
    #[strum(transparent)]
    Other(MessagesOtherStreamEvent),
}

#[cfg(test)]
mod tests {
    use crate::{formats::messages::streaming::MessagesStreamEvent, recognized::Recognized};

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
    #[case::start(json!({"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "x", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {}}}))]
    #[case::block_start(json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))]
    #[case::block_delta(json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "a"}}))]
    #[case::block_stop(json!({"type": "content_block_stop", "index": 0}))]
    #[case::delta(json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}))]
    #[case::stop(json!({"type": "message_stop"}))]
    #[case::ping(json!({"type": "ping"}))]
    #[case::error(json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}))]
    #[case::unknown(json!({"type": "future_event", "payload": 1}))]
    fn the_event_name_is_the_serialized_type_tag(#[case] wire: Value) {
        let event: MessagesStreamEvent = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(
            Some(event.as_ref()),
            wire.get("type").and_then(Value::as_str)
        );
    }

    #[rstest]
    #[case::a_start_short_of_every_field(json!({"type": "message_start", "message": {"stop_reason": null, "stop_sequence": null}}))]
    #[case::an_error_without_a_message(json!({"type": "error", "error": {"type": "overloaded_error"}}))]
    fn an_event_a_host_sends_short_reads_as_its_variant(#[case] wire: Value) {
        let event: MessagesStreamEvent = serde_json::from_value(wire.clone()).unwrap();

        assert!(!matches!(event, MessagesStreamEvent::Other(_)));
        assert_eq!(serde_json::to_value(event).unwrap(), wire);
    }

    #[rstest]
    #[case::unknown_delta(json!({"type": "future_delta", "payload": [1]}))]
    #[case::malformed_known_delta(json!({"type": "input_json_delta", "partial_json": 3}))]
    fn a_delta_that_is_not_recognized_is_still_a_content_block_delta(#[case] delta: Value) {
        let wire = json!({"type": "content_block_delta", "index": 0, "delta": delta});

        let event: MessagesStreamEvent = serde_json::from_value(wire.clone()).unwrap();

        assert_eq!(
            (&event, serde_json::to_value(&event).unwrap()),
            (
                &MessagesStreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: Recognized::Unrecognized(delta),
                },
                wire
            )
        );
    }

    #[rstest]
    #[case::unknown_event(json!({"type": "future_event", "payload": {"keep": null}}))]
    #[case::invalid_index(json!({"type": "content_block_stop", "index": "zero"}))]
    #[case::missing_delta(json!({"type": "content_block_delta", "index": 0}))]
    #[case::error_without_a_type(json!({"type": "error", "error": {"message": "busy"}}))]
    fn an_event_that_does_not_read_as_a_variant_is_kept_as_sent(#[case] wire: Value) {
        let event: MessagesStreamEvent = serde_json::from_value(wire.clone()).unwrap();

        assert!(matches!(event, MessagesStreamEvent::Other(_)));
        assert_eq!(serde_json::to_value(event).unwrap(), wire);
    }

    #[rstest]
    #[case::no_type(json!({"index": 0}))]
    #[case::type_that_is_not_text(json!({"type": 7}))]
    #[case::not_an_object(json!("ping"))]
    fn an_event_without_a_type_is_rejected(#[case] wire: Value) {
        assert!(serde_json::from_value::<MessagesStreamEvent>(wire).is_err());
    }
}
