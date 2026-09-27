use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicResponseContentBlock {
    Known(KnownContentBlock),
    Other(Value),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KnownContentBlock {
    Text {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<Value>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    RedactedThinking {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ToolUse {
        #[serde(default)]
        input: Value,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ServerToolUse {
        #[serde(default)]
        input: Value,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnthropicUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<AnthropicResponseContentBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AnthropicUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn response(
        stop_reason: Option<&str>,
        stop_sequence: Option<&str>,
        usage: Option<AnthropicUsage>,
        container: Option<Value>,
    ) -> AnthropicMessagesResponse {
        AnthropicMessagesResponse {
            id: "msg_1".to_string(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude".to_string(),
            content: vec![],
            stop_reason: stop_reason.map(str::to_string),
            stop_sequence: stop_sequence.map(str::to_string),
            usage,
            container,
            extra: Map::new(),
        }
    }

    #[rstest]
    #[case::turn_in_progress(None, None, json!(null), json!(null))]
    #[case::ended_on_end_turn(Some("end_turn"), None, json!("end_turn"), json!(null))]
    #[case::ended_on_stop_sequence(Some("stop_sequence"), Some("###"), json!("stop_sequence"), json!("###"))]
    fn stop_fields_are_always_serialized(
        #[case] stop_reason: Option<&str>,
        #[case] stop_sequence: Option<&str>,
        #[case] expected_reason: Value,
        #[case] expected_sequence: Value,
    ) {
        let body: Value = serde_json::to_value(response(stop_reason, stop_sequence, None, None))
            .expect("serializable");
        assert_eq!(body.get("stop_reason"), Some(&expected_reason));
        assert_eq!(body.get("stop_sequence"), Some(&expected_sequence));
    }

    #[rstest]
    #[case::absent(None, None)]
    #[case::present(Some(AnthropicUsage { input_tokens: Some(1), ..AnthropicUsage::default() }), Some(json!({"id": "c_1"})))]
    fn usage_and_container_are_omitted_only_when_none(
        #[case] usage: Option<AnthropicUsage>,
        #[case] container: Option<Value>,
    ) {
        let body: Value =
            serde_json::to_value(response(None, None, usage.clone(), container.clone()))
                .expect("serializable");
        assert_eq!(
            body.get("usage").cloned(),
            usage.map(|usage| serde_json::to_value(usage).unwrap())
        );
        assert_eq!(body.get("container").cloned(), container);
    }

    #[rstest]
    #[case::text_with_citations(json!({"type": "text", "text": "hi", "citations": [{"type": "char_location"}]}))]
    #[case::thinking(json!({"type": "thinking", "thinking": "hm", "signature": "sig"}))]
    #[case::thinking_with_non_string_signature(json!({"type": "thinking", "thinking": "hm", "signature": 7}))]
    #[case::redacted_thinking(json!({"type": "redacted_thinking", "data": "abc"}))]
    #[case::tool_use(json!({"type": "tool_use", "id": "t1", "name": "lookup", "input": {"q": "x"}, "caller": {"type": "direct"}}))]
    #[case::server_tool_use(json!({"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "y"}}))]
    #[case::unknown_type(json!({"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}))]
    #[case::malformed_text(json!({"type": "text", "text": null}))]
    #[case::non_object_element(json!("stray"))]
    fn content_blocks_round_trip_verbatim(#[case] block: Value) {
        let decoded: AnthropicResponseContentBlock = serde_json::from_value(block.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), block);
    }

    #[rstest]
    #[case::malformed_text(json!({"type": "text", "text": null}))]
    #[case::unknown_type(json!({"type": "image_result"}))]
    #[case::non_object_element(json!(3))]
    fn unrecognized_blocks_are_kept_as_other(#[case] block: Value) {
        let decoded: AnthropicResponseContentBlock = serde_json::from_value(block.clone()).unwrap();
        assert_eq!(decoded, AnthropicResponseContentBlock::Other(block));
    }

    #[rstest]
    #[case::null_cache_counter(json!({"input_tokens": 3, "output_tokens": 4, "cache_creation_input_tokens": null}))]
    #[case::only_extras(json!({"cache_read_input_tokens": 7, "server_tool_use": {"web_search_requests": 1}}))]
    #[case::empty(json!({}))]
    fn usage_round_trips_verbatim(#[case] usage: Value) {
        let decoded: AnthropicUsage = serde_json::from_value(usage.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), usage);
    }
}
