use serde_json::{Map, Value};

use crate::formats::chat_completions::{
    ChatCompletionThinkingBlock, ChatCompletionToolCallChunk, ChatCompletionsUsage,
};

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ChatCompletionDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionToolCallChunk>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_blocks: Option<Vec<ChatCompletionThinkingBlock>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionStreamingChoice {
    pub index: u64,
    pub delta: ChatCompletionDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

/// Compatible hosts omit `id`, `created` and `object` on chunks, and Python's
/// `ModelResponseStream` fills them in, so none of the envelope fields is required here.
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ChatCompletionChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatCompletionStreamingChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatCompletionsUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest]
    #[case::openai(json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi", "future": 1}, "finish_reason": null, "logprobs": null}],
        "usage": null
    }))]
    #[case::usage_only(json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [],
        "usage": {
            "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 0, "text_tokens": 1}
        }
    }))]
    #[case::bare_host(json!({"choices": [{"index": 0, "delta": {"content": "hi"}}]}))]
    #[case::host_extensions(json!({"id": "x", "choices": [], "citations": ["a"], "future": {"k": null}}))]
    fn chunks_round_trip_without_losing_fields(#[case] wire: Value) {
        let chunk: ChatCompletionChunk = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(chunk).unwrap(), without_nulls(wire));
    }

    /// Typed optional fields serialize by omission; extension fields keep their nulls.
    fn without_nulls(value: Value) -> Value {
        let Value::Object(fields) = value else {
            return value;
        };
        Value::Object(
            fields
                .into_iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| match (key.as_str(), value) {
                    ("choices", Value::Array(choices)) => (
                        key,
                        Value::Array(choices.into_iter().map(without_nulls).collect()),
                    ),
                    (_, value) => (key, value),
                })
                .collect(),
        )
    }

    #[rstest]
    #[case::choices_not_a_list(json!({"choices": {}}))]
    #[case::choice_without_delta(json!({"choices": [{"index": 0}]}))]
    #[case::created_not_a_number(json!({"created": "now", "choices": []}))]
    #[case::non_object(json!([]))]
    fn malformed_chunks_are_rejected(#[case] wire: Value) {
        assert!(serde_json::from_value::<ChatCompletionChunk>(wire).is_err());
    }
}
