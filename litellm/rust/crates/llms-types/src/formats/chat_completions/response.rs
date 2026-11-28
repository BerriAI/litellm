use serde_json::{Map, Value};

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionToolCallFunctionChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Map<String, Value>>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionToolCallChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatCompletionToolCallFunctionChunk,
    pub index: i64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionThinkingBlock {
    Thinking {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    RedactedThinking {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct CacheCreationTokenDetails {
    pub ephemeral_5m_input_tokens: Option<u64>,
    pub ephemeral_30m_input_tokens: Option<u64>,
    pub ephemeral_1h_input_tokens: Option<u64>,
    pub ephemeral_12h_input_tokens: Option<u64>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct PromptTokensDetails {
    pub cached_tokens: Option<u64>,
    pub text_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::recognized::known_or_absent")]
    pub cache_write_tokens: Option<u64>,
    pub cache_creation_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::recognized::known_or_absent")]
    pub cache_creation_token_details: Option<CacheCreationTokenDetails>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<u64>,
    pub text_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// OpenAI `usage` with the fields LiteLLM's Python `Usage` adds, in its order, so cost
/// tracking and a Python `ModelResponse` built from it see the same numbers on either path.
#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ChatCompletionsUsage {
    pub completion_tokens: u64,
    pub prompt_tokens: u64,
    pub total_tokens: u64,
    #[serde(default, deserialize_with = "crate::recognized::known_or_absent")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, deserialize_with = "crate::recognized::known_or_absent")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::recognized::known_or_absent")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::recognized::known_or_absent")]
    pub inference_geo: Option<String>,
    #[serde(default, deserialize_with = "crate::recognized::known_or_absent")]
    pub speed: Option<String>,
    #[serde(default, deserialize_with = "crate::recognized::known_or_absent")]
    pub service_tier: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionsChoiceMessage {
    pub role: String,
    // Whether an empty turn is `None` or `""` is the provider's choice, not a
    // shared invariant: Anthropic's transform ends on `merged_text or None`
    // while Converse assigns the joined string unconditionally. Each config
    // mirrors its own, so keep this optional and serialize it even when None.
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_blocks: Option<Vec<ChatCompletionThinkingBlock>>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionsChoice {
    pub index: u64,
    pub message: ChatCompletionsChoiceMessage,
    pub finish_reason: String,
}

/// The normalized response handed back to the host.
///
/// There is deliberately no `id`: Python mints the `chatcmpl-…` id on the
/// `ModelResponse` it already created, and echoing the provider's own id here
/// would change it. Pinned by `response_carries_no_id` in the Anthropic chat transformation tests.
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChatCompletionsResponse {
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionsChoice>,
    pub usage: ChatCompletionsUsage,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::ChatCompletionsUsage;
    use crate::test_support::round_trip;

    #[rstest]
    #[case::python_anthropic_usage(json!({
        "completion_tokens": 4, "prompt_tokens": 17, "total_tokens": 21,
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 4},
        "prompt_tokens_details": {
            "cached_tokens": 0, "text_tokens": 17, "cache_write_tokens": 0, "cache_creation_tokens": 0,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}
        },
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "inference_geo": "global", "speed": "standard", "service_tier": "standard"
    }))]
    #[case::openai_detail_keys(json!({
        "prompt_tokens": 5726, "completion_tokens": 4, "total_tokens": 5730,
        "prompt_tokens_details": {"cached_tokens": 5504, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0
        }
    }))]
    #[case::host_extension(json!({
        "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
        "prompt_cache_hit_tokens": 2, "future": {"k": null}
    }))]
    fn usage_round_trips_without_losing_fields(#[case] wire: Value) {
        round_trip::<ChatCompletionsUsage>(wire);
    }

    #[rstest]
    #[case::missing_prompt_tokens(json!({"completion_tokens": 1, "total_tokens": 1}))]
    #[case::missing_completion_tokens(json!({"prompt_tokens": 1, "total_tokens": 1}))]
    #[case::missing_total_tokens(json!({"prompt_tokens": 1, "completion_tokens": 1}))]
    fn usage_without_a_count_is_rejected(#[case] wire: Value) {
        assert!(serde_json::from_value::<ChatCompletionsUsage>(wire).is_err());
    }

    fn base_usage() -> Map<String, Value> {
        let usage = json!({
            "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 0, "text_tokens": 3}
        });
        usage.as_object().cloned().unwrap_or_default()
    }

    fn decoded(fields: Value) -> Result<ChatCompletionsUsage, serde_json::Error> {
        let wire: Map<String, Value> = base_usage()
            .into_iter()
            .chain(fields.as_object().cloned().unwrap_or_default())
            .collect();
        serde_json::from_value(Value::Object(wire))
    }

    #[rstest]
    #[case::service_tier(json!({"service_tier": 1}))]
    #[case::speed(json!({"speed": 2}))]
    #[case::inference_geo(json!({"inference_geo": ["us"]}))]
    #[case::cache_read_input_tokens(json!({"cache_read_input_tokens": "many"}))]
    #[case::cache_creation_input_tokens(json!({"cache_creation_input_tokens": -1}))]
    #[case::completion_tokens_details(json!({"completion_tokens_details": {"reasoning_tokens": "x"}}))]
    #[case::cache_write_tokens(json!({
        "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 0, "text_tokens": 3, "cache_write_tokens": "x"}
    }))]
    #[case::cache_creation_token_details(json!({
        "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_tokens": 0, "text_tokens": 3, "cache_creation_token_details": 3}
    }))]
    fn field_the_old_contract_ignored_stays_absent_when_it_has_another_type(#[case] fields: Value) {
        let usage = decoded(fields).map(serde_json::to_value);
        assert_eq!(
            usage.ok().and_then(Result::ok),
            Some(Value::Object(base_usage()))
        );
    }

    #[rstest]
    #[case::cached_tokens(json!({"prompt_tokens_details": {"cached_tokens": "x"}}))]
    #[case::text_tokens(json!({"prompt_tokens_details": {"text_tokens": -1}}))]
    #[case::details(json!({"prompt_tokens_details": 3}))]
    fn detail_the_old_contract_typed_is_still_rejected_with_another_type(#[case] fields: Value) {
        assert!(decoded(fields).is_err());
    }
}
