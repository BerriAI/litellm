//! Response normalization shared by every chat completions provider config.

use std::time::{SystemTime, UNIX_EPOCH};

use litellm_llms_types::formats::chat_completions::{
    ChatCompletionsUsage, CompletionTokensDetails, PromptTokensDetails,
};

/// OpenAI finish reasons, mirroring Python's `_FINISH_REASON_MAP` for the
/// reasons the providers on this route can emit. Python warns and falls back to
/// `stop` for anything unmapped, so do the same.
const FINISH_REASONS: &[(&str, &str)] = &[
    ("end_turn", "stop"),
    ("stop_sequence", "stop"),
    ("max_tokens", "length"),
    ("refusal", "content_filter"),
    ("compaction", "length"),
    ("pause_turn", "length"),
    ("model_context_window_exceeded", "length"),
    ("guardrail_intervened", "content_filter"),
    ("content_filtered", "content_filter"),
    ("content_filter", "content_filter"),
    ("stop", "stop"),
    ("length", "length"),
];

pub fn finish_reason_for(provider_reason: &str) -> &'static str {
    FINISH_REASONS
        .iter()
        .find(|(reason, _)| *reason == provider_reason)
        .map_or("stop", |(_, mapped)| *mapped)
}

/// Python's Anthropic and Bedrock usage builders: cache tokens fold into `prompt_tokens`, the
/// split goes under `prompt_tokens_details`, and the top-level cache counts are echoed.
pub fn usage_from_parts(
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_creation_tokens: u64,
) -> ChatCompletionsUsage {
    let prompt_tokens = input_tokens + cache_read_tokens + cache_creation_tokens;
    ChatCompletionsUsage {
        completion_tokens: output_tokens,
        prompt_tokens,
        total_tokens: prompt_tokens + output_tokens,
        completion_tokens_details: Some(CompletionTokensDetails {
            reasoning_tokens: Some(0),
            text_tokens: Some(output_tokens),
            ..CompletionTokensDetails::default()
        }),
        prompt_tokens_details: Some(PromptTokensDetails {
            cached_tokens: Some(cache_read_tokens),
            text_tokens: Some(input_tokens),
            cache_write_tokens: Some(cache_creation_tokens),
            cache_creation_tokens: Some(cache_creation_tokens),
            ..PromptTokensDetails::default()
        }),
        cache_creation_input_tokens: Some(cache_creation_tokens),
        cache_read_input_tokens: Some(cache_read_tokens),
        ..ChatCompletionsUsage::default()
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

pub fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    #[case::end_turn("end_turn", "stop")]
    #[case::stop_sequence("stop_sequence", "stop")]
    #[case::max_tokens("max_tokens", "length")]
    #[case::pause_turn("pause_turn", "length")]
    #[case::model_context_window_exceeded("model_context_window_exceeded", "length")]
    #[case::refusal("refusal", "content_filter")]
    #[case::guardrail_intervened("guardrail_intervened", "content_filter")]
    #[case::content_filtered("content_filtered", "content_filter")]
    #[case::content_filter("content_filter", "content_filter")]
    fn maps_every_reason_the_route_can_observe(#[case] reason: &str, #[case] expected: &str) {
        assert_eq!(finish_reason_for(reason), expected);
    }

    #[rstest]
    #[case::unknown("something_new")]
    #[case::empty("")]
    fn defaults_an_unmapped_reason_to_stop_like_python(#[case] reason: &str) {
        assert_eq!(finish_reason_for(reason), "stop");
    }

    #[rstest]
    #[case::cache_read_and_write(10, 4, 7, 3, json!({
        "completion_tokens": 4, "prompt_tokens": 20, "total_tokens": 24,
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 4},
        "prompt_tokens_details": {"cached_tokens": 7, "text_tokens": 10, "cache_write_tokens": 3, "cache_creation_tokens": 3},
        "cache_creation_input_tokens": 3, "cache_read_input_tokens": 7
    }))]
    #[case::no_cache(12, 5, 0, 0, json!({
        "completion_tokens": 5, "prompt_tokens": 12, "total_tokens": 17,
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 5},
        "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 12, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0
    }))]
    fn builds_the_usage_python_builds_from_provider_counts(
        #[case] input: u64,
        #[case] output: u64,
        #[case] cache_read: u64,
        #[case] cache_creation: u64,
        #[case] expected: serde_json::Value,
    ) {
        let usage = usage_from_parts(input, output, cache_read, cache_creation);
        assert_eq!(serde_json::to_value(usage).unwrap(), expected);
    }
}
