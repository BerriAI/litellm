//! Response normalization shared by every chat completions provider config.

use std::time::{SystemTime, UNIX_EPOCH};

use litellm_types::utils::{ChatCompletionsUsage, PromptTokensDetails};

enum ProviderFinishReason {
    EndTurn,
    StopSequence,
    MaxTokens,
    Refusal,
    Compaction,
    GuardrailIntervened,
    ContentFiltered,
    ContentFilter,
    Stop,
    Length,
}

impl ProviderFinishReason {
    fn parse(reason: &str) -> Option<Self> {
        match reason {
            "end_turn" => Some(Self::EndTurn),
            "stop_sequence" => Some(Self::StopSequence),
            "max_tokens" => Some(Self::MaxTokens),
            "refusal" => Some(Self::Refusal),
            "compaction" => Some(Self::Compaction),
            "guardrail_intervened" => Some(Self::GuardrailIntervened),
            "content_filtered" => Some(Self::ContentFiltered),
            "content_filter" => Some(Self::ContentFilter),
            "stop" => Some(Self::Stop),
            "length" => Some(Self::Length),
            _ => None,
        }
    }
}

enum FinishReason {
    Stop,
    Length,
    ContentFilter,
}

impl FinishReason {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ContentFilter => "content_filter",
        }
    }
}

pub fn finish_reason_for(provider_reason: Option<&str>) -> &'static str {
    match provider_reason.and_then(ProviderFinishReason::parse) {
        Some(
            ProviderFinishReason::MaxTokens
            | ProviderFinishReason::Compaction
            | ProviderFinishReason::Length,
        ) => FinishReason::Length,
        Some(
            ProviderFinishReason::Refusal
            | ProviderFinishReason::GuardrailIntervened
            | ProviderFinishReason::ContentFiltered
            | ProviderFinishReason::ContentFilter,
        ) => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
    .as_str()
}

/// Python folds cache tokens into `prompt_tokens` and reports the split under
/// `prompt_tokens_details`; mirror that so cost tracking agrees on both paths.
pub fn usage_from_parts(
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_creation_tokens: u64,
) -> ChatCompletionsUsage {
    let prompt_tokens = input_tokens + cache_read_tokens + cache_creation_tokens;
    ChatCompletionsUsage {
        prompt_tokens,
        completion_tokens: output_tokens,
        total_tokens: prompt_tokens + output_tokens,
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: cache_read_tokens,
            cache_creation_tokens,
            text_tokens: input_tokens,
        },
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
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::end_turn(Some("end_turn"), "stop")]
    #[case::stop_sequence(Some("stop_sequence"), "stop")]
    #[case::max_tokens(Some("max_tokens"), "length")]
    #[case::compaction(Some("compaction"), "length")]
    #[case::refusal(Some("refusal"), "content_filter")]
    #[case::guardrail(Some("guardrail_intervened"), "content_filter")]
    #[case::content_filtered(Some("content_filtered"), "content_filter")]
    #[case::content_filter(Some("content_filter"), "content_filter")]
    #[case::stop(Some("stop"), "stop")]
    #[case::length(Some("length"), "length")]
    #[case::unknown(Some("something_new"), "stop")]
    #[case::empty(Some(""), "stop")]
    #[case::missing(None, "stop")]
    fn normalizes_provider_finish_reasons(#[case] input: Option<&str>, #[case] expected: &str) {
        assert_eq!(finish_reason_for(input), expected);
    }

    #[test]
    fn folds_cache_tokens_into_prompt_tokens() {
        let usage = usage_from_parts(10, 4, 7, 3);
        assert_eq!(usage.prompt_tokens, 20);
        assert_eq!(usage.completion_tokens, 4);
        assert_eq!(usage.total_tokens, 24);
        assert_eq!(usage.prompt_tokens_details.cached_tokens, 7);
        assert_eq!(usage.prompt_tokens_details.cache_creation_tokens, 3);
        assert_eq!(usage.prompt_tokens_details.text_tokens, 10);
    }

    #[test]
    fn reports_raw_input_tokens_when_no_cache_is_involved() {
        let usage = usage_from_parts(12, 5, 0, 0);
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.total_tokens, 17);
        assert_eq!(usage.prompt_tokens_details.text_tokens, 12);
    }
}
