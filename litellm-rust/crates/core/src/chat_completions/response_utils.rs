//! Response normalization shared by every chat completions provider config.

use std::time::{SystemTime, UNIX_EPOCH};

use super::types::{ChatCompletionsUsage, PromptTokensDetails};

/// OpenAI finish reasons, mirroring Python's `_FINISH_REASON_MAP` for the
/// reasons the providers on this route can emit. Python warns and falls back to
/// `stop` for anything unmapped, so do the same.
const FINISH_REASONS: &[(&str, &str)] = &[
    ("end_turn", "stop"),
    ("stop_sequence", "stop"),
    ("max_tokens", "length"),
    ("refusal", "content_filter"),
    ("compaction", "length"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_every_reason_the_route_can_observe() {
        assert_eq!(finish_reason_for("end_turn"), "stop");
        assert_eq!(finish_reason_for("stop_sequence"), "stop");
        assert_eq!(finish_reason_for("max_tokens"), "length");
        assert_eq!(finish_reason_for("refusal"), "content_filter");
        assert_eq!(finish_reason_for("guardrail_intervened"), "content_filter");
        // Converse emits these two, and folding them into `stop` would report a
        // filtered completion as a normal one.
        assert_eq!(finish_reason_for("content_filtered"), "content_filter");
        assert_eq!(finish_reason_for("content_filter"), "content_filter");
    }

    #[test]
    fn defaults_an_unmapped_reason_to_stop_like_python() {
        // Python warns and falls back to `stop` for a reason its own map does
        // not carry, so only a reason absent from `_FINISH_REASON_MAP` belongs
        // here.
        assert_eq!(finish_reason_for("something_new"), "stop");
        assert_eq!(finish_reason_for(""), "stop");
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
