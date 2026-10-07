use litellm_llms_types::formats::chat_completions::{ChatCompletionsUsage, PromptTokensDetails};

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
