use litellm_llms::base_llm::chat::normalization::{finish_reason_for, usage_from_parts};
use rstest::rstest;

#[rstest]
#[case::end_turn("end_turn", "stop")]
#[case::stop_sequence("stop_sequence", "stop")]
#[case::max_tokens("max_tokens", "length")]
#[case::refusal("refusal", "content_filter")]
#[case::compaction("compaction", "length")]
#[case::guardrail_intervened("guardrail_intervened", "content_filter")]
#[case::content_filtered("content_filtered", "content_filter")]
#[case::content_filter("content_filter", "content_filter")]
#[case::stop("stop", "stop")]
#[case::length("length", "length")]
#[case::unknown("something_new", "stop")]
#[case::empty("", "stop")]
fn normalizes_finish_reasons(#[case] reason: &str, #[case] expected: &str) {
    assert_eq!(finish_reason_for(reason), expected);
}

#[rstest]
#[case::both_cache_kinds(10, 4, 7, 3)]
#[case::without_cache(12, 5, 0, 0)]
#[case::cache_read_only(10, 4, 7, 0)]
#[case::cache_creation_only(10, 4, 0, 3)]
#[case::zero_usage(0, 0, 0, 0)]
fn folds_cache_tokens_into_prompt_usage(
    #[case] input_tokens: u64,
    #[case] output_tokens: u64,
    #[case] cache_read_tokens: u64,
    #[case] cache_creation_tokens: u64,
) {
    let usage = usage_from_parts(
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_creation_tokens,
    );
    assert_eq!(
        usage.prompt_tokens,
        input_tokens + cache_read_tokens + cache_creation_tokens
    );
    assert_eq!(usage.completion_tokens, output_tokens);
    assert_eq!(usage.total_tokens, usage.prompt_tokens + output_tokens);
    assert_eq!(usage.prompt_tokens_details.cached_tokens, cache_read_tokens);
    assert_eq!(
        usage.prompt_tokens_details.cache_creation_tokens,
        cache_creation_tokens
    );
    assert_eq!(usage.prompt_tokens_details.text_tokens, input_tokens);
}
