#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_missing_cache_read_rate_resolves_to_input_rate

use litellm_cost::generic_usage::{
    get_billable_input_tokens, parse_completion_tokens_details, parse_prompt_tokens_details,
};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case(100, 20, 80)]
#[case(100, 0, 100)]
#[case(10, 20, -10)]
fn get_billable_input_tokens_subtracts_cached_tokens(
    #[case] prompt_tokens: u64,
    #[case] cache_hit_tokens: u64,
    #[case] expected: i128,
) {
    assert_eq!(
        get_billable_input_tokens(prompt_tokens, cache_hit_tokens),
        expected
    );
}

#[rstest]
fn parse_prompt_tokens_details_subtracts_each_cached_modality_once() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "prompt_tokens_details": {
            "cached_tokens": 200,
            "cached_tokens_details": {"audio_tokens": 80, "text_tokens": 90, "image_tokens": 60},
            "text_tokens": 500,
            "audio_tokens": 250,
            "image_tokens": 150,
            "video_tokens": 25
        }
    }}))
    .unwrap()
    .unwrap();
    let details = parse_prompt_tokens_details(&usage);
    assert_eq!(details.cache_hit_tokens, 200);
    assert_eq!(details.cache_hit_audio_tokens, 80);
    assert_eq!(details.text_tokens, 410);
    assert_eq!(details.audio_tokens, 170);
    assert_eq!(details.image_tokens, 120);
    assert_eq!(details.video_tokens, 25);
}

#[rstest]
#[case(30, 20, 30)]
#[case(0, 20, 0)]
fn parse_prompt_tokens_details_prefers_the_cache_write_alias_even_at_zero(
    #[case] write: u64,
    #[case] creation: u64,
    #[case] expected: u64,
) {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_tokens_details": {
            "cache_write_tokens": write,
            "cache_creation_tokens": creation,
            "text_tokens": 70
        }
    }}))
    .unwrap()
    .unwrap();
    assert_eq!(
        parse_prompt_tokens_details(&usage).cache_creation_tokens,
        expected
    );
}

#[rstest]
fn parse_prompt_tokens_details_preserves_non_token_usage() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_tokens_details": {
            "character_count": 800,
            "image_count": 2,
            "video_length_seconds": 3.5,
            "audio_length_seconds": 1.25,
            "query_count": 4
        }
    }}))
    .unwrap()
    .unwrap();
    let details = parse_prompt_tokens_details(&usage);
    assert_eq!(details.character_count, 800);
    assert_eq!(details.image_count, 2);
    assert_eq!(details.video_length_seconds, 3.5);
    assert_eq!(details.audio_length_seconds, 1.25);
    assert_eq!(details.query_count, 4);
}

#[rstest]
#[case(90, 20, 10, 0, 0, 70)]
#[case(70, 20, 10, 0, 0, 70)]
#[case(0, 20, 10, 0, 0, 0)]
fn parse_completion_tokens_details_partitions_nested_reasoning(
    #[case] text: u64,
    #[case] reasoning: u64,
    #[case] audio: u64,
    #[case] image: u64,
    #[case] video: u64,
    #[case] expected_text: u64,
) {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 10,
        "completion_tokens": 100,
        "total_tokens": 110,
        "completion_tokens_details": {
            "text_tokens": text,
            "reasoning_tokens": reasoning,
            "audio_tokens": audio,
            "image_tokens": image,
            "video_tokens": video
        }
    }}))
    .unwrap()
    .unwrap();
    let details = parse_completion_tokens_details(&usage);
    assert_eq!(details.text_tokens, expected_text);
    assert_eq!(details.reasoning_tokens, reasoning);
    assert_eq!(details.audio_tokens, audio);
}
