#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/responses/test_responses_utils.py::TestResponseAPILoggingUtils
use litellm_cost::error::CostError;

use litellm_cost::generic_cost::calculate_generic_cost_from_model_info_with_region;
use litellm_cost::responses_usage::{
    is_response_api_usage, text_tokens_without_nested_reasoning,
    transform_response_api_usage_to_chat_usage,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case(json!({"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}), 10, 20, 30)]
#[case(json!({"input_tokens": 15, "output_tokens": 25, "total_tokens": null}), 15, 25, 40)]
fn transform_response_api_usage_to_chat_usage_normalizes_totals(
    #[case] raw: Value,
    #[case] input: u64,
    #[case] output: u64,
    #[case] total: u64,
) {
    let usage = transform_response_api_usage_to_chat_usage(&raw).unwrap();
    assert_eq!(usage.prompt_tokens, input);
    assert_eq!(usage.completion_tokens, output);
    assert_eq!(usage.total_tokens, total);
}

#[rstest]
fn transform_response_api_usage_to_chat_usage_preserves_modalities() {
    let raw = json!({
        "input_tokens": 100,
        "output_tokens": 200,
        "input_tokens_details": {
            "cached_tokens": 50,
            "audio_tokens": 10,
            "image_tokens": 20,
            "text_tokens": 20
        },
        "output_tokens_details": {
            "reasoning_tokens": 30,
            "image_tokens": 100,
            "text_tokens": 50,
            "audio_tokens": 20
        }
    });
    let usage = transform_response_api_usage_to_chat_usage(&raw).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    let completion = usage.completion_tokens_details.unwrap();
    assert_eq!(prompt.cached_tokens, 50);
    assert_eq!(prompt.audio_tokens, Some(10));
    assert_eq!(prompt.image_tokens, Some(20));
    assert_eq!(prompt.text_tokens, Some(20));
    assert_eq!(completion.reasoning_tokens, Some(30));
    assert_eq!(completion.image_tokens, Some(100));
    assert_eq!(completion.audio_tokens, Some(20));
    assert_eq!(completion.text_tokens, Some(50));
}

#[rstest]
#[case("cache_write_tokens", 10059, Some(10059))]
#[case("cache_creation_tokens", 500, None)]
#[case("cache_creation_input_tokens", 500, None)]
fn transform_response_api_usage_to_chat_usage_keeps_only_cache_write_tokens(
    #[case] key: &str,
    #[case] tokens: u64,
    #[case] expected: Option<u64>,
) {
    let raw = json!({
        "input_tokens": 10062,
        "output_tokens": 16,
        "input_tokens_details": {(key): tokens, "cached_tokens": 0}
    });
    let usage = transform_response_api_usage_to_chat_usage(&raw).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    assert_eq!(prompt.cache_write_tokens, expected);
    assert_eq!(prompt.cache_creation_tokens, expected);
    assert_eq!(prompt.cached_tokens, 0);
}

#[rstest]
#[case(
    json!({"input_tokens": 10, "output_tokens": 20, "input_token_details": {"text_tokens": 8, "audio_tokens": 2}, "output_token_details": {"text_tokens": 12, "audio_tokens": 8}}),
    8,
    12
)]
#[case(
    json!({"input_tokens": 10, "output_tokens": 20, "input_tokens_details": {"text_tokens": 10}, "output_tokens_details": {"text_tokens": 20}, "input_token_details": {"text_tokens": 1}, "output_token_details": {"text_tokens": 2}}),
    10,
    20
)]
fn transform_response_api_usage_to_chat_usage_prefers_plural_details(
    #[case] raw: Value,
    #[case] prompt_text: u64,
    #[case] completion_text: u64,
) {
    let usage = transform_response_api_usage_to_chat_usage(&raw).unwrap();
    assert_eq!(
        usage.prompt_tokens_details.unwrap().text_tokens,
        Some(prompt_text)
    );
    assert_eq!(
        usage.completion_tokens_details.unwrap().text_tokens,
        Some(completion_text)
    );
}

#[rstest]
#[case(70, 70, 52, 0, 18)]
#[case(70, 39, 23, 31, 16)]
#[case(20, 12, 5, 0, 12)]
fn text_tokens_without_nested_reasoning_matches_python_cases(
    #[case] completion: u64,
    #[case] text: u64,
    #[case] reasoning: u64,
    #[case] other: u64,
    #[case] expected: u64,
) {
    assert_eq!(
        text_tokens_without_nested_reasoning(completion, text, reasoning, other),
        expected
    );
}

#[rstest]
fn transform_response_api_usage_to_chat_usage_preserves_cached_modality_split() {
    let raw = json!({
        "input_tokens": 283,
        "output_tokens": 0,
        "input_token_details": {
            "text_tokens": 116,
            "audio_tokens": 167,
            "cached_tokens": 192,
            "cached_tokens_details": {"text_tokens": 64, "audio_tokens": 128}
        }
    });
    let usage = transform_response_api_usage_to_chat_usage(&raw).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    let cached = prompt.cached_tokens_details.unwrap();
    assert_eq!(prompt.cached_tokens, 192);
    assert_eq!(cached.text_tokens, Some(64));
    assert_eq!(cached.audio_tokens, Some(128));
}

#[rstest]
fn transform_response_api_usage_to_chat_usage_keeps_provider_extras_and_cost() {
    let raw = json!({
        "input_tokens": 100,
        "output_tokens": 20,
        "cost": 0.5,
        "server_side_tool_usage_details": {"web_search_calls": 2},
        "prompt_tokens": 100,
        "prompt_tokens_details": {"text_tokens": 100}
    });
    let usage = transform_response_api_usage_to_chat_usage(&raw).unwrap();
    assert_eq!(usage.cost, Some(0.5));
    assert_eq!(
        usage.extra.get("server_side_tool_usage_details"),
        Some(&json!({"web_search_calls": 2}))
    );
    assert_eq!(usage.extra.get("prompt_tokens"), None);
    assert_eq!(usage.prompt_tokens_details, None);
}

#[rstest]
fn is_response_api_usage_requires_both_token_fields() {
    assert!(is_response_api_usage(
        &json!({"input_tokens": 1, "output_tokens": 2})
    ));
    assert!(!is_response_api_usage(&json!({"input_tokens": 1})));
    assert_eq!(
        transform_response_api_usage_to_chat_usage(
            &json!({"prompt_tokens": 1, "completion_tokens": 2})
        ),
        Err(CostError::InvalidShape)
    );
}

#[rstest]
fn transformed_responses_usage_reaches_cache_aware_token_calculation() {
    let raw = json!({
        "input_tokens": 1000,
        "output_tokens": 100,
        "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100}
    });
    let usage = transform_response_api_usage_to_chat_usage(&raw).unwrap();
    let model_info = json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_read_input_token_cost": 0.2e-6,
        "cache_creation_input_token_cost": 1.25e-6
    });
    let (input, output) = calculate_generic_cost_from_model_info_with_region(
        &usage,
        &model_info,
        None,
        false,
        None,
        None,
        "2026-01-01T12:00Z".parse().unwrap(),
    );
    assert!((input - (700.0 * 1e-6 + 200.0 * 0.2e-6 + 100.0 * 1.25e-6)).abs() < 1e-12);
    assert!((output - 100.0 * 2e-6).abs() < 1e-12);
}

#[rstest]
fn transform_response_api_usage_to_chat_usage_drops_fields_python_does_not_copy() {
    let raw = json!({
        "input_tokens": 100,
        "output_tokens": 50,
        "input_tokens_details": {
            "cached_tokens": 3,
            "web_search_requests": 2,
            "character_count": 5,
            "image_count": 1,
            "query_count": 4,
            "audio_length_seconds": 1.5,
            "cache_creation_token_details": {"ephemeral_5m_input_tokens": 3}
        },
        "output_tokens_details": {"reasoning_tokens": 4, "video_tokens": 11}
    });
    let usage = transform_response_api_usage_to_chat_usage(&raw).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    assert_eq!(prompt.cached_tokens, 3);
    assert_eq!(prompt.web_search_requests, Some(2));
    assert_eq!(prompt.character_count, None);
    assert_eq!(prompt.image_count, None);
    assert_eq!(prompt.query_count, None);
    assert_eq!(prompt.audio_length_seconds, None);
    assert_eq!(prompt.cache_creation_token_details, None);
    let completion = usage.completion_tokens_details.unwrap();
    assert_eq!(completion.reasoning_tokens, Some(4));
    assert_eq!(completion.video_tokens, None);
}

#[rstest]
#[case::integral_float_with_total(json!({"input_tokens": 10.0, "output_tokens": 5, "total_tokens": 15}), Ok(10))]
#[case::numeric_string_with_total(json!({"input_tokens": "10", "output_tokens": 5, "total_tokens": 15}), Ok(10))]
#[case::boolean_without_total(json!({"input_tokens": true, "output_tokens": 5}), Ok(1))]
#[case::float_needs_a_total(json!({"input_tokens": 10.0, "output_tokens": 5}), Err(CostError::InvalidUsage))]
#[case::string_needs_a_total(json!({"input_tokens": "10", "output_tokens": 5}), Err(CostError::InvalidUsage))]
#[case::fractional_float(json!({"input_tokens": 10.5, "output_tokens": 5, "total_tokens": 15}), Err(CostError::InvalidUsage))]
fn transform_response_api_usage_to_chat_usage_validates_counts_like_response_api_usage(
    #[case] raw: Value,
    #[case] expected: Result<u64, CostError>,
) {
    assert_eq!(
        transform_response_api_usage_to_chat_usage(&raw).map(|usage| usage.prompt_tokens),
        expected
    );
}
