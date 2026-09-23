#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/responses/test_responses_utils.py::TestResponseAPILoggingUtils

use litellm_cost::responses_usage::{
    UsageError, is_response_api_usage, text_tokens_without_nested_reasoning,
    transform_response_api_usage_to_chat_usage,
};
use litellm_cost::{Pricing, Rate, Rates, Request, ServiceTier, ThresholdPolicy, calculate};
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
#[case("cache_write_tokens", 10059)]
#[case("cache_creation_tokens", 500)]
fn transform_response_api_usage_to_chat_usage_mirrors_cache_write_aliases(
    #[case] key: &str,
    #[case] tokens: u64,
) {
    let raw = json!({
        "input_tokens": 10062,
        "output_tokens": 16,
        "input_tokens_details": {(key): tokens, "cached_tokens": 0}
    });
    let usage = transform_response_api_usage_to_chat_usage(&raw).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    assert_eq!(prompt.cache_write_tokens, Some(tokens));
    assert_eq!(prompt.cache_creation_tokens, Some(tokens));
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
        Err(UsageError::InvalidShape)
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
    let pricing = Pricing {
        standard: Rates {
            input: Rate::Value(1e-6),
            output: Rate::Value(2e-6),
            cache_read: Rate::Value(0.2e-6),
            cache_write: Rate::Value(1.25e-6),
            cache_write_1h: Rate::Missing,
        },
        tiers: &[],
        thresholds: &[],
        off_peak: None,
    };
    let request = Request {
        usage: usage.token_usage(),
        service_tier: ServiceTier::Standard,
        threshold_policy: ThresholdPolicy::Exclusive,
        region_multiplier: None,
        billed_at_utc_minute: None,
    };
    let cost = calculate(&pricing, &request).unwrap();
    assert!((cost.input() - (700.0 * 1e-6 + 200.0 * 0.2e-6 + 100.0 * 1.25e-6)).abs() < 1e-12);
    assert!((cost.output() - 100.0 * 2e-6).abs() < 1e-12);
}
