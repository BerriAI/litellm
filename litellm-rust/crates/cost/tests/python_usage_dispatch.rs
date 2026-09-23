#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_usage_object_transformation.py

use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case(json!({}), None)]
#[case(json!({"usage": null}), None)]
fn get_usage_object_returns_none_when_usage_is_missing(
    #[case] response: Value,
    #[case] expected: Option<u64>,
) {
    assert_eq!(
        get_usage_object(&response)
            .unwrap()
            .map(|usage| usage.prompt_tokens),
        expected
    );
}

#[rstest]
#[case(json!({"usage": {"input_tokens": 3, "output_tokens": 5, "cache_read_input_tokens": 4}}), 7, 4, 5)]
#[case(json!({"usage": {"input_tokens": 7, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 4}}}), 7, 4, 5)]
#[case(json!({"usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12, "prompt_tokens_details": {"cached_tokens": 4}}}), 7, 4, 5)]
fn get_usage_object_keeps_cache_conventions_distinct(
    #[case] response: Value,
    #[case] expected_prompt: u64,
    #[case] expected_cached: u64,
    #[case] expected_output: u64,
) {
    let usage = get_usage_object(&response).unwrap().unwrap();
    assert_eq!(usage.prompt_tokens, expected_prompt);
    assert_eq!(usage.completion_tokens, expected_output);
    assert_eq!(
        usage.prompt_tokens_details.unwrap().cached_tokens,
        expected_cached
    );
}

#[rstest]
fn get_usage_object_maps_chat_cache_fields_into_prompt_details() {
    let response = json!({
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}
        }
    });
    let usage = get_usage_object(&response).unwrap().unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    assert_eq!(prompt.cached_tokens, 200);
    assert_eq!(prompt.cache_write_tokens, Some(100));
    assert_eq!(prompt.cache_creation_tokens, Some(100));
}

#[rstest]
fn get_usage_object_preserves_chat_reasoning_and_provider_fields() {
    let response = json!({
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "reasoning_tokens": 5,
            "server_side_tool_usage_details": {"web_search_calls": 2}
        }
    });
    let usage = get_usage_object(&response).unwrap().unwrap();
    let completion = usage.completion_tokens_details.unwrap();
    assert_eq!(completion.reasoning_tokens, Some(5));
    assert_eq!(completion.text_tokens, Some(15));
    assert_eq!(
        usage.extra.get("server_side_tool_usage_details"),
        Some(&json!({"web_search_calls": 2}))
    );
}

#[rstest]
fn get_usage_object_maps_interactions_shape() {
    let response = json!({"usage": {"total_input_tokens": 100, "total_output_tokens": 20}});
    let usage = get_usage_object(&response).unwrap().unwrap();
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 20);
    assert_eq!(usage.total_tokens, 120);
}

#[test]
fn chat_usage_coerces_numeric_strings_and_bools_like_pydantic() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": "150",
        "completion_tokens": true,
        "total_tokens": 151,
        "cache_read_input_tokens": "40",
        "cache_creation_input_tokens": true,
        "reasoning_tokens": "12"
    }}))
    .unwrap()
    .unwrap();
    assert_eq!(usage.prompt_tokens, 150);
    assert_eq!(usage.completion_tokens, 1);
    let prompt = usage.prompt_tokens_details.expect("details");
    assert_eq!(prompt.cached_tokens, 40);
    assert_eq!(prompt.cache_creation_tokens, Some(1));
    let completion = usage.completion_tokens_details.expect("details");
    assert_eq!(completion.reasoning_tokens, Some(12));
}

#[test]
fn chat_usage_still_rejects_uncoercible_counts() {
    assert!(get_usage_object(&json!({"usage": {"prompt_tokens": "not-a-number"}})).is_err());
    assert!(get_usage_object(&json!({"usage": {"prompt_tokens": 12.5}})).is_err());
}
