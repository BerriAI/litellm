#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_usage_object_transformation.py

use litellm_cost::error::CostError;
use litellm_cost::usage_dispatch::{chat_usage, get_usage_object};
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

#[rstest]
fn chat_usage_coerces_numeric_strings_and_bools_like_pydantic() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": "150",
        "completion_tokens": true,
        "total_tokens": 151,
        "cache_read_input_tokens": "40",
        "cache_creation_input_tokens": true
    }}))
    .unwrap()
    .unwrap();
    assert_eq!(usage.prompt_tokens, 150);
    assert_eq!(usage.completion_tokens, 1);
    let prompt = usage.prompt_tokens_details.expect("details");
    assert_eq!(
        prompt.cached_tokens, 0,
        "string cache_read_input_tokens fails Python's isinstance(int) guard and is not folded"
    );
    assert_eq!(prompt.cache_creation_tokens, Some(1));
    assert_eq!(
        usage
            .extra
            .get("_cache_creation_input_tokens")
            .and_then(Value::as_u64),
        Some(1)
    );
    assert!(!usage.extra.contains_key("_cache_read_input_tokens"));
}

#[rstest]
fn chat_usage_folds_deepseek_prompt_cache_hit_tokens() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "prompt_cache_hit_tokens": 40
    }}))
    .unwrap()
    .unwrap();
    let prompt = usage.prompt_tokens_details.expect("details");
    assert_eq!(prompt.cached_tokens, 40);
    assert_eq!(
        usage
            .extra
            .get("_cache_read_input_tokens")
            .and_then(Value::as_u64),
        Some(40)
    );
}

#[rstest]
fn chat_usage_reasoning_string_parses_where_python_raises() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "reasoning_tokens": "12"
    }}))
    .unwrap()
    .unwrap();
    let completion = usage.completion_tokens_details.expect("details");
    assert_eq!(
        completion.reasoning_tokens,
        Some(12),
        "divergence: Python's Usage.__init__ raises TypeError on a string reasoning_tokens here"
    );
}

#[rstest]
fn chat_usage_still_rejects_uncoercible_counts() {
    assert!(get_usage_object(&json!({"usage": {"prompt_tokens": "not-a-number"}})).is_err());
    assert!(get_usage_object(&json!({"usage": {"prompt_tokens": 12.5}})).is_err());
}

#[rstest]
#[case::top_level_zero_overrides_details(json!({"cache_creation_input_tokens": 0, "prompt_tokens_details": {"cache_write_tokens": 7}}), Some(0))]
#[case::top_level_count_overrides_details(json!({"cache_creation_input_tokens": 3, "prompt_tokens_details": {"cache_write_tokens": 7}}), Some(3))]
#[case::details_write_zero_beats_creation(json!({"prompt_tokens_details": {"cache_write_tokens": 0, "cache_creation_tokens": 5}}), Some(0))]
#[case::details_creation_alias(json!({"prompt_tokens_details": {"cache_creation_tokens": 5}}), Some(5))]
#[case::float_top_level_is_not_an_int(json!({"cache_creation_input_tokens": 3.0, "prompt_tokens_details": {"cache_write_tokens": 7}}), Some(7))]
fn chat_usage_takes_cache_writes_in_python_usage_init_order(
    #[case] fields: Value,
    #[case] expected: Option<u64>,
) {
    let raw = json!({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})
        .as_object()
        .unwrap()
        .clone()
        .into_iter()
        .chain(fields.as_object().unwrap().clone())
        .collect::<serde_json::Map<_, _>>();
    let usage = chat_usage(&Value::Object(raw)).unwrap();
    let details = usage.prompt_tokens_details.unwrap();
    assert_eq!(details.cache_write_tokens, expected);
    assert_eq!(details.cache_creation_tokens, expected);
}

#[rstest]
#[case::integral_float(json!(5.0), Ok(5))]
#[case::numeric_string(json!(" 5 "), Ok(5))]
#[case::boolean(json!(true), Ok(1))]
#[case::fractional_float(json!(5.5), Err(CostError::InvalidUsage))]
#[case::negative(json!(-1), Err(CostError::InvalidUsage))]
fn chat_usage_reads_token_counts_as_pydantic_ints(
    #[case] prompt_tokens: Value,
    #[case] expected: Result<u64, CostError>,
) {
    let usage = chat_usage(&json!({"prompt_tokens": prompt_tokens, "completion_tokens": 1}));
    assert_eq!(usage.map(|usage| usage.prompt_tokens), expected);
}
