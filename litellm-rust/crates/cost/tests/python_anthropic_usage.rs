#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/llms/anthropic/chat/test_anthropic_chat_transformation.py::test_calculate_usage_aggregates_cache_creation_split_across_iterations
use litellm_cost::error::CostError;

use litellm_cost::anthropic_usage::{
    is_anthropic_usage_object, transform_anthropic_usage_to_chat_usage,
};
use litellm_cost::generic_cost::calculate_generic_cost_from_model_info_with_region;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case(json!({"input_tokens": 3, "output_tokens": 5, "cache_read_input_tokens": 4014}), true)]
#[case(json!({"input_tokens": 3, "output_tokens": 5, "cache_creation_input_tokens": 10}), true)]
#[case(json!({"prompt_tokens": 4017, "input_tokens": 3, "cache_read_input_tokens": 4014}), false)]
#[case(json!({"input_tokens": 3, "output_tokens": 5}), false)]
#[case(json!({"input_tokens": 3, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 2}}), false)]
fn is_anthropic_usage_object_distinguishes_cache_convention(
    #[case] usage: Value,
    #[case] expected: bool,
) {
    assert_eq!(is_anthropic_usage_object(&usage), expected);
}

#[rstest]
#[case(json!({"input_tokens": null, "output_tokens": 43, "cache_read_input_tokens": null, "cache_creation_input_tokens": null}), 0, 43, 0, 0)]
#[case(json!({"input_tokens": 1, "output_tokens": null, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 100}), 301, 0, 200, 100)]
fn calculate_usage_adds_anthropic_cache_counts_once(
    #[case] raw: Value,
    #[case] expected_prompt: u64,
    #[case] expected_output: u64,
    #[case] expected_read: u64,
    #[case] expected_creation: u64,
) {
    let usage = transform_anthropic_usage_to_chat_usage(&raw, None, false).unwrap();
    let prompt = usage.prompt_tokens_details.unwrap();
    assert_eq!(usage.prompt_tokens, expected_prompt);
    assert_eq!(usage.completion_tokens, expected_output);
    assert_eq!(usage.total_tokens, expected_prompt + expected_output);
    assert_eq!(prompt.cached_tokens, expected_read);
    assert_eq!(prompt.cache_creation_tokens, Some(expected_creation));
    assert_eq!(
        prompt.text_tokens,
        Some(expected_prompt - expected_read - expected_creation)
    );
}

#[rstest]
#[case(10_000, 10_000, true, 0, 20_000)]
#[case(10_000, 7_000, false, 7_000, 10_000)]
fn calculate_usage_aggregates_iteration_cache_write_details(
    #[case] first_write: u64,
    #[case] second_write: u64,
    #[case] second_has_breakdown: bool,
    #[case] expected_5m: u64,
    #[case] expected_1h: u64,
) {
    let second_details = second_has_breakdown.then(|| {
        json!({
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": second_write
        })
    });
    let raw = json!({
        "input_tokens": 0,
        "output_tokens": 5,
        "cache_creation_input_tokens": 0,
        "iterations": [
            {"input_tokens": 0, "output_tokens": 3, "cache_creation_input_tokens": first_write, "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": first_write}},
            {"input_tokens": 0, "output_tokens": 2, "cache_creation_input_tokens": second_write, "cache_creation": second_details}
        ]
    });
    let usage = transform_anthropic_usage_to_chat_usage(&raw, None, false).unwrap();
    let details = usage
        .prompt_tokens_details
        .as_ref()
        .unwrap()
        .cache_creation_token_details
        .as_ref()
        .unwrap();
    assert_eq!(usage.prompt_tokens, first_write + second_write);
    assert_eq!(details.ephemeral_5m_input_tokens, Some(expected_5m));
    assert_eq!(details.ephemeral_1h_input_tokens, Some(expected_1h));
    let model_info = json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "cache_creation_input_token_cost": 1.25e-6,
        "cache_creation_input_token_cost_above_1hr": 2e-6
    });
    let (input, _) = calculate_generic_cost_from_model_info_with_region(
        &usage,
        &model_info,
        None,
        false,
        None,
        None,
        "2026-01-01T12:00Z".parse().unwrap(),
    );
    assert!((input - (expected_5m as f64 * 1.25e-6 + expected_1h as f64 * 2e-6)).abs() < 1e-12);
}

#[rstest]
#[case(json!({"input_tokens": 32, "output_tokens": 421, "output_tokens_details": {"thinking_tokens": 372}}), None, false, Some(372), Some(49))]
#[case(json!({"input_tokens": 32, "output_tokens": 10}), Some(15), false, Some(10), Some(0))]
#[case(json!({"input_tokens": 32, "output_tokens": 10}), None, true, None, None)]
#[case(json!({"input_tokens": 32, "output_tokens": 10}), None, false, Some(0), Some(10))]
fn calculate_usage_partitions_reported_or_estimated_reasoning(
    #[case] raw: Value,
    #[case] estimated_reasoning: Option<u64>,
    #[case] thinking_block: bool,
    #[case] expected_reasoning: Option<u64>,
    #[case] expected_text: Option<u64>,
) {
    let usage =
        transform_anthropic_usage_to_chat_usage(&raw, estimated_reasoning, thinking_block).unwrap();
    let details = usage.completion_tokens_details.unwrap();
    assert_eq!(details.reasoning_tokens, expected_reasoning);
    assert_eq!(details.text_tokens, expected_text);
}

#[rstest]
fn calculate_usage_sums_reported_reasoning_across_iterations() {
    let raw = json!({
        "input_tokens": 10,
        "output_tokens": 300,
        "iterations": [
            {"input_tokens": 5, "output_tokens": 100, "output_tokens_details": {"thinking_tokens": 60}},
            {"input_tokens": 5, "output_tokens": 200, "output_tokens_details": {"thinking_tokens": 90}}
        ]
    });
    let usage = transform_anthropic_usage_to_chat_usage(&raw, None, false).unwrap();
    let details = usage.completion_tokens_details.unwrap();
    assert_eq!(usage.completion_tokens, 300);
    assert_eq!(details.reasoning_tokens, Some(150));
    assert_eq!(details.text_tokens, Some(150));
}

#[rstest]
fn iteration_sums_overflow_errors_where_python_yields_a_big_int() {
    let usage = transform_anthropic_usage_to_chat_usage(
        &json!({
            "input_tokens": 0,
            "output_tokens": 0,
            "iterations": [
                {"input_tokens": u64::MAX, "output_tokens": 1},
                {"input_tokens": 1, "output_tokens": 1}
            ]
        }),
        None,
        false,
    );
    assert_eq!(
        usage.unwrap_err(),
        CostError::TokenCountOverflow,
        "divergence: Python sums arbitrary-precision ints and returns a huge cost"
    );
}

#[rstest]
#[case::integer(json!(5), Some(5), Some(5))]
#[case::numeric_string(json!("5"), Some(5), Some(5))]
#[case::integral_float(json!(5.0), Some(5), Some(5))]
fn calculate_usage_reads_detail_counts_with_pydantic_lax_ints(
    #[case] count: Value,
    #[case] expected_thinking: Option<u64>,
    #[case] expected_five_minute: Option<u64>,
) {
    let usage = transform_anthropic_usage_to_chat_usage(
        &json!({
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 5,
            "cache_creation": {"ephemeral_5m_input_tokens": count},
            "output_tokens_details": {"thinking_tokens": count}
        }),
        None,
        true,
    )
    .unwrap();
    assert_eq!(
        usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        expected_thinking
    );
    assert_eq!(
        usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cache_creation_token_details.as_ref())
            .and_then(|details| details.ephemeral_5m_input_tokens),
        expected_five_minute
    );
}
