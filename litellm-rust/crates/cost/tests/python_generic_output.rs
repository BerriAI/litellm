#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_generic_cost_per_token_bills_nested_reasoning_once_beside_audio_output

use litellm_cost::generic_output::{calculate_output_cost, resolve_reasoning_token_cost};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn usage(completion_tokens: u64, details: Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": {
        "prompt_tokens": 10,
        "completion_tokens": completion_tokens,
        "total_tokens": completion_tokens + 10,
        "completion_tokens_details": details
    }}))
    .unwrap()
    .unwrap()
}

#[rstest]
#[case(json!({"output_cost_per_reasoning_token": 4e-6}), None, 4e-6)]
#[case(json!({"output_cost_per_reasoning_token": 4e-6, "output_cost_per_token_priority": 6e-6}), Some("priority"), 6e-6)]
#[case(json!({"output_cost_per_reasoning_token": 4e-6, "output_cost_per_token_priority": 6e-6, "output_cost_per_reasoning_token_priority": 8e-6}), Some("priority"), 8e-6)]
#[case(json!({"output_cost_per_reasoning_token": 4e-6, "output_cost_per_token_priority": 6e-6, "output_cost_per_reasoning_token_priority": null}), Some("priority"), 6e-6)]
#[case(json!({"output_cost_per_reasoning_token": 4e-6}), Some("priority"), 4e-6)]
fn resolve_reasoning_token_cost_follows_service_tier_precedence(
    #[case] model_info: Value,
    #[case] tier: Option<&str>,
    #[case] expected: f64,
) {
    assert_eq!(
        resolve_reasoning_token_cost(&model_info, tier, 6e-6),
        expected
    );
}

#[rstest]
fn calculate_output_cost_bills_nested_reasoning_once_beside_audio() {
    let usage = usage(
        100,
        json!({"text_tokens": 90, "reasoning_tokens": 20, "audio_tokens": 10}),
    );
    let model_info = json!({
        "output_cost_per_reasoning_token": 4e-6,
        "output_cost_per_audio_token": "8e-6"
    });
    let expected = 70.0 * 2e-6 + 20.0 * 4e-6 + 10.0 * 8e-6;
    assert!(
        (calculate_output_cost(&usage, &model_info, 2e-6, None, None) - expected).abs() < 1e-12
    );
}

#[rstest]
fn calculate_output_cost_bills_image_video_and_reasoning_separately() {
    let usage = usage(
        100,
        json!({
            "reasoning_tokens": 20,
            "image_tokens": 30,
            "video_tokens": 10,
            "audio_tokens": 5
        }),
    );
    let model_info = json!({
        "output_cost_per_image_token": 5e-6,
        "output_cost_per_video_token": 7e-6
    });
    let expected = 35.0 * 2e-6 + 20.0 * 3e-6 + 30.0 * 5e-6 + 10.0 * 7e-6 + 5.0 * 2e-6;
    assert!(
        (calculate_output_cost(&usage, &model_info, 2e-6, None, Some(3e-6)) - expected).abs()
            < 1e-12
    );
}

#[rstest]
fn calculate_output_cost_uses_base_rate_when_details_are_absent() {
    let usage = usage(100, json!({}));
    assert_eq!(
        calculate_output_cost(&usage, &json!({}), 2e-6, None, None),
        100.0 * 2e-6
    );
}

#[rstest]
fn calculate_output_cost_falls_back_to_base_rate_for_unpriced_modalities() {
    let usage = usage(100, json!({"text_tokens": 30, "audio_tokens": 70}));
    assert_eq!(
        calculate_output_cost(
            &usage,
            &json!({"output_cost_per_audio_token": null}),
            2e-6,
            None,
            None
        ),
        100.0 * 2e-6
    );
}
