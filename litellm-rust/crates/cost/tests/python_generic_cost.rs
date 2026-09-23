#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_generic_cost_per_token_overlapping_cached_and_image_tokens

use jiff::Timestamp;
use litellm_cost::generic_cost::{
    ResolvedTokenRates, billable_prompt_details, calculate_generic_cost_from_model_info,
    calculate_generic_cost_from_model_info_without_off_peak,
    calculate_generic_cost_with_resolved_rates,
};
use litellm_cost::generic_input::InputBaseRates;
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

fn rates() -> ResolvedTokenRates {
    ResolvedTokenRates {
        input: InputBaseRates {
            prompt: 2e-6,
            cache_read: 0.5e-6,
            cache_creation: 3e-6,
            cache_creation_above_1hr: 3e-6,
        },
        output: 7e-6,
        reasoning: None,
        multiplier: 1.0,
    }
}

#[rstest]
fn calculate_generic_cost_with_resolved_rates_bills_plain_tokens() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100
    }}))
    .unwrap()
    .unwrap();
    let (prompt, completion) =
        calculate_generic_cost_with_resolved_rates(&usage, &json!({}), rates(), None);
    assert!((prompt - 1000.0 * 2e-6).abs() < 1e-12);
    assert!((completion - 100.0 * 7e-6).abs() < 1e-12);
}

#[rstest]
fn calculate_generic_cost_from_model_info_parses_rates_with_surrounding_whitespace() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110
    }}))
    .unwrap()
    .unwrap();
    let model_info = json!({
        "input_cost_per_token": " 2e-6 ",
        "output_cost_per_token": " 4e-6 "
    });
    let actual = calculate_generic_cost_from_model_info_without_off_peak(
        &usage,
        &model_info,
        None,
        false,
        1.0,
    );
    assert!((actual.0 - 100.0 * 2e-6).abs() < 1e-12);
    assert!((actual.1 - 10.0 * 4e-6).abs() < 1e-12);
}

#[rstest]
fn billable_prompt_details_clamps_overlapping_cache_and_modalities() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "prompt_tokens_details": {
            "cached_tokens": 200,
            "cache_write_tokens": 100,
            "text_tokens": 1000,
            "audio_tokens": 100,
            "image_tokens": 50,
            "video_tokens": 50
        }
    }}))
    .unwrap()
    .unwrap();
    let details = billable_prompt_details(&usage);
    assert_eq!(details.text_tokens, 500);
    assert_eq!(details.audio_tokens, 100);
    assert_eq!(details.image_tokens, 50);
    assert_eq!(details.video_tokens, 50);
    let model_info = json!({
        "input_cost_per_audio_token": 4e-6,
        "input_cost_per_image_token": 5e-6,
        "input_cost_per_video_token": 6e-6
    });
    let (prompt, completion) =
        calculate_generic_cost_with_resolved_rates(&usage, &model_info, rates(), None);
    assert!((prompt - 0.00235).abs() < 1e-12);
    assert!((completion - 100.0 * 7e-6).abs() < 1e-12);
}

#[rstest]
fn billable_prompt_details_uses_uncached_remainder_when_text_is_missing() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 10,
        "total_tokens": 1010,
        "prompt_tokens_details": {"cached_tokens": 200, "audio_tokens": 100}
    }}))
    .unwrap()
    .unwrap();
    let details = billable_prompt_details(&usage);
    assert_eq!(details.text_tokens, 700);
    assert_eq!(details.audio_tokens, 100);
    assert_eq!(details.cache_hit_tokens, 200);
}

#[rstest]
fn calculate_generic_cost_with_resolved_rates_applies_multiplier_to_both_sides() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    }}))
    .unwrap()
    .unwrap();
    let base = calculate_generic_cost_with_resolved_rates(&usage, &json!({}), rates(), None);
    let uplifted = calculate_generic_cost_with_resolved_rates(
        &usage,
        &json!({}),
        ResolvedTokenRates {
            multiplier: 1.1,
            ..rates()
        },
        None,
    );
    assert!((uplifted.0 - base.0 * 1.1).abs() < 1e-12);
    assert!((uplifted.1 - base.1 * 1.1).abs() < 1e-12);
}

#[rstest]
fn calculate_generic_cost_from_model_info_without_off_peak_uses_threshold_cache_and_reasoning_rates()
 {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 128_001,
        "completion_tokens": 100,
        "total_tokens": 128_101,
        "prompt_tokens_details": {"cached_tokens": 20_000},
        "completion_tokens_details": {"reasoning_tokens": 40}
    }}))
    .unwrap()
    .unwrap();
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "output_cost_per_reasoning_token": 8e-6,
        "input_cost_per_token_above_128k_tokens": 5e-6,
        "output_cost_per_token_above_128k_tokens": 7e-6,
        "cache_read_input_token_cost_above_128k_tokens": 1e-6
    });
    let actual = calculate_generic_cost_from_model_info_without_off_peak(
        &usage,
        &model_info,
        None,
        false,
        1.0,
    );
    let expected_prompt = 108_001.0 * 5e-6 + 20_000.0 * 1e-6;
    let expected_output = 60.0 * 7e-6 + 40.0 * 8e-6;
    assert!((actual.0 - expected_prompt).abs() < 1e-12);
    assert!((actual.1 - expected_output).abs() < 1e-12);
}

#[rstest]
fn calculate_generic_cost_from_model_info_without_off_peak_uses_selected_tier_for_reasoning() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 32_001,
        "completion_tokens": 100,
        "total_tokens": 32_101,
        "completion_tokens_details": {"reasoning_tokens": 40}
    }}))
    .unwrap()
    .unwrap();
    let model_info = json!({
        "output_cost_per_token": 4e-6,
        "tiered_pricing": [
            {"range": [0, 32_000], "input_cost_per_token": 2e-6, "output_cost_per_token": 3e-6},
            {"range": [32_000, 128_000], "input_cost_per_token": 5e-6, "output_cost_per_reasoning_token": 9e-6}
        ]
    });
    let actual = calculate_generic_cost_from_model_info_without_off_peak(
        &usage,
        &model_info,
        None,
        false,
        1.0,
    );
    assert!((actual.0 - 32_001.0 * 5e-6).abs() < 1e-12);
    assert!((actual.1 - (60.0 * 4e-6 + 40.0 * 9e-6)).abs() < 1e-12);
}

#[rstest]
fn calculate_generic_cost_from_model_info_uses_off_peak_reasoning_and_cache_rates() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 100,
        "total_tokens": 200,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 20}
    }}))
    .unwrap()
    .unwrap();
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "output_cost_per_reasoning_token": 8e-6,
        "off_peak_pricing": {
            "hours_utc": "16:30-00:30",
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "output_cost_per_reasoning_token": 3e-6
        }
    });
    let at: Timestamp = "2026-01-01T18:00Z".parse().unwrap();
    let actual = calculate_generic_cost_from_model_info(&usage, &model_info, None, false, 1.0, at);
    let expected_prompt = 60.0 * 1e-6 + 40.0 * 1e-6;
    let expected_output = 80.0 * 2e-6 + 20.0 * 3e-6;
    assert!((actual.0 - expected_prompt).abs() < 1e-12);
    assert!((actual.1 - expected_output).abs() < 1e-12);
}
