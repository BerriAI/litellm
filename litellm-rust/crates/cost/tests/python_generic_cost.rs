use litellm_cost::generic_cost::{
    ResolvedTokenRates, billable_prompt_details, calculate_generic_cost_with_resolved_rates,
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
