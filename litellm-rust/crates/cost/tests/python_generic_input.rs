#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_calculate_cache_writing_cost

use litellm_cost::generic_input::{
    InputBaseRates, calculate_cache_writing_cost, calculate_cost_component, calculate_input_cost,
    get_cost_per_unit,
};
use litellm_cost::generic_usage::ParsedPromptDetails;
use litellm_cost::responses_usage::CacheCreationTokenDetails;
use rstest::rstest;
use serde_json::json;

fn rates() -> InputBaseRates {
    InputBaseRates {
        prompt: 2e-6,
        cache_read: 0.5e-6,
        cache_creation: 2.5e-6,
        cache_creation_above_1hr: 4e-6,
    }
}

#[rstest]
#[case(json!({"input_cost_per_audio_token": "3e-6"}), "input_cost_per_audio_token", None, Some(3e-6))]
#[case(json!({"input_cost_per_audio_token": " 3e-6 "}), "input_cost_per_audio_token", None, Some(3e-6))]
#[case(json!({"input_cost_per_audio_token": 0.0}), "input_cost_per_audio_token", Some(1e-6), Some(0.0))]
#[case(json!({"input_cost_per_audio_token": "3e-6"}), "input_cost_per_audio_token_priority", None, Some(3e-6))]
#[case(json!({"input_cost_per_audio_token": "bad"}), "input_cost_per_audio_token", Some(1e-6), Some(1e-6))]
#[case(json!({"input_cost_per_audio_token": 3e-6}), "input_cost_per_audio_token_auto", None, Some(3e-6))]
#[case(json!({"input_cost_per_audio_token": 3e-6}), "input_cost_per_audio_token_standard", None, None)]
fn get_cost_per_unit_parses_string_rates_and_falls_back_from_tier(
    #[case] model_info: serde_json::Value,
    #[case] key: &str,
    #[case] default: Option<f64>,
    #[case] expected: Option<f64>,
) {
    assert_eq!(get_cost_per_unit(&model_info, key, default), expected);
}

#[rstest]
fn calculate_cost_component_charges_only_positive_usage() {
    let model_info = json!({"input_cost_per_query": "0.01"});
    assert_eq!(
        calculate_cost_component(&model_info, "input_cost_per_query", 4.0),
        0.04
    );
    assert_eq!(
        calculate_cost_component(&model_info, "input_cost_per_query", 0.0),
        0.0
    );
    assert_eq!(calculate_cost_component(&model_info, "missing", 4.0), 0.0);
}

#[rstest]
fn calculate_cache_writing_cost_uses_ephemeral_details_even_when_total_is_zero() {
    let details = CacheCreationTokenDetails {
        ephemeral_5m_input_tokens: Some(100),
        ephemeral_1h_input_tokens: Some(200),
    };
    assert!(
        (calculate_cache_writing_cost(0, Some(&details), 6e-6, 3.75e-6) - 0.001575).abs() < 1e-12
    );
    assert!((calculate_cache_writing_cost(300, None, 6e-6, 3.75e-6) - 0.001125).abs() < 1e-12);
}

#[rstest]
fn calculate_input_cost_prices_cached_audio_and_every_reported_unit() {
    let details = ParsedPromptDetails {
        cache_hit_tokens: 100,
        cache_hit_audio_tokens: 40,
        cache_creation_tokens: 50,
        text_tokens: 500,
        audio_tokens: 30,
        image_tokens: 20,
        video_tokens: 10,
        character_count: 200,
        image_count: 0,
        video_length_seconds: 0.0,
        audio_length_seconds: 0.0,
        query_count: 2,
        ..ParsedPromptDetails::default()
    };
    let model_info = json!({
        "cache_read_input_audio_token_cost_priority": 0.25e-6,
        "input_cost_per_audio_token_priority": 3e-6,
        "input_cost_per_image_token": 4e-6,
        "input_cost_per_video_token": 5e-6,
        "input_cost_per_character": 1e-7,
        "input_cost_per_query": 0.01
    });
    let expected = 500.0 * 2e-6
        + 60.0 * 0.5e-6
        + 40.0 * 0.25e-6
        + 30.0 * 3e-6
        + 20.0 * 4e-6
        + 10.0 * 5e-6
        + 50.0 * 2.5e-6
        + 200.0 * 1e-7
        + 2.0 * 0.01;
    let actual = calculate_input_cost(&details, &model_info, rates(), Some("priority"));
    assert!((actual - expected).abs() < 1e-12);
}

#[rstest]
fn calculate_input_cost_uses_count_and_duration_instead_of_duplicate_token_charges() {
    let details = ParsedPromptDetails {
        audio_tokens: 300,
        image_tokens: 200,
        video_tokens: 100,
        image_count: 2,
        audio_length_seconds: 3.0,
        video_length_seconds: 4.0,
        ..ParsedPromptDetails::default()
    };
    let model_info = json!({
        "input_cost_per_audio_token": 1e-6,
        "input_cost_per_image_token": 2e-6,
        "input_cost_per_video_token": 3e-6,
        "input_cost_per_image": 0.02,
        "input_cost_per_audio_per_second": 0.01,
        "input_cost_per_video_per_second": 0.03
    });
    let expected = 2.0 * 0.02 + 3.0 * 0.01 + 4.0 * 0.03;
    let actual = calculate_input_cost(&details, &model_info, rates(), None);
    assert!((actual - expected).abs() < 1e-12);
}

#[rstest]
#[case::positive_seconds(3.0, 3.0 * 0.01)]
#[case::negative_seconds_still_replace_tokens(-1.0, 0.0)]
#[case::zero_seconds_bill_tokens(0.0, 300.0 * 1e-6)]
fn calculate_input_cost_treats_any_nonzero_audio_duration_as_reported(
    #[case] seconds: f64,
    #[case] expected: f64,
) {
    let details = ParsedPromptDetails {
        audio_tokens: 300,
        audio_length_seconds: seconds,
        ..ParsedPromptDetails::default()
    };
    let model_info = json!({
        "input_cost_per_audio_token": 1e-6,
        "input_cost_per_audio_per_second": 0.01
    });
    let actual = calculate_input_cost(&details, &model_info, rates(), None);
    assert!((actual - expected).abs() < 1e-12);
}
