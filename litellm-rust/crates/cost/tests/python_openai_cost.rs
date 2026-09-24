#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/test_video_generation.py::TestVideoGeneration
// mirrors: unit/llms/openai/transcriptions/test_transcription_duration_hidden.py::TestCostCalculatorReadsDurationFromHiddenParams
use litellm_cost::error::CostError;

use litellm_cost::openai_cost::{
    cost_router, video_generation_cost, video_output_cost_per_second,
    video_resolution_to_cost_field_suffix,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case("transcription", "cost_per_second")]
#[case("atranscription", "cost_per_second")]
#[case("completion", "cost_per_token")]
#[case("video_generation", "cost_per_token")]
fn openai_cost_router_selects_transcription_duration_path(
    #[case] call_type: &str,
    #[case] expected: &str,
) {
    assert_eq!(cost_router(call_type), expected);
}

#[rstest]
#[case(" 1080P ", Some("1080p"))]
#[case("HD/720p", Some("hd720p"))]
#[case(" ", None)]
#[case("////////////////", None)]
#[case("1234567890123456789012345", None)]
fn video_resolution_suffix_normalizes_and_rejects_unusable_keys(
    #[case] resolution: &str,
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        video_resolution_to_cost_field_suffix(resolution).as_deref(),
        expected
    );
}

#[rstest]
#[case(json!({"output_cost_per_second_1080p": 0.08, "output_cost_per_second": 0.05}), Some("1080p"), Some(0.08))]
#[case(json!({"output_cost_per_second_1080p": 0.0, "output_cost_per_second": 0.05}), Some("1080p"), Some(0.0))]
#[case(json!({"output_cost_per_second": 0.05}), Some("1080p"), Some(0.05))]
#[case(json!({"output_cost_per_second_1080p": "0.08", "output_cost_per_second": 0.05}), Some("1080p"), Some(0.08))]
#[case(json!({}), Some("1080p"), None)]
fn video_output_rate_prefers_resolution_and_preserves_zero(
    #[case] model_info: Value,
    #[case] resolution: Option<&str>,
    #[case] expected: Option<f64>,
) {
    assert_eq!(
        video_output_cost_per_second(&model_info, resolution),
        expected
    );
}

#[rstest]
#[case(json!({"output_cost_per_video_per_second": 0.2, "output_cost_per_second_1080p": 0.08, "output_cost_per_second": 0.05}), Some("1080p"), 2.0)]
#[case(json!({"output_cost_per_second_1080p": 0.08, "output_cost_per_second": 0.05}), Some("1080p"), 0.8)]
#[case(json!({"output_cost_per_second": 0.05}), Some("1080p"), 0.5)]
#[case(json!({}), Some("1080p"), 0.0)]
fn video_generation_cost_matches_python_rate_precedence(
    #[case] model_info: Value,
    #[case] resolution: Option<&str>,
    #[case] expected: f64,
) {
    assert!(
        (video_generation_cost(&model_info, 10.0, resolution).unwrap() - expected).abs() < 1e-12
    );
}

#[rstest]
fn video_generation_cost_rejects_invalid_duration() {
    assert_eq!(
        video_generation_cost(&json!({"output_cost_per_second": 0.05}), -1.0, None),
        Err(CostError::InvalidQuantity)
    );
}

#[rstest]
#[case::padded_string(json!({"output_cost_per_second": " 0.05 "}), Some(0.05))]
#[case::boolean(json!({"output_cost_per_second": true}), Some(1.0))]
#[case::resolution_tier_first(json!({"output_cost_per_second": 0.05, "output_cost_per_second_720p": "0.1"}), Some(0.1))]
fn video_output_cost_per_second_converts_rates_like_python_float(
    #[case] model_info: serde_json::Value,
    #[case] expected: Option<f64>,
) {
    assert_eq!(
        video_output_cost_per_second(&model_info, Some("720p")),
        expected
    );
}
