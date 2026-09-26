#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_data_residency_applies_uplift

use jiff::Timestamp;
use litellm_cost::generic_cost::calculate_generic_cost_from_model_info_with_region;
use litellm_cost::regional_uplift::{
    apply_regional_totals_uplift, get_provider_specific_geo_multiplier,
    get_regional_uplift_multiplier, get_vertex_regional_endpoint_uplift,
};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case(Some("eu"), 1.2)]
#[case(Some("EU"), 1.2)]
#[case(Some("us"), 1.1)]
#[case(Some("apac"), 1.0)]
#[case(None, 1.0)]
fn get_regional_uplift_multiplier_selects_supported_residencies(
    #[case] residency: Option<&str>,
    #[case] expected: f64,
) {
    let model_info = json!({
        "regional_processing_uplift_multiplier_eu": 1.2,
        "regional_processing_uplift_multiplier_us": 1.1
    });
    assert_eq!(
        get_regional_uplift_multiplier(&model_info, residency),
        expected
    );
}

#[rstest]
#[case(None, 1.0)]
#[case(Some("global"), 1.0)]
#[case(Some("GLOBAL"), 1.0)]
#[case(Some("us-east5"), 1.3)]
fn get_vertex_regional_endpoint_uplift_ignores_global(
    #[case] location: Option<&str>,
    #[case] expected: f64,
) {
    assert_eq!(
        get_vertex_regional_endpoint_uplift(
            &json!({"regional_endpoint_uplift_multiplier": 1.3}),
            location
        ),
        expected
    );
    assert_eq!(
        get_vertex_regional_endpoint_uplift(
            &json!({"regional_endpoint_uplift_multiplier": "invalid"}),
            location
        ),
        1.0
    );
}

#[rstest]
#[case(None, 1.0)]
#[case(Some("global"), 1.0)]
#[case(Some("not_available"), 1.0)]
#[case(Some("US"), 1.4)]
fn get_provider_specific_geo_multiplier_uses_reported_geo(
    #[case] geo: Option<&str>,
    #[case] expected: f64,
) {
    assert_eq!(
        get_provider_specific_geo_multiplier(&json!({"provider_specific_entry": {"us": 1.4}}), geo),
        expected
    );
}

#[rstest]
fn calculate_generic_cost_from_model_info_with_region_scales_both_sides() {
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "inference_geo": "us"
    }}))
    .unwrap()
    .unwrap();
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "regional_processing_uplift_multiplier_eu": 1.2,
        "regional_endpoint_uplift_multiplier": 1.3,
        "provider_specific_entry": {"us": 1.4}
    });
    let at: Timestamp = "2026-01-01T18:00Z".parse().unwrap();
    let actual = calculate_generic_cost_from_model_info_with_region(
        &usage,
        &model_info,
        None,
        false,
        Some("eu"),
        Some("us-east5"),
        at,
    );
    assert_eq!(
        actual,
        (100.0 * 2e-6 * 1.2 * 1.3, 50.0 * 4e-6 * 1.2 * 1.3),
        "generic totals exclude the provider-specific geo multiplier; only the anthropic wrapper applies it"
    );
}

#[rstest]
#[case::padded_string(json!(" 1.5 "), 1.5)]
#[case::boolean_false(json!(false), 0.0)]
#[case::boolean_true(json!(true), 1.0)]
#[case::null(json!(null), 1.0)]
#[case::unparseable(json!("high"), 1.0)]
#[case::object(json!({"x": 2}), 1.0)]
fn uplift_multipliers_are_coerced_like_python_float(
    #[case] raw: serde_json::Value,
    #[case] expected: f64,
) {
    let model_info = json!({
        "regional_processing_uplift_multiplier_eu": raw,
        "regional_endpoint_uplift_multiplier": raw
    });
    assert_eq!(
        get_regional_uplift_multiplier(&model_info, Some("eu")),
        expected
    );
    assert_eq!(
        get_vertex_regional_endpoint_uplift(&model_info, Some("us-east5")),
        expected
    );
}

#[rstest]
fn apply_regional_totals_uplift_multiplies_in_python_order() {
    let model_info = json!({
        "regional_processing_uplift_multiplier_eu": 1.1,
        "regional_endpoint_uplift_multiplier": 1.1
    });
    let prompt = 7.0 * 3e-7;
    let (actual, _) =
        apply_regional_totals_uplift((prompt, 0.0), &model_info, Some("eu"), Some("us-east5"));
    assert_eq!(actual, prompt * 1.1 * 1.1);
    assert_ne!(actual, prompt * (1.1 * 1.1));
}
