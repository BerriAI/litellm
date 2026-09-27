#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/test_cost_calculator.py::test_cost_per_token_per_second_pricing

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::per_second::get_replicate_completion_pricing;
use litellm_cost::per_second::{
    bills_wall_clock_seconds, has_token_or_tiered_pricing, per_second_pricing_cost,
};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case(json!({"input_cost_per_second": 0.5, "output_cost_per_second": 1.0}), Some((1.0, 2.0)))]
#[case(json!({"input_cost_per_second": 0.5}), Some((1.0, 0.0)))]
#[case(json!({"output_cost_per_second": 1.0}), Some((0.0, 2.0)))]
#[case(json!({"input_cost_per_second": 0.5, "input_cost_per_token": 1e-6}), None)]
#[case(json!({"input_cost_per_second": 0.5, "tiered_pricing": []}), None)]
#[case(json!({"input_cost_per_second": 0.5, "mode": "image_generation"}), None)]
#[case(json!({"input_cost_per_token": 1e-6}), None)]
fn per_second_pricing_cost_only_prices_eligible_models(
    #[case] model_info: serde_json::Value,
    #[case] expected: Option<(f64, f64)>,
) {
    assert_eq!(per_second_pricing_cost(&model_info, Some(2000.0)), expected);
}

#[rstest]
fn per_second_pricing_cost_uses_zero_for_missing_response_time() {
    assert_eq!(
        per_second_pricing_cost(&json!({"input_cost_per_second": 0.5}), None),
        Some((0.0, 0.0))
    );
    assert!(!has_token_or_tiered_pricing(
        &json!({"input_cost_per_token": 0.0})
    ));
    assert!(bills_wall_clock_seconds(&json!({"mode": "responses"})));
}

#[rstest]
fn model_info_catalog_routes_wall_clock_pricing_before_tokens() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "provider/model".to_owned(),
        json!({"mode": "responses", "input_cost_per_second": 0.5, "output_cost_per_second": 1.0}),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    }}))
    .unwrap()
    .unwrap();
    let at: Timestamp = "2026-01-01T18:00Z".parse().unwrap();
    assert_eq!(
        litellm_cost::cost_calculator::cost_per_token(
            &catalog,
            ModelCostRequest {
                model: "model",
                provider: Some("provider"),
                region: None,
                usage: &usage,
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at,
                response_time_ms: Some(2000.0),
            }
        )
        .unwrap(),
        (1.0, 2.0)
    );
}

#[rstest]
#[case("perplexity")]
#[case("xai")]
fn wall_clock_pricing_precedes_provider_reported_cost(#[case] provider: &str) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        format!("{provider}/model"),
        json!({"mode": "responses", "input_cost_per_second": 0.5, "output_cost_per_second": 1.0}),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cost": 0.7
    }}))
    .unwrap()
    .unwrap();
    assert_eq!(
        litellm_cost::cost_calculator::cost_per_token(
            &catalog,
            ModelCostRequest {
                model: "model",
                provider: Some(provider),
                region: None,
                usage: &usage,
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at: "2026-01-01T18:00Z".parse().unwrap(),
                response_time_ms: Some(2000.0),
            }
        )
        .unwrap(),
        (1.0, 2.0)
    );
}
#[rstest]
#[case(2500.0, None, None, 100.0, 0.05)]
#[case(0.0, Some(100.0), Some(102.0), 200.0, 0.00004)]
#[case(0.0, None, None, 200.0, 0.0)]
fn replicate_completion_pricing_matches_python_duration_units(
    #[case] total_time_ms: f64,
    #[case] created_seconds: Option<f64>,
    #[case] ended_seconds: Option<f64>,
    #[case] now_seconds: f64,
    #[case] expected: f64,
) {
    let actual = get_replicate_completion_pricing(
        total_time_ms,
        created_seconds,
        ended_seconds,
        now_seconds,
        0.02,
    );
    assert!((actual - expected).abs() < 1e-12);
}
