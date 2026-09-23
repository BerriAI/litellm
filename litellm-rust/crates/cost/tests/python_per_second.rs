use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
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
        catalog
            .cost_per_token(ModelCostRequest {
                model: "model",
                provider: Some("provider"),
                region: None,
                usage: &usage,
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at,
                response_time_ms: Some(2000.0),
            })
            .unwrap(),
        (1.0, 2.0)
    );
}
