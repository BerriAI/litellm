#![allow(clippy::disallowed_types)]

// mirrors: local_testing/test_completion_cost.py::test_gemini_completion_cost

use std::collections::HashMap;

use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case("openai")]
#[case("bedrock")]
#[case("gemini")]
#[case("deepseek")]
#[case("tencent")]
#[case("amazon_nova")]
fn generic_provider_wrappers_price_cached_and_output_tokens(#[case] provider: &str) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        format!("{provider}/model"),
        json!({
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 3e-6,
            "cache_read_input_token_cost": 0.5e-6
        }),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_tokens_details": {"cached_tokens": 20}
    }}))
    .unwrap()
    .unwrap();
    let (prompt, completion) = litellm_cost::cost_calculator::cost_per_token(
        &catalog,
        ModelCostRequest {
            model: "model",
            provider: Some(provider),
            region: None,
            usage: &usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            at: "2026-01-01T12:00Z".parse().unwrap(),
            response_time_ms: None,
        },
    )
    .unwrap();
    assert!((prompt - (80.0 * 2e-6 + 20.0 * 0.5e-6)).abs() < 1e-12);
    assert!((completion - 10.0 * 3e-6).abs() < 1e-12);
}
