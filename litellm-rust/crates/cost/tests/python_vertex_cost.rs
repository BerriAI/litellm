#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/test_cost_calculator.py::test_vertex_uplift_composes_with_above_128k_pricing
use litellm_cost::vertex_cost::vertex_cost;

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::responses_usage::ChatUsage;
use litellm_cost::vertex_cost::{
    CostRoute, cost_per_character, cost_per_token, cost_router, handle_128k_pricing,
};
use rstest::rstest;
use serde_json::json;

fn at() -> Timestamp {
    "2026-09-22T12:00:00Z".parse().unwrap()
}

#[rstest]
#[case("vertex_ai", "claude-model", "completion", CostRoute::PerToken)]
#[case("vertex_ai", "gemini-2-flash", "completion", CostRoute::PerToken)]
#[case("vertex_ai", "gemini-3-flash", "embedding", CostRoute::PerToken)]
#[case("vertex_ai", "gemini-3-flash", "completion", CostRoute::PerCharacter)]
#[case("gemini", "gemini-2-flash", "completion", CostRoute::PerCharacter)]
fn cost_router_matches_provider_model_and_call_type(
    #[case] provider: &str,
    #[case] model: &str,
    #[case] call_type: &str,
    #[case] expected: CostRoute,
) {
    assert_eq!(cost_router(model, provider, call_type), expected);
}

#[rstest]
#[case(128_000, 128_000, 128_000.0 * 0.001, 128_000.0 * 0.002)]
#[case(128_001, 128_001, 128_001.0 * 0.003, 128_001.0 * 0.004)]
fn handle_128k_pricing_selects_each_side_independently(
    #[case] prompt_tokens: u64,
    #[case] completion_tokens: u64,
    #[case] expected_input: f64,
    #[case] expected_output: f64,
) {
    let info = json!({
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "input_cost_per_token_above_128k_tokens": 0.003,
        "output_cost_per_token_above_128k_tokens": 0.004
    });
    let usage = ChatUsage {
        prompt_tokens,
        completion_tokens,
        ..ChatUsage::default()
    };
    assert_eq!(
        handle_128k_pricing(&info, &usage),
        (expected_input, expected_output)
    );
}

#[rstest]
fn cost_per_token_applies_vertex_uplift_to_128k_rates() {
    let info = json!({
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "input_cost_per_token_above_128k_tokens": 0.003,
        "regional_endpoint_uplift_multiplier": 1.1
    });
    let usage = ChatUsage {
        prompt_tokens: 128_001,
        completion_tokens: 10,
        ..ChatUsage::default()
    };
    let global = cost_per_token(&usage, &info, None, Some("global"), at());
    let regional = cost_per_token(&usage, &info, None, Some("us-east5"), at());
    assert_eq!(global, (128_001.0 * 0.003, 10.0 * 0.002));
    assert!((regional.0 - global.0 * 1.1).abs() < 1e-10);
    assert!((regional.1 - global.1 * 1.1).abs() < 1e-10);
}

#[rstest]
fn cost_per_character_falls_back_per_side_and_keeps_served_tier() {
    let info = json!({
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "input_cost_per_token_flex": 0.0005,
        "output_cost_per_token_flex": 0.001,
        "input_cost_per_character": 0.0001,
        "regional_endpoint_uplift_multiplier": 1.1
    });
    let usage = ChatUsage {
        prompt_tokens: 100,
        completion_tokens: 20,
        ..ChatUsage::default()
    };
    let cost = cost_per_character(
        "gemini-3-model",
        &usage,
        &info,
        (Some(200.0), Some(80.0)),
        Some("flex"),
        Some("us-east5"),
        at(),
    );
    assert!((cost.0 - 200.0 * 0.0001 * 1.1).abs() < 1e-12);
    assert!((cost.1 - 20.0 * 0.001 * 1.1).abs() < 1e-12);
}

#[rstest]
fn cost_per_character_uses_python_dynamic_output_quantity() {
    let info = json!({
        "input_cost_per_character": 0.001,
        "output_cost_per_character": 0.002,
        "input_cost_per_character_above_128k_tokens": 0.003,
        "output_cost_per_character_above_128k_tokens": 0.004
    });
    let usage = ChatUsage {
        prompt_tokens: 10,
        completion_tokens: 7,
        ..ChatUsage::default()
    };
    let cost = cost_per_character(
        "gemini-3-model",
        &usage,
        &info,
        (Some(32_001.0), Some(32_001.0)),
        None,
        None,
        at(),
    );
    assert_eq!(cost, (32_001.0 * 0.003, 7.0 * 0.004));
}

#[rstest]
fn catalog_vertex_cost_routes_token_and_character_calls() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "vertex_ai/claude-model".to_owned(),
            json!({
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "input_cost_per_token_above_128k_tokens": 0.003
            }),
        ),
        (
            "vertex_ai/gemini-3-model".to_owned(),
            json!({
                "input_cost_per_character": 0.0001,
                "output_cost_per_character": 0.0002
            }),
        ),
    ]));
    let usage = ChatUsage {
        prompt_tokens: 10,
        completion_tokens: 5,
        ..ChatUsage::default()
    };
    let request = |model| ModelCostRequest {
        model,
        provider: Some("vertex_ai"),
        region: None,
        usage: &usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: at(),
        response_time_ms: None,
    };
    assert_eq!(
        vertex_cost(&catalog, request("claude-model"), "completion", None, None).unwrap(),
        (0.01, 0.01)
    );
    assert_eq!(
        vertex_cost(
            &catalog,
            request("gemini-3-model"),
            "completion",
            Some(100.0),
            Some(50.0),
        )
        .unwrap(),
        (0.01, 0.01)
    );
}

#[rstest]
#[case::bare_model("gemini-pro")]
#[case::provider_prefixed_model("vertex_ai/gemini-pro")]
fn vertex_cost_matches_models_without_dynamic_pricing_after_stripping_the_prefix(
    #[case] model: &str,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "vertex_ai/gemini-pro".to_owned(),
        json!({
            "input_cost_per_character": 1e-6,
            "input_cost_per_character_above_128k_tokens": 9e-6,
            "output_cost_per_character": 1e-6
        }),
    )]));
    let usage = ChatUsage::default();
    let (prompt, _) = vertex_cost(
        &catalog,
        ModelCostRequest {
            model,
            provider: Some("vertex_ai"),
            region: None,
            usage: &usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            at: at(),
            response_time_ms: None,
        },
        "completion",
        Some(40_000.0),
        Some(0.0),
    )
    .unwrap();
    assert!((prompt - 0.04).abs() < 1e-12);
}
