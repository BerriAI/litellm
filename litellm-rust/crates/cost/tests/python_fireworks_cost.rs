#![allow(clippy::disallowed_types)]

// mirrors: unit/llms/fireworks_ai/test_fireworks_ai_cost_calculator.py

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::fireworks_cost::{
    FireworksThresholds, cost_per_token, get_base_model_for_pricing,
};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn at(value: &str) -> Timestamp {
    value.parse().unwrap()
}

fn usage(value: Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": value})).unwrap().unwrap()
}

#[rstest]
#[case("MODEL-4B", "fireworks-ai-up-to-4b")]
#[case("model-16b", "fireworks-ai-4.1b-to-16b")]
#[case("model-17b", "fireworks-ai-above-16b")]
#[case("model-7x8b", "fireworks-ai-moe-up-to-56b")]
#[case("model-8x22b", "fireworks-ai-56b-to-176b")]
#[case("model-8x30b", "fireworks-ai-above-16b")]
#[case("model-without-size", "fireworks-ai-default")]
#[case("foo-99999999999999999999b", "fireworks-ai-above-16b")]
#[case("99999999999999999999x2b", "fireworks-ai-up-to-4b")]
fn fireworks_model_size_category_matches_python_pattern_order(
    #[case] model: &str,
    #[case] expected: &str,
) {
    assert_eq!(
        get_base_model_for_pricing(model, FireworksThresholds::default()),
        expected
    );
}

#[rstest]
fn fireworks_model_size_categories_accept_configured_thresholds() {
    let thresholds = FireworksThresholds {
        small: 8,
        medium: 32,
        moe_small: 80,
        moe_medium: 200,
    };
    assert_eq!(
        get_base_model_for_pricing("model-8b", thresholds),
        "fireworks-ai-up-to-4b"
    );
    assert_eq!(
        get_base_model_for_pricing("model-8x10b", thresholds),
        "fireworks-ai-moe-up-to-56b"
    );
}

#[rstest]
#[case(false, 2e-6)]
#[case(true, 5e-6)]
fn fireworks_catalog_prefers_exact_model_then_falls_back_to_size_category(
    #[case] exact_model_exists: bool,
    #[case] selected_input_rate: f64,
) {
    let category = (
        "fireworks_ai/fireworks-ai-moe-up-to-56b".to_owned(),
        json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6}),
    );
    let exact = exact_model_exists.then(|| {
        (
            "fireworks_ai/accounts/models/model-7x8b".to_owned(),
            json!({"input_cost_per_token": 5e-6, "output_cost_per_token": 4e-6}),
        )
    });
    let catalog = ModelInfoCatalog::new(
        [Some(category), exact]
            .into_iter()
            .flatten()
            .collect::<HashMap<_, _>>(),
    );
    let usage = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120
    }));
    let (prompt, completion) = litellm_cost::cost_calculator::cost_per_token(
        &catalog,
        ModelCostRequest {
            model: "accounts/models/model-7x8b",
            provider: Some("fireworks_ai"),
            region: None,
            usage: &usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            at: at("2026-01-01T12:00Z"),
            response_time_ms: None,
        },
    )
    .unwrap();
    assert!((prompt - 100.0 * selected_input_rate).abs() < 1e-12);
    assert!((completion - 20.0 * 4e-6).abs() < 1e-12);
}

#[rstest]
#[case("2026-01-01T18:00Z", 1e-6, 2e-6)]
#[case("2026-01-01T12:00Z", 2e-6, 4e-6)]
fn fireworks_cost_per_token_derives_cache_read_from_selected_window(
    #[case] billed_at: &str,
    #[case] input_rate: f64,
    #[case] output_rate: f64,
) {
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "off_peak_pricing": {
            "hours_utc": "16:30-00:30",
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6
        }
    });
    let usage = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 40}
    }));
    let (prompt, completion) = cost_per_token(&usage, &model_info, at(billed_at));
    assert!((prompt - (60.0 * input_rate + 40.0 * input_rate * 0.5)).abs() < 1e-12);
    assert!((completion - 20.0 * output_rate).abs() < 1e-12);
}
