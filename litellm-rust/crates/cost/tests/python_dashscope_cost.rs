#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/llms/dashscope/test_dashscope_cost_calculator.py::TestDashscopeCostCalculator

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::dashscope_cost::{cost_per_token, extract_token_breakdown};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::{Value, json};

fn at(value: &str) -> Timestamp {
    value.parse().unwrap()
}

fn usage(value: Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": value})).unwrap().unwrap()
}

fn tiered_model() -> Value {
    json!({
        "output_cost_per_reasoning_token": 7e-6,
        "tiered_pricing": [
            {"range": [0, 1000], "input_cost_per_token": "1e-6", "output_cost_per_token": "2e-6", "cache_read_input_token_cost": 0.2e-6, "cache_creation_input_token_cost": 1.5e-6},
            {"range": [1000, 2000], "input_cost_per_token": "3e-6", "output_cost_per_token": "4e-6", "cache_read_input_token_cost": 0.6e-6, "cache_creation_input_token_cost": 4.5e-6}
        ]
    })
}

#[rstest]
#[case(1000, 1e-6, 2e-6)]
#[case(1001, 3e-6, 4e-6)]
#[case(2500, 3e-6, 4e-6)]
fn dashscope_cost_per_token_uses_one_input_selected_tier_for_both_directions(
    #[case] prompt_tokens: u64,
    #[case] input_rate: f64,
    #[case] output_rate: f64,
) {
    let usage = usage(json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 3000,
        "total_tokens": prompt_tokens + 3000
    }));
    let actual = cost_per_token(&usage, &tiered_model(), at("2026-01-01T12:00Z"));
    assert!((actual.0 - prompt_tokens as f64 * input_rate).abs() < 1e-12);
    assert!((actual.1 - 3000.0 * output_rate).abs() < 1e-12);
}

#[rstest]
fn dashscope_cost_per_token_prices_nested_cache_creation_and_reasoning_at_selected_tier() {
    let usage = usage(json!({
        "prompt_tokens": 1500,
        "completion_tokens": 200,
        "total_tokens": 1700,
        "prompt_tokens_details": {
            "text_tokens": 1500,
            "cached_tokens": 400,
            "cache_creation_input_tokens": 600
        },
        "completion_tokens_details": {"reasoning_tokens": 50}
    }));
    let breakdown = extract_token_breakdown(&usage);
    assert_eq!(breakdown.text_tokens, 500);
    assert_eq!(breakdown.total_input_tokens(), 1500);
    let actual = cost_per_token(&usage, &tiered_model(), at("2026-01-01T12:00Z"));
    assert!((actual.0 - (500.0 * 3e-6 + 400.0 * 0.6e-6 + 600.0 * 4.5e-6)).abs() < 1e-12);
    assert!((actual.1 - 200.0 * 4e-6).abs() < 1e-12);
}

#[rstest]
#[case(json!({"output_cost_per_token": 2e-6, "output_cost_per_reasoning_token": 0}), 50.0 * 2e-6)]
#[case(json!({"output_cost_per_token": 2e-6}), 200.0 * 2e-6)]
#[case(json!({"input_cost_per_token": 1e-6}), 50.0 * 2e-6 + 150.0 * 7e-6)]
fn dashscope_tier_reasoning_fallback_and_explicit_zero(
    #[case] tier_fields: Value,
    #[case] expected_completion: f64,
) {
    let tier = Value::Object(
        json!({"range": [0, 1000], "input_cost_per_token": 1e-6})
            .as_object()
            .unwrap()
            .iter()
            .chain(tier_fields.as_object().unwrap().iter())
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    );
    let model = json!({
        "output_cost_per_token": 2e-6,
        "output_cost_per_reasoning_token": 7e-6,
        "tiered_pricing": [tier]
    });
    let usage = usage(json!({
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "total_tokens": 700,
        "completion_tokens_details": {"reasoning_tokens": 150}
    }));
    let actual = cost_per_token(&usage, &model, at("2026-01-01T12:00Z"));
    assert!((actual.1 - expected_completion).abs() < 1e-12);
}

#[rstest]
#[case("2026-09-03T17:25Z", 0.1e-6, 0.4e-6, 0.8e-6)]
#[case("2026-09-03T09:00Z", 3e-6, 4e-6, 4e-6)]
fn dashscope_off_peak_rates_override_selected_tier_and_reasoning(
    #[case] billed_at: &str,
    #[case] input_rate: f64,
    #[case] output_rate: f64,
    #[case] reasoning_rate: f64,
) {
    let model = json!({
        "tiered_pricing": [{"range": [0, 1000], "input_cost_per_token": 3e-6, "output_cost_per_token": 4e-6}],
        "off_peak_pricing": {
            "hours_utc": "14:00-00:00",
            "input_cost_per_token": 0.1e-6,
            "output_cost_per_token": 0.4e-6,
            "output_cost_per_reasoning_token": 0.8e-6
        }
    });
    let usage = usage(json!({
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "total_tokens": 600,
        "completion_tokens_details": {"reasoning_tokens": 40}
    }));
    let actual = cost_per_token(&usage, &model, at(billed_at));
    assert!((actual.0 - 500.0 * input_rate).abs() < 1e-12);
    assert!((actual.1 - (60.0 * output_rate + 40.0 * reasoning_rate)).abs() < 1e-12);
}

#[rstest]
#[case("dashscope")]
#[case("qwencloud")]
#[case("qwen_ai_platform")]
fn dashscope_cost_per_token_dispatches_brand_aliases(#[case] provider: &str) {
    let model_info = json!({
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 4e-6,
        "cache_read_input_token_cost": 0.5e-6
    });
    let usage = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 40}
    }));
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        format!("{provider}/model"),
        model_info.clone(),
    )]));
    let actual = litellm_cost::cost_calculator::cost_per_token(
        &catalog,
        ModelCostRequest {
            model: "model",
            provider: Some(provider),
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
    assert_eq!(
        actual,
        cost_per_token(&usage, &model_info, at("2026-01-01T12:00Z"))
    );
    assert!((actual.0 - (60.0 * 2e-6 + 40.0 * 0.5e-6)).abs() < 1e-12);
}
