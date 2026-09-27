#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/llms/perplexity/test_perplexity_cost_calculator.py::TestPerplexityCostCalculator

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::perplexity_cost::cost_per_token;
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
#[case("2026-09-03T17:25Z", 1e-7, 2e-7)]
#[case("2026-09-03T09:00Z", 1e-6, 1e-6)]
fn perplexity_cost_per_token_keeps_citation_search_and_reasoning_fees_during_off_peak(
    #[case] billed_at: &str,
    #[case] input_rate: f64,
    #[case] output_rate: f64,
) {
    let model_info = json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 1e-6,
        "output_cost_per_reasoning_token": 3e-6,
        "citation_cost_per_token": 2e-6,
        "search_context_cost_per_query": {"search_context_size_low": 0.005},
        "off_peak_pricing": {
            "hours_utc": "14:00-00:00",
            "input_cost_per_token": 1e-7,
            "output_cost_per_token": 2e-7
        }
    });
    let usage = usage(json!({
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "citation_tokens": 100,
        "prompt_tokens_details": {"web_search_requests": 2},
        "completion_tokens_details": {"reasoning_tokens": 50}
    }));
    let expected = (
        1000.0 * input_rate + 100.0 * 2e-6,
        150.0 * output_rate + 50.0 * 3e-6 + 2.0 * 0.005,
    );
    let actual = cost_per_token(&usage, &model_info, at(billed_at));
    assert!((actual.0 - expected.0).abs() < 1e-12);
    assert!((actual.1 - expected.1).abs() < 1e-12);
    let catalog =
        ModelInfoCatalog::new(HashMap::from([("perplexity/model".to_owned(), model_info)]));
    assert_eq!(
        litellm_cost::cost_calculator::cost_per_token(
            &catalog,
            ModelCostRequest {
                model: "model",
                provider: Some("perplexity"),
                region: None,
                usage: &usage,
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at: at(billed_at),
                response_time_ms: None,
            }
        )
        .unwrap(),
        actual
    );
}

#[rstest]
#[case(json!(0.008))]
#[case(json!({"input_tokens_cost": 0.001, "request_cost": 0.007, "total_cost": 0.008}))]
fn perplexity_cost_per_token_prefers_provider_total_without_model_pricing(#[case] cost: Value) {
    let usage = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost": cost
    }));
    assert_eq!(usage.cost, Some(0.008));
    let catalog = ModelInfoCatalog::new(HashMap::new());
    assert_eq!(
        litellm_cost::cost_calculator::cost_per_token(
            &catalog,
            ModelCostRequest {
                model: "unmapped",
                provider: Some("perplexity"),
                region: None,
                usage: &usage,
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at: at("2026-09-03T17:25Z"),
                response_time_ms: None,
            }
        )
        .unwrap(),
        (0.0, 0.008)
    );
}

#[rstest]
fn perplexity_response_usage_preserves_provider_cost_object() {
    let usage = usage(json!({
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": {"total_cost": 0.008}
    }));
    assert_eq!(usage.cost, Some(0.008));
    assert_eq!(
        cost_per_token(&usage, &json!({}), at("2026-09-03T17:25Z")),
        (0.0, 0.008)
    );
}

#[rstest]
fn perplexity_cost_per_token_uses_flat_search_rate_and_basic_fallbacks() {
    let usage = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "citation_tokens": 25,
        "prompt_tokens_details": {"web_search_requests": 2}
    }));
    let model_info = json!({
        "input_cost_per_token": "0.000002",
        "output_cost_per_token": 8e-6,
        "search_queries_cost_per_query": 0.003,
        "search_context_cost_per_query": {"search_context_size_low": 0.005}
    });
    let actual = cost_per_token(&usage, &model_info, at("2026-09-03T17:25Z"));
    assert!((actual.0 - 100.0 * 2e-6).abs() < 1e-12);
    assert!((actual.1 - (50.0 * 8e-6 + 2.0 * 0.003)).abs() < 1e-12);
}
