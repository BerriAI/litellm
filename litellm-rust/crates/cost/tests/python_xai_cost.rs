#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/llms/xai/test_xai_cost_calculator.py::TestXAICostCalculator

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::usage_dispatch::get_usage_object;
use litellm_cost::xai_cost::{
    apply_server_side_tool_usage_details_to_usage, cost_per_token, cost_per_web_search_request,
    reported_cost, web_search_cost_per_call_from_model_info,
};
use rstest::rstest;
use serde_json::{Value, json};

fn at() -> Timestamp {
    "2026-01-01T12:00Z".parse().unwrap()
}

fn usage(value: Value) -> litellm_cost::responses_usage::ChatUsage {
    get_usage_object(&json!({"usage": value})).unwrap().unwrap()
}

fn model_info() -> Value {
    json!({
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "input_cost_per_token_above_200k_tokens": 3e-6,
        "output_cost_per_token_above_200k_tokens": 4e-6
    })
}

#[rstest]
#[case(50, 200_070)]
#[case(70, 200_070)]
fn xai_cost_per_token_bills_raw_and_normalized_reasoning_once_at_inclusive_threshold(
    #[case] completion_tokens: u64,
    #[case] total_tokens: u64,
) {
    let usage = usage(json!({
        "prompt_tokens": 200_000,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_details": {"reasoning_tokens": 20}
    }));
    let direct = cost_per_token(&usage, &model_info(), at());
    assert!((direct.0 - 200_000.0 * 3e-6).abs() < 1e-12);
    assert!((direct.1 - 70.0 * 4e-6).abs() < 1e-12);
    let catalog = ModelInfoCatalog::new(HashMap::from([("xai/model".to_owned(), model_info())]));
    assert_eq!(
        litellm_cost::cost_calculator::cost_per_token(
            &catalog,
            ModelCostRequest {
                model: "model",
                provider: Some("xai"),
                region: None,
                usage: &usage,
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at: at(),
                response_time_ms: None,
            }
        )
        .unwrap(),
        direct
    );
}

#[rstest]
#[case(0.0037756)]
#[case(0.0)]
fn xai_cost_per_token_prefers_nonnegative_provider_total_without_model_metadata(
    #[case] provider_cost: f64,
) {
    let usage = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost": provider_cost
    }));
    assert_eq!(reported_cost(&usage), Some(provider_cost));
    let catalog = ModelInfoCatalog::new(HashMap::new());
    assert_eq!(
        litellm_cost::cost_calculator::cost_per_token(
            &catalog,
            ModelCostRequest {
                model: "unmapped",
                provider: Some("xai"),
                region: None,
                usage: &usage,
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at: at(),
                response_time_ms: None,
            }
        )
        .unwrap(),
        (0.0, provider_cost)
    );
}

#[rstest]
fn xai_cost_per_token_ignores_negative_provider_total() {
    let usage = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost": -0.5
    }));
    assert_eq!(reported_cost(&usage), None);
    let (prompt, completion) = cost_per_token(&usage, &model_info(), at());
    assert!((prompt - 100e-6).abs() < 1e-12);
    assert!((completion - 100e-6).abs() < 1e-12);
}

#[rstest]
#[case(json!({"web_search_calls": 2}), Some(2))]
#[case(json!({"web_search_calls": "3"}), Some(3))]
#[case(json!({"web_search_calls": "invalid"}), None)]
fn apply_server_side_tool_usage_details_to_usage_mirrors_positive_web_search_calls(
    #[case] details: Value,
    #[case] expected_web_search_requests: Option<u64>,
) {
    let usage = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
    }));
    let updated = apply_server_side_tool_usage_details_to_usage(&usage, Some(&details));
    assert_eq!(
        updated
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.web_search_requests),
        expected_web_search_requests
    );
    assert_eq!(
        updated.extra.get("server_side_tool_usage_details"),
        Some(&details)
    );
    assert!(
        (cost_per_web_search_request(&updated, &json!({}), 0.002)
            - expected_web_search_requests.unwrap_or(0) as f64 * 0.002)
            .abs()
            < 1e-12
    );
    assert_eq!(usage.extra.get("server_side_tool_usage_details"), None);
}

#[rstest]
#[case(json!({"search_context_size_medium": 0.009, "search_context_size_low": 0.003}), 0.009)]
#[case(json!({"search_context_size_medium": 0, "search_context_size_low": "0.003"}), 0.003)]
#[case(json!({"search_context_size_medium": "invalid", "search_context_size_high": 0.007}), 0.007)]
#[case(json!({"search_context_size_medium": 0, "search_context_size_low": "invalid"}), 0.002)]
fn xai_web_search_price_selects_first_positive_context_rate(
    #[case] prices: Value,
    #[case] expected: f64,
) {
    let model_info = json!({"search_context_cost_per_query": prices});
    assert_eq!(
        web_search_cost_per_call_from_model_info(&model_info, 0.002),
        expected
    );
    let usage = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "server_side_tool_usage_details": {"web_search_calls": 3}
    }));
    assert!(
        (cost_per_web_search_request(&usage, &model_info, 0.002) - 3.0 * expected).abs() < 1e-12
    );
}

#[rstest]
fn xai_web_search_cost_is_included_in_a_reported_total() {
    let usage = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cost": 0.03,
        "server_side_tool_usage_details": {"web_search_calls": 3}
    }));
    assert_eq!(cost_per_web_search_request(&usage, &json!({}), 0.002), 0.0);
}

#[rstest]
fn apply_server_side_tool_usage_details_preserves_existing_cache_usage() {
    let usage = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 7}
    }));
    let updated = apply_server_side_tool_usage_details_to_usage(
        &usage,
        Some(&json!({"web_search_calls": 4})),
    );
    let prompt = updated.prompt_tokens_details.unwrap();
    assert_eq!(prompt.cached_tokens, 7);
    assert_eq!(prompt.web_search_requests, Some(4));
}

#[rstest]
#[case::padded_string(json!({"search_context_cost_per_query": {"search_context_size_medium": " 0.02 "}}), 0.02)]
#[case::skips_unparseable_and_zero(json!({"search_context_cost_per_query": {"search_context_size_medium": "n/a", "search_context_size_low": 0, "search_context_size_high": 0.03}}), 0.03)]
#[case::default_when_nothing_prices(json!({}), 0.005)]
fn web_search_cost_per_call_takes_the_first_positive_float(
    #[case] model_info: Value,
    #[case] expected: f64,
) {
    assert_eq!(
        litellm_cost::xai_cost::web_search_cost_per_call_from_model_info(&model_info, 0.005),
        expected
    );
}
