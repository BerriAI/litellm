#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/llms/azure_ai/test_azure_ai_cost_calculator.py::TestAzureModelRouterFlatCost
use litellm_cost::azure_ai_cost::azure_ai_cost_per_token;
use litellm_cost::cost_calculator::cost_per_token;
use litellm_cost::error::CostError;

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::azure_ai_cost::{
    calculate_azure_model_router_flat_cost, is_azure_model_router, is_router_fee_entry,
    router_fee_entry_name, router_fee_name,
};
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::responses_usage::ChatUsage;
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

fn catalog() -> ModelInfoCatalog {
    ModelInfoCatalog::new(HashMap::from([
        (
            "azure_ai/model_router".to_owned(),
            json!({"input_cost_per_token": 0.001}),
        ),
        (
            "azure_ai/model-router".to_owned(),
            json!({"input_cost_per_token": 0.001}),
        ),
        (
            "azure_ai/routed".to_owned(),
            json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003, "input_cost_per_token_priority": 0.005, "output_cost_per_token_priority": 0.007}),
        ),
    ]))
}

fn request<'a>(
    model: &'a str,
    usage: &'a ChatUsage,
    tier: Option<&'a str>,
) -> ModelCostRequest<'a> {
    ModelCostRequest {
        model,
        provider: Some("azure_ai"),
        region: None,
        usage,
        service_tier: tier,
        data_residency: None,
        vertex_location: None,
        at: "2026-09-22T12:00:00Z".parse::<Timestamp>().unwrap(),
        response_time_ms: None,
    }
}

#[rstest]
#[case("azure-model-router", true)]
#[case("MODEL_ROUTER/deployment", true)]
#[case("my-model-router-deployment", true)]
#[case("regular-deployment", false)]
fn is_azure_model_router_matches_router_names(#[case] model: &str, #[case] expected: bool) {
    assert_eq!(is_azure_model_router(model), expected);
}

#[rstest]
fn router_fee_helpers_skip_fee_entry_and_select_requested_router() {
    assert!(is_router_fee_entry("azure_ai/model-router"));
    assert!(!is_router_fee_entry("azure_ai/azure_ai/model-router"));
    assert_eq!(
        router_fee_entry_name("azure_ai/model-router"),
        "model-router"
    );
    assert_eq!(router_fee_entry_name("azure-model-router"), "model_router");
    assert_eq!(
        router_fee_name("model_router", Some("azure_ai/model-router")),
        None
    );
    assert_eq!(
        router_fee_name("routed", Some("azure_ai/model-router")),
        Some("azure_ai/model-router")
    );
    assert_eq!(
        router_fee_name("model_router/deployment", Some("azure_ai/model-router")),
        Some("model_router/deployment")
    );
}

#[rstest]
fn calculate_azure_model_router_flat_cost_bills_full_prompt_count() {
    let info = json!({"input_cost_per_token": 0.001});
    assert_eq!(
        calculate_azure_model_router_flat_cost("model_router/deployment", 1000, &info),
        1.0
    );
    assert_eq!(
        calculate_azure_model_router_flat_cost("ordinary", 1000, &info),
        0.0
    );
}

#[rstest]
fn azure_ai_cost_per_token_prices_unmapped_router_and_cached_prompt_once() {
    let usage = get_usage_object(&json!({"usage": {"prompt_tokens": 1000, "completion_tokens": 500, "cache_read_input_tokens": 200}}))
        .unwrap()
        .unwrap();
    let actual = azure_ai_cost_per_token(
        &catalog(),
        request("azure-model-router", &usage, None),
        None,
    )
    .unwrap();
    assert_eq!(actual, (1.0, 0.0));
    let both_names = azure_ai_cost_per_token(
        &catalog(),
        request("model_router/deployment", &usage, None),
        Some("azure_ai/model-router"),
    )
    .unwrap();
    assert_eq!(both_names, (1.0, 0.0));
}

#[rstest]
fn azure_ai_cost_per_token_adds_fee_for_routed_response_model_only_once() {
    let usage =
        get_usage_object(&json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20}}))
            .unwrap()
            .unwrap();
    let catalog = catalog();
    let response_only = cost_per_token(&catalog, request("routed", &usage, None)).unwrap();
    assert_eq!(response_only, (0.2, 0.06));
    let routed = azure_ai_cost_per_token(
        &catalog,
        request("routed", &usage, None),
        Some("azure_ai/model_router"),
    )
    .unwrap();
    assert!((routed.0 - (response_only.0 + 100.0 * 0.001)).abs() < 1e-12);
    assert_eq!(routed.1, response_only.1);
    assert_eq!(
        azure_ai_cost_per_token(
            &catalog,
            request("model_router", &usage, None),
            Some("azure_ai/model_router"),
        )
        .unwrap(),
        (0.1, 0.0)
    );
}

#[rstest]
fn azure_ai_cost_per_token_passes_service_tier_to_response_model() {
    let usage =
        get_usage_object(&json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20}}))
            .unwrap()
            .unwrap();
    let catalog = catalog();
    let standard = azure_ai_cost_per_token(
        &catalog,
        request("routed", &usage, None),
        Some("model-router"),
    )
    .unwrap();
    let priority = azure_ai_cost_per_token(
        &catalog,
        request("routed", &usage, Some("priority")),
        Some("model-router"),
    )
    .unwrap();
    assert!((priority.0 - (100.0 * 0.005 + 100.0 * 0.001)).abs() < 1e-12);
    assert!((priority.1 - 20.0 * 0.007).abs() < 1e-12);
    assert!(priority.0 > standard.0);
    assert!(priority.1 > standard.1);
}

#[rstest]
fn azure_ai_cost_per_token_rejects_unknown_non_router_model() {
    let catalog = catalog();
    assert_eq!(
        azure_ai_cost_per_token(
            &catalog,
            request("unknown", &ChatUsage::default(), None),
            None
        ),
        Err(CostError::ModelNotFound)
    );
}

#[rstest]
fn azure_ai_cost_per_token_preserves_wall_clock_precedence_over_router_fee() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "azure_ai/model_router".to_owned(),
            json!({"input_cost_per_token": 0.001}),
        ),
        (
            "azure_ai/duration".to_owned(),
            json!({"mode": "chat", "input_cost_per_second": 0.02}),
        ),
    ]));
    let usage =
        get_usage_object(&json!({"usage": {"prompt_tokens": 100, "completion_tokens": 20}}))
            .unwrap()
            .unwrap();
    let actual = azure_ai_cost_per_token(
        &catalog,
        ModelCostRequest {
            response_time_ms: Some(2000.0),
            ..request("duration", &usage, None)
        },
        Some("model_router"),
    )
    .unwrap();
    assert_eq!(actual, (0.04, 0.0));
}
