#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/test_cost_calculator.py::test_cost_discount_not_applied_to_other_providers

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{
    BuiltInToolCharge, CompletionCostRequest, ModelCostRequest, ModelInfoCatalog,
    ResponseCostRequest,
};
use litellm_cost::completion_cost::{
    ResponseCostError, apply_cost_discount, apply_cost_margin, completion_cost,
    get_response_cost_from_hidden_params, response_cost_calculator,
};
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

#[rstest]
fn completion_cost_applies_discount_before_provider_margin() {
    let actual = completion_cost(
        1.0,
        2.0,
        0.5,
        &[0.25],
        Some("provider"),
        &json!({"provider": 0.1}),
        &json!({"provider": {"percentage": 0.2, "fixed_amount": 0.1}, "global": 0.5}),
    );
    assert_eq!(
        (
            actual.input,
            actual.output,
            actual.built_in_tools,
            actual.additional
        ),
        (1.0, 2.0, 0.5, 0.25)
    );
    assert_eq!(
        actual.original,
        actual.input + actual.output + actual.built_in_tools + actual.additional
    );
    assert_eq!(actual.original, 3.75);
    assert_eq!(actual.discounted, 3.375);
    assert_eq!(actual.discount_percent, 0.1);
    assert_eq!(actual.discount_amount, 0.375);
    assert!((actual.margin_total_amount - 0.775).abs() < 1e-12);
    assert!((actual.total - 4.15).abs() < 1e-12);
}

#[rstest]
fn apply_cost_discount_ignores_other_providers() {
    assert_eq!(
        apply_cost_discount(2.0, Some("provider"), &json!({"other": 0.1})),
        (2.0, 0.0, 0.0)
    );
}

#[rstest]
fn apply_cost_margin_uses_global_when_provider_has_no_entry() {
    assert_eq!(
        apply_cost_margin(2.0, Some("provider"), &json!({"other": 0.1, "global": 0.2})),
        (2.4, 0.2, 0.0, 0.4)
    );
}

#[rstest]
#[case(json!({"additional_headers": {"llm_provider-x-litellm-response-cost": "1.25"}}), Ok(Some(1.25)))]
#[case(json!({"additional_headers": {"llm_provider-x-litellm-response-cost": " 1.25 "}}), Ok(Some(1.25)))]
#[case(json!({"additional_headers": {"llm_provider-x-litellm-response-cost": true}}), Ok(Some(1.0)))]
#[case(json!({"additional_headers": {"llm_provider-x-litellm-response-cost": null}}), Ok(None))]
#[case(json!({"additional_headers": {"other": "1.25"}}), Ok(None))]
#[case(json!({"additional_headers": {"llm_provider-x-litellm-response-cost": "invalid"}}), Err(ResponseCostError::InvalidProviderCost))]
fn get_response_cost_from_hidden_params_parses_provider_value(
    #[case] hidden_params: serde_json::Value,
    #[case] expected: Result<Option<f64>, ResponseCostError>,
) {
    assert_eq!(
        get_response_cost_from_hidden_params(&hidden_params),
        expected
    );
}

#[rstest]
fn response_cost_calculator_honors_cache_then_provider_then_calculation() {
    let reported = json!({"additional_headers": {"llm_provider-x-litellm-response-cost": 1.25}});
    assert_eq!(response_cost_calculator(true, &reported, 2.0), Ok(0.0));
    assert_eq!(response_cost_calculator(false, &reported, 2.0), Ok(1.25));
    assert_eq!(response_cost_calculator(false, &json!({}), 2.0), Ok(2.0));
}

#[rstest]
fn model_info_catalog_completion_and_response_cost_use_selected_metadata() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "provider/model".to_owned(),
        json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6}),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    }}))
    .unwrap()
    .unwrap();
    let at: Timestamp = "2026-01-01T18:00Z".parse().unwrap();
    let discount = json!({"provider": 0.1});
    let margin = json!({"global": {"percentage": 0.2, "fixed_amount": 0.001}});
    let completion = CompletionCostRequest {
        token: ModelCostRequest {
            model: "model",
            provider: Some("provider"),
            region: None,
            usage: &usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            at,
            response_time_ms: None,
        },
        built_in_tools: BuiltInToolCharge::Provided(0.01),
        additional_costs: &[0.02],
        discount_config: &discount,
        margin_config: &margin,
    };
    let calculated = catalog.completion_cost(completion).unwrap();
    assert_eq!(calculated.built_in_tools, 0.01);
    assert_eq!(calculated.additional, 0.02);
    assert!((calculated.input + calculated.output - 0.0004).abs() < 1e-12);
    assert!((calculated.original - 0.0304).abs() < 1e-12);
    assert!((calculated.total - 0.033832).abs() < 1e-12);
    assert_eq!(
        catalog.response_cost_calculator(ResponseCostRequest {
            completion,
            cache_hit: false,
            hidden_params: &json!({}),
        }),
        Ok(calculated.total)
    );
    assert_eq!(
        catalog.response_cost_calculator(ResponseCostRequest {
            completion,
            cache_hit: true,
            hidden_params: &json!({}),
        }),
        Ok(0.0)
    );
}
