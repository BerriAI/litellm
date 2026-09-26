#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/test_cost_calculator.py::test_cost_discount_not_applied_to_other_providers
use litellm_cost::error::CostError;

use std::collections::HashMap;

use jiff::Timestamp;
use litellm_cost::catalog::{ModelCostRequest, ModelInfoCatalog};
use litellm_cost::completion_cost::{
    apply_cost_discount, apply_cost_margin, completion_cost, get_response_cost_from_hidden_params,
    response_cost_calculator,
};
use litellm_cost::cost_calculator::{
    BuiltInToolCharge, CompletionCostRequest, ResponseCostRequest,
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
#[case(json!({"additional_headers": {"llm_provider-x-litellm-response-cost": "invalid"}}), Err(CostError::InvalidProviderCost))]
fn get_response_cost_from_hidden_params_parses_provider_value(
    #[case] hidden_params: serde_json::Value,
    #[case] expected: Result<Option<f64>, CostError>,
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
    let calculated = litellm_cost::cost_calculator::completion_cost(&catalog, completion).unwrap();
    assert_eq!(calculated.built_in_tools, 0.01);
    assert_eq!(calculated.additional, 0.02);
    assert!((calculated.input + calculated.output - 0.0004).abs() < 1e-12);
    assert!((calculated.original - 0.0304).abs() < 1e-12);
    assert!((calculated.total - 0.033832).abs() < 1e-12);
    assert_eq!(
        litellm_cost::cost_calculator::response_cost_calculator(
            &catalog,
            ResponseCostRequest {
                completion,
                cache_hit: false,
                hidden_params: &json!({}),
            }
        ),
        Ok(calculated.total)
    );
    assert_eq!(
        litellm_cost::cost_calculator::response_cost_calculator(
            &catalog,
            ResponseCostRequest {
                completion,
                cache_hit: true,
                hidden_params: &json!({}),
            }
        ),
        Ok(0.0)
    );
}

#[rstest]
#[case::number(json!({"provider": 0.1}), (1.8, 0.1, 0.2))]
#[case::boolean_is_an_int(json!({"provider": true}), (0.0, 1.0, 2.0))]
#[case::string_is_not_applied(json!({"provider": "0.1"}), (2.0, 0.0, 0.0))]
fn apply_cost_discount_uses_only_numeric_config(
    #[case] config: serde_json::Value,
    #[case] expected: (f64, f64, f64),
) {
    let (cost, percent, amount) = apply_cost_discount(2.0, Some("provider"), &config);
    assert!((cost - expected.0).abs() < 1e-12);
    assert_eq!(percent, expected.1);
    assert!((amount - expected.2).abs() < 1e-12);
}

#[rstest]
#[case::scalar_number(json!({"provider": 0.1}), (2.2, 0.1, 0.0, 0.2))]
#[case::scalar_string_is_ignored(json!({"provider": "0.1"}), (2.0, 0.0, 0.0, 0.0))]
#[case::provider_null_does_not_fall_back_to_global(json!({"provider": null, "global": 0.5}), (2.0, 0.0, 0.0, 0.0))]
#[case::dict_strings_convert_like_float(json!({"provider": {"percentage": " 0.1 ", "fixed_amount": "1"}}), (3.2, 0.1, 1.0, 1.2))]
#[case::dict_with_only_a_fixed_amount(json!({"provider": {"fixed_amount": 0.5}}), (2.5, 0.0, 0.5, 0.5))]
#[case::empty_provider_uses_global(json!({"": 0.9, "global": 0.5}), (3.0, 0.5, 0.0, 1.0))]
fn apply_cost_margin_reads_config_like_python(
    #[case] config: serde_json::Value,
    #[case] expected: (f64, f64, f64, f64),
) {
    let provider = if config.get("").is_some() {
        Some("")
    } else {
        Some("provider")
    };
    let (cost, percent, fixed, total) = apply_cost_margin(2.0, provider, &config);
    assert!((cost - expected.0).abs() < 1e-12);
    assert_eq!((percent, fixed), (expected.1, expected.2));
    assert!((total - expected.3).abs() < 1e-12);
}
