#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/litellm_core_utils/llm_cost_calc/test_llm_cost_calc_utils.py::test_token_type_cost_breakdown_reconciles_with_generic_total

use std::collections::BTreeMap;

use litellm_cost::billed_token_rates::{BilledTokenRates, TokenTypeCostBreakdown};
use litellm_cost::completion_cost::completion_cost;
use litellm_cost::cost_breakdown::{PricingBasis, cost_breakdown};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case(0.0, None)]
#[case(0.003, Some(0.003))]
fn cost_breakdown_keeps_python_cost_components_and_positive_token_details(
    #[case] cache_read_cost: f64,
    #[case] expected_cache_read_cost: Option<f64>,
) {
    let additional_costs = BTreeMap::from([("router_fee".to_owned(), 0.2)]);
    let cost = completion_cost(
        1.0,
        2.0,
        0.5,
        &[0.2],
        Some("provider"),
        &json!({"provider": 0.1}),
        &json!({"provider": {"percentage": 0.2, "fixed_amount": 0.05}}),
    );
    let rates = BilledTokenRates {
        input_cost_per_token: 0.01,
        output_cost_per_token: 0.02,
        cache_read_input_token_cost: 0.005,
        cache_read_input_audio_token_cost: 0.005,
        cache_creation_input_token_cost: 0.015,
        cache_creation_input_token_cost_above_1hr: 0.03,
        output_cost_per_reasoning_token: 0.025,
    };
    let breakdown = cost_breakdown(
        cost,
        Some(TokenTypeCostBreakdown {
            reasoning_cost: 0.02,
            cache_read_cost,
            cache_creation_cost: 0.01,
            rates: Some(rates),
        }),
        Some(additional_costs.clone()),
        PricingBasis {
            service_tier: Some("priority"),
            data_residency: Some("eu"),
            vertex_location: None,
        },
    );

    assert_eq!(breakdown.input_cost, 1.0);
    assert_eq!(breakdown.output_cost, 2.0);
    assert_eq!(breakdown.tool_usage_cost, 0.5);
    assert_eq!(breakdown.additional_costs, Some(additional_costs));
    assert_eq!(breakdown.original_cost, 3.7);
    assert_eq!(breakdown.discount_percent, 0.1);
    assert!((breakdown.discount_amount - 0.37).abs() < 1e-12);
    assert_eq!(breakdown.margin_percent, 0.2);
    assert_eq!(breakdown.margin_fixed_amount, 0.05);
    assert!((breakdown.margin_total_amount - 0.716).abs() < 1e-12);
    assert!((breakdown.total_cost - 4.046).abs() < 1e-12);
    assert_eq!(breakdown.cache_read_cost, expected_cache_read_cost);
    assert_eq!(breakdown.cache_creation_cost, Some(0.01));
    assert_eq!(breakdown.reasoning_cost, Some(0.02));
    assert_eq!(breakdown.billed_token_rates, Some(rates));
    assert_eq!(breakdown.service_tier.as_deref(), Some("priority"));
    assert_eq!(breakdown.data_residency.as_deref(), Some("eu"));
    assert_eq!(breakdown.vertex_location, None);
}

#[rstest]
#[case(None)]
#[case(Some(BTreeMap::new()))]
fn cost_breakdown_omits_unprovided_or_empty_details(
    #[case] additional_costs: Option<BTreeMap<String, f64>>,
) {
    let cost = completion_cost(0.0, 0.0, 0.0, &[], None, &json!({}), &json!({}));
    let breakdown = cost_breakdown(cost, None, additional_costs, PricingBasis::default());

    assert_eq!(breakdown.total_cost, 0.0);
    assert_eq!(breakdown.additional_costs, None);
    assert_eq!(breakdown.cache_read_cost, None);
    assert_eq!(breakdown.cache_creation_cost, None);
    assert_eq!(breakdown.reasoning_cost, None);
    assert_eq!(breakdown.billed_token_rates, None);
}
