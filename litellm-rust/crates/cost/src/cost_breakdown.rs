use std::collections::BTreeMap;

use crate::billed_token_rates::{BilledTokenRates, TokenTypeCostBreakdown};
use crate::completion_cost::CompletionCost;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PricingBasis<'a> {
    pub service_tier: Option<&'a str>,
    pub data_residency: Option<&'a str>,
    pub vertex_location: Option<&'a str>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CostBreakdown {
    pub service_tier: Option<String>,
    pub data_residency: Option<String>,
    pub vertex_location: Option<String>,
    pub input_cost: f64,
    pub output_cost: f64,
    pub total_cost: f64,
    pub tool_usage_cost: f64,
    pub cache_read_cost: Option<f64>,
    pub cache_creation_cost: Option<f64>,
    pub reasoning_cost: Option<f64>,
    pub additional_costs: Option<BTreeMap<String, f64>>,
    pub original_cost: f64,
    pub discount_percent: f64,
    pub discount_amount: f64,
    pub margin_percent: f64,
    pub margin_fixed_amount: f64,
    pub margin_total_amount: f64,
    pub billed_token_rates: Option<BilledTokenRates>,
}

pub fn cost_breakdown(
    cost: CompletionCost,
    token_breakdown: Option<TokenTypeCostBreakdown>,
    additional_costs: Option<BTreeMap<String, f64>>,
    basis: PricingBasis<'_>,
) -> CostBreakdown {
    CostBreakdown {
        service_tier: basis.service_tier.map(str::to_owned),
        data_residency: basis.data_residency.map(str::to_owned),
        vertex_location: basis.vertex_location.map(str::to_owned),
        input_cost: cost.input,
        output_cost: cost.output,
        total_cost: cost.total,
        tool_usage_cost: cost.built_in_tools,
        cache_read_cost: token_breakdown
            .map(|breakdown| breakdown.cache_read_cost)
            .filter(|value| *value > 0.0),
        cache_creation_cost: token_breakdown
            .map(|breakdown| breakdown.cache_creation_cost)
            .filter(|value| *value > 0.0),
        reasoning_cost: token_breakdown
            .map(|breakdown| breakdown.reasoning_cost)
            .filter(|value| *value > 0.0),
        additional_costs: additional_costs.filter(|costs| !costs.is_empty()),
        original_cost: cost.original,
        discount_percent: cost.discount_percent,
        discount_amount: cost.discount_amount,
        margin_percent: cost.margin_percent,
        margin_fixed_amount: cost.margin_fixed_amount,
        margin_total_amount: cost.margin_total_amount,
        billed_token_rates: token_breakdown.and_then(|breakdown| breakdown.rates),
    }
}
