use crate::error::CostError;
use serde_json::Value;

use crate::wire::{py_float, py_real};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompletionCost {
    pub input: f64,
    pub output: f64,
    pub built_in_tools: f64,
    pub additional: f64,
    pub original: f64,
    pub discounted: f64,
    pub total: f64,
    pub discount_percent: f64,
    pub discount_amount: f64,
    pub margin_percent: f64,
    pub margin_fixed_amount: f64,
    pub margin_total_amount: f64,
}

pub fn apply_cost_discount(
    base_cost: f64,
    provider: Option<&str>,
    config: &Value,
) -> (f64, f64, f64) {
    let percent = provider
        .filter(|provider| !provider.is_empty())
        .and_then(|provider| config.get(provider))
        .and_then(py_real)
        .unwrap_or(0.0);
    let amount = base_cost * percent;
    (base_cost - amount, percent, amount)
}

pub fn apply_cost_margin(
    base_cost: f64,
    provider: Option<&str>,
    config: &Value,
) -> (f64, f64, f64, f64) {
    let selected = provider
        .filter(|provider| !provider.is_empty())
        .and_then(|provider| config.get(provider))
        .or_else(|| config.get("global"));
    let (percent, fixed) = match selected {
        Some(Value::Object(config)) => (
            config.get("percentage").and_then(py_float).unwrap_or(0.0),
            config.get("fixed_amount").and_then(py_float).unwrap_or(0.0),
        ),
        Some(value) => (py_real(value).unwrap_or(0.0), 0.0),
        None => (0.0, 0.0),
    };
    let total = base_cost * percent + fixed;
    (base_cost + total, percent, fixed, total)
}

pub fn completion_cost(
    prompt: f64,
    output: f64,
    built_in_tools: f64,
    additional_costs: &[f64],
    provider: Option<&str>,
    discount_config: &Value,
    margin_config: &Value,
) -> CompletionCost {
    let additional = additional_costs.iter().sum::<f64>();
    let original = prompt + output + built_in_tools + additional;
    let (discounted, discount_percent, discount_amount) =
        apply_cost_discount(original, provider, discount_config);
    let (total, margin_percent, margin_fixed_amount, margin_total_amount) =
        apply_cost_margin(discounted, provider, margin_config);
    CompletionCost {
        input: prompt,
        output,
        built_in_tools,
        additional,
        original,
        discounted,
        total,
        discount_percent,
        discount_amount,
        margin_percent,
        margin_fixed_amount,
        margin_total_amount,
    }
}

pub fn get_response_cost_from_hidden_params(
    hidden_params: &Value,
) -> Result<Option<f64>, CostError> {
    let Some(value) = hidden_params
        .get("additional_headers")
        .and_then(|headers| headers.get("llm_provider-x-litellm-response-cost"))
    else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    py_float(value)
        .map(Some)
        .ok_or(CostError::InvalidProviderCost)
}

pub fn response_cost_calculator(
    cache_hit: bool,
    hidden_params: &Value,
    calculated_completion_cost: f64,
) -> Result<f64, CostError> {
    if cache_hit {
        return Ok(0.0);
    }
    Ok(get_response_cost_from_hidden_params(hidden_params)?.unwrap_or(calculated_completion_cost))
}
