use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::wire::{lax_bool, lax_float};

pub type CostByUnit = BTreeMap<String, Option<f64>>;

fn in_spend(entry: &Map<String, Value>) -> Option<bool> {
    match entry.get("guardrail_cost_in_spend") {
        None | Some(Value::Null) => Some(true),
        Some(value) => lax_bool(value),
    }
}

pub fn billed_guardrail_cost_by_unit(raw: &Value) -> Option<CostByUnit> {
    let entry = raw.as_object()?;
    if !in_spend(entry)? {
        return None;
    }
    let values = entry.get("guardrail_cost_by_unit")?.as_object()?;
    values
        .iter()
        .map(|(key, value)| {
            let price = match value {
                Value::Null => Some(None),
                value => lax_float(value)
                    .filter(|price| price.is_finite() && *price >= 0.0)
                    .map(Some),
            }?;
            Some((key.clone(), price))
        })
        .collect()
}

fn pricing_entry(raw: &Value) -> Option<BTreeMap<String, f64>> {
    raw.as_object()?
        .get("guardrail_cost_per_unit")?
        .as_object()?
        .iter()
        .map(|(key, value)| Some((key.clone(), lax_float(value)?)))
        .collect()
}

pub fn bedrock_guardrail_cost_by_unit(
    usage_units: &BTreeMap<String, i64>,
    aws_region_name: Option<&str>,
    model_cost: &BTreeMap<String, Value>,
) -> Option<CostByUnit> {
    let regional = aws_region_name
        .and_then(|region| model_cost.get(&format!("bedrock/{region}/guardrails")))
        .and_then(pricing_entry);
    let pricing =
        regional.or_else(|| model_cost.get("bedrock/guardrails").and_then(pricing_entry))?;
    Some(
        usage_units
            .iter()
            .map(|(counter, units)| {
                (
                    counter.clone(),
                    pricing.get(counter).map(|rate| *units as f64 * rate),
                )
            })
            .collect(),
    )
}

pub fn guardrail_cost_total(cost_by_unit: Option<&CostByUnit>) -> f64 {
    cost_by_unit
        .into_iter()
        .flat_map(BTreeMap::values)
        .flatten()
        .sum()
}

pub fn bedrock_guardrail_cost(
    usage_units: &BTreeMap<String, i64>,
    aws_region_name: Option<&str>,
    model_cost: &BTreeMap<String, Value>,
) -> f64 {
    guardrail_cost_total(
        bedrock_guardrail_cost_by_unit(usage_units, aws_region_name, model_cost).as_ref(),
    )
}

pub fn azure_prompt_shield_guardrail_cost(
    usage_units: &BTreeMap<String, i64>,
    cost_tier: Option<&str>,
    price_per_1000_text_records: Option<f64>,
) -> Option<f64> {
    if cost_tier == Some("free") {
        return Some(0.0);
    }
    price_per_1000_text_records
        .map(|price| *usage_units.get("text_records").unwrap_or(&0) as f64 * price / 1000.0)
}

fn entry_cost(raw: &Value) -> f64 {
    let Some(entry) = raw.as_object() else {
        return 0.0;
    };
    let Some(in_spend) = in_spend(entry) else {
        return 0.0;
    };
    let cost = match entry.get("guardrail_cost") {
        None | Some(Value::Null) => None,
        Some(value) => match lax_float(value) {
            Some(cost) => Some(cost),
            None => return 0.0,
        },
    };
    cost.filter(|cost| in_spend && cost.is_finite() && *cost > 0.0)
        .unwrap_or(0.0)
}

pub fn guardrail_information_cost(raw: &Value) -> f64 {
    match raw {
        Value::Array(entries) => entries.iter().map(entry_cost).sum(),
        entry => entry_cost(entry),
    }
}

pub fn cost_breakdown_with_guardrail(
    cost_breakdown: Option<&BTreeMap<String, f64>>,
    guardrail_cost: f64,
) -> Option<BTreeMap<String, f64>> {
    if guardrail_cost <= 0.0 {
        return cost_breakdown.cloned();
    }
    let existing = cost_breakdown.cloned().unwrap_or_default();
    Some(BTreeMap::from_iter(
        existing
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "guardrail_cost" | "total_cost"))
            .map(|(key, value)| (key.clone(), *value))
            .chain([
                ("guardrail_cost".to_owned(), guardrail_cost),
                (
                    "total_cost".to_owned(),
                    existing.get("total_cost").copied().unwrap_or(0.0) + guardrail_cost,
                ),
            ]),
    ))
}
