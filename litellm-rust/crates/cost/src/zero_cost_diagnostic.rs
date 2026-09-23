use serde_json::Value;

use crate::responses_usage::ChatUsage;

pub const ZERO_COST_COUNTER_NAME: &str = "litellm_zero_cost_requests_total";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZeroCostReason {
    MissingPricingKey,
    PricingNotApplied,
    CostCalculationError,
}

impl ZeroCostReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MissingPricingKey => "missing_pricing_key",
            Self::PricingNotApplied => "pricing_not_applied",
            Self::CostCalculationError => "cost_calculation_error",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ZeroCostDiagnostic {
    pub reason: ZeroCostReason,
    pub pricing_model: String,
    pub missing_pricing_keys: Vec<&'static str>,
}

pub fn used_pricing_keys(usage: &ChatUsage) -> Vec<&'static str> {
    let prompt_audio = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.audio_tokens)
        .unwrap_or(0);
    let completion_audio = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.audio_tokens)
        .unwrap_or(0);
    [
        (
            "input_cost_per_token",
            usage.prompt_tokens.saturating_sub(prompt_audio),
        ),
        ("input_cost_per_audio_token", prompt_audio),
        (
            "output_cost_per_token",
            usage.completion_tokens.saturating_sub(completion_audio),
        ),
        ("output_cost_per_audio_token", completion_audio),
    ]
    .into_iter()
    .filter_map(|(key, count)| (count > 0).then_some(key))
    .collect()
}

fn declares_rate(value: &Value, depth: u8) -> bool {
    if depth == 0 {
        return value.as_f64().is_some_and(|rate| rate > 0.0);
    }
    match value {
        Value::Object(fields) => fields
            .iter()
            .filter(|(key, _)| key.contains("cost") || key.contains("pricing"))
            .any(|(_, value)| declares_rate(value, depth - 1)),
        Value::Array(values) => values.iter().any(|value| declares_rate(value, depth - 1)),
        _ => value.as_f64().is_some_and(|rate| rate > 0.0),
    }
}

pub fn diagnose_zero_cost(
    usage: &ChatUsage,
    pricing_model: &str,
    pricing_entry: &Value,
    calculation_failed: bool,
) -> Option<ZeroCostDiagnostic> {
    let used_keys = used_pricing_keys(usage);
    if used_keys.is_empty() {
        return None;
    }
    let missing_pricing_keys: Vec<_> = used_keys
        .iter()
        .copied()
        .filter(|key| pricing_entry.get(key).is_none_or(Value::is_null))
        .collect();
    if missing_pricing_keys.is_empty()
        && used_keys.iter().all(|key| {
            pricing_entry
                .get(key)
                .and_then(Value::as_f64)
                .is_some_and(|rate| rate == 0.0)
        })
    {
        return None;
    }
    if !declares_rate(pricing_entry, 4) {
        return None;
    }
    let reason = if calculation_failed {
        ZeroCostReason::CostCalculationError
    } else if missing_pricing_keys.is_empty() {
        ZeroCostReason::PricingNotApplied
    } else {
        ZeroCostReason::MissingPricingKey
    };
    Some(ZeroCostDiagnostic {
        reason,
        pricing_model: pricing_model.to_owned(),
        missing_pricing_keys: if calculation_failed {
            Vec::new()
        } else {
            missing_pricing_keys
        },
    })
}

pub fn zero_cost_warning(
    diagnostic: &ZeroCostDiagnostic,
    model_group: Option<&str>,
    model: &str,
    provider: Option<&str>,
    usage: &ChatUsage,
) -> String {
    let cause = match diagnostic.reason {
        ZeroCostReason::MissingPricingKey => format!(
            "pricing entry '{}' has no {}. Set the missing rate in the deployment's model_info or in the model cost map, or set every rate to 0 to mark the model free",
            diagnostic.pricing_model,
            diagnostic.missing_pricing_keys.join(", ")
        ),
        ZeroCostReason::PricingNotApplied => format!(
            "pricing entry '{}' declares non-zero rates for this usage, but the cost calculator returned $0",
            diagnostic.pricing_model
        ),
        ZeroCostReason::CostCalculationError => format!(
            "cost calculation raised for pricing entry '{}', see response_cost_failure_debug_information",
            diagnostic.pricing_model
        ),
    };
    format!(
        "Billable request priced at $0 and logged as such (model_group={} model={} provider={} prompt_tokens={} completion_tokens={}): {}. Counted in {}{{reason=\"{}\"}}",
        model_group.unwrap_or(model),
        model,
        provider.unwrap_or("unknown"),
        usage.prompt_tokens,
        usage.completion_tokens,
        cause,
        ZERO_COST_COUNTER_NAME,
        diagnostic.reason.as_str()
    )
}
