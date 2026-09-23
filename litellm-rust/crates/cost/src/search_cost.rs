use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParallelAiPricing {
    pub request_cost: f64,
    pub default_results: u64,
    pub additional_result_cost: f64,
}

fn non_negative_int(value: Option<&Value>) -> Option<u64> {
    value?.as_u64()
}

pub fn usage_count(usage: &[Value], sku: &str) -> Option<u64> {
    let counts: Vec<u64> = usage
        .iter()
        .filter(|item| item.get("name").and_then(Value::as_str) == Some(sku))
        .filter_map(|item| non_negative_int(item.get("count")))
        .collect();
    (!counts.is_empty()).then(|| counts.into_iter().fold(0u64, u64::saturating_add))
}

pub fn effective_mode(optional_params: &Value) -> &str {
    optional_params
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            if optional_params.get("processor").and_then(Value::as_str) == Some("pro") {
                "advanced"
            } else {
                "basic"
            }
        })
}

pub fn effective_max_results(optional_params: &Value, default_results: u64) -> u64 {
    optional_params
        .get("advanced_settings")
        .and_then(|settings| non_negative_int(settings.get("max_results")))
        .or_else(|| non_negative_int(optional_params.get("max_results")))
        .unwrap_or(default_results)
}

pub fn parallel_ai_search_cost(
    optional_params: &Value,
    usage: Option<&[Value]>,
    pricing: ParallelAiPricing,
) -> f64 {
    let request_count = usage
        .and_then(|usage| usage_count(usage, "sku_search"))
        .unwrap_or(1);
    let additional_results = usage.map_or_else(
        || {
            effective_max_results(optional_params, pricing.default_results)
                .saturating_sub(pricing.default_results)
        },
        |usage| usage_count(usage, "sku_search_additional_results").unwrap_or(0),
    );
    request_count as f64 * pricing.request_cost
        + additional_results as f64 * pricing.additional_result_cost
}

fn rate(value: Option<&Value>) -> f64 {
    match value {
        Some(Value::Number(value)) => value.as_f64().unwrap_or(0.0),
        Some(Value::String(value)) => value.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

pub fn search_provider_cost_per_query(
    model_info: &Value,
    number_of_queries: u64,
    optional_params: &Value,
) -> (f64, f64) {
    let tiered = model_info
        .get("tiered_pricing")
        .and_then(Value::as_array)
        .filter(|tiers| !tiers.is_empty());
    let cost_per_query = tiered.map_or_else(
        || rate(model_info.get("input_cost_per_query")),
        |tiers| {
            let max_results = optional_params
                .get("max_results")
                .and_then(Value::as_i64)
                .unwrap_or(10);
            let tier = tiers
                .iter()
                .find(|tier| {
                    tier.get("max_results_range")
                        .and_then(Value::as_array)
                        .is_some_and(|bounds| {
                            matches!(bounds.as_slice(), [min, max] if min.as_i64().is_some_and(|min| min <= max_results) && max.as_i64().is_some_and(|max| max_results <= max))
                        })
                })
                .or_else(|| tiers.last());
            rate(tier.and_then(|tier| tier.get("input_cost_per_query")))
        },
    );
    (number_of_queries as f64 * cost_per_query, 0.0)
}

pub fn provider_usage(optional_params: &Value) -> Option<&[Value]> {
    optional_params
        .get("_parallel_ai_usage")
        .and_then(Value::as_array)
        .filter(|usage| usage.iter().all(Value::is_object))
        .map(Vec::as_slice)
}
