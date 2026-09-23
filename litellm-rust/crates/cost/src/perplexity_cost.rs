use jiff::Timestamp;
use serde_json::Value;

use crate::off_peak::{open_off_peak_block, parse_off_peak_rate};
use crate::responses_usage::ChatUsage;

fn rate(value: Option<&Value>) -> f64 {
    match value {
        Some(Value::Number(value)) => value.as_f64().unwrap_or(0.0),
        Some(Value::String(value)) => value.parse().unwrap_or(0.0),
        Some(Value::Bool(value)) => f64::from(*value as u8),
        _ => 0.0,
    }
}

fn cost_per_query(model_info: &Value) -> f64 {
    let value = model_info
        .get("search_queries_cost_per_query")
        .filter(|value| match value {
            Value::Null => false,
            Value::Bool(value) => *value,
            Value::Number(_) => rate(Some(value)) != 0.0,
            Value::String(value) => !value.is_empty(),
            Value::Array(values) => !values.is_empty(),
            Value::Object(values) => !values.is_empty(),
        })
        .or_else(|| model_info.get("search_context_cost_per_query"));
    match value {
        Some(Value::Object(prices)) => rate(prices.get("search_context_size_low")),
        _ => rate(value),
    }
}

pub fn cost_per_token(usage: &ChatUsage, model_info: &Value, at: Timestamp) -> (f64, f64) {
    if let Some(cost) = usage.cost {
        return (0.0, cost);
    }
    let off_peak = open_off_peak_block(model_info, at);
    let selected_rate = |key| {
        off_peak
            .and_then(|block| parse_off_peak_rate(block.get(key)))
            .unwrap_or_else(|| rate(model_info.get(key)))
    };
    let input_rate = selected_rate("input_cost_per_token");
    let output_rate = selected_rate("output_cost_per_token");
    let citations = usage
        .extra
        .get("citation_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let prompt_cost = usage.prompt_tokens as f64 * input_rate
        + citations as f64 * rate(model_info.get("citation_cost_per_token"));
    let reasoning = usage
        .extra
        .get("reasoning_tokens")
        .and_then(Value::as_u64)
        .filter(|tokens| *tokens > 0)
        .or_else(|| {
            usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens)
        })
        .unwrap_or(0);
    let reasoning_rate = model_info
        .get("output_cost_per_reasoning_token")
        .filter(|value| !value.is_null());
    let completion_cost = if reasoning > 0 && reasoning_rate.is_some() {
        usage.completion_tokens.saturating_sub(reasoning) as f64 * output_rate
            + reasoning as f64 * rate(reasoning_rate)
    } else {
        usage.completion_tokens as f64 * output_rate
    };
    let queries = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.web_search_requests)
        .unwrap_or(0);
    (
        prompt_cost,
        completion_cost + queries as f64 * cost_per_query(model_info),
    )
}
