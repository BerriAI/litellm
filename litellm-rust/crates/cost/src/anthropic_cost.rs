use serde_json::Value;

use crate::responses_usage::ChatUsage;

pub fn fast_speed_multiplier(model_info: &Value, usage: &ChatUsage) -> f64 {
    if usage.extra.get("speed").and_then(Value::as_str) != Some("fast") {
        return 1.0;
    }
    model_info
        .get("provider_specific_entry")
        .and_then(|entry| entry.get("fast"))
        .and_then(Value::as_f64)
        .unwrap_or(1.0)
}

pub fn get_anthropic_web_search_requests_from_response(response: &Value) -> Option<i64> {
    response
        .pointer("/usage/server_tool_use/web_search_requests")
        .and_then(Value::as_i64)
}

pub fn get_web_search_requests_from_usage(usage: &ChatUsage) -> Option<i64> {
    usage
        .extra
        .get("server_tool_use")
        .and_then(|value| value.get("web_search_requests"))
        .and_then(Value::as_i64)
}

pub fn get_cost_for_anthropic_web_search(
    model_info: Option<&Value>,
    usage: Option<&ChatUsage>,
) -> f64 {
    let Some((model_info, requests)) =
        model_info.zip(usage.and_then(get_web_search_requests_from_usage))
    else {
        return 0.0;
    };
    let rate = model_info
        .get("search_context_cost_per_query")
        .and_then(|pricing| pricing.get("search_context_size_medium"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    rate * requests as f64
}
