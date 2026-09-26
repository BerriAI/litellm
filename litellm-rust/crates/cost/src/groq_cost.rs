use serde_json::Value;

use crate::responses_usage::ChatUsage;

pub fn cost_per_web_search_request(
    usage: &ChatUsage,
    model_info: &Value,
    browser_open_rate: f64,
) -> f64 {
    let server_tool_use = usage.extra.get("server_tool_use");
    let Some(server_tool_use) = server_tool_use else {
        return 0.0;
    };
    let searches = server_tool_use
        .get("web_search_requests")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let opens = server_tool_use
        .get("browser_open_requests")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let search_rate = model_info
        .get("search_context_cost_per_query")
        .and_then(|rates| rates.get("search_context_size_medium"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    searches as f64 * search_rate + opens as f64 * browser_open_rate
}
