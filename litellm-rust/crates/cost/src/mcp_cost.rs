use serde_json::Value;

pub fn calculate_mcp_tool_call_cost(details: Option<&Value>) -> f64 {
    let Some(details) = details else {
        return 0.0;
    };
    if let Some(cost) = details.get("response_cost").and_then(Value::as_f64) {
        return cost;
    }
    let metadata = details.get("mcp_tool_call_metadata");
    let prices = metadata.and_then(|metadata| metadata.get("mcp_server_cost_info"));
    metadata
        .and_then(|metadata| metadata.get("name"))
        .and_then(Value::as_str)
        .and_then(|name| {
            prices
                .and_then(|prices| prices.get("tool_name_to_cost_per_query"))
                .and_then(|tools| tools.get(name))
        })
        .or_else(|| prices.and_then(|prices| prices.get("default_cost_per_query")))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}
