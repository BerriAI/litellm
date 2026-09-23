use serde_json::Value;

pub fn is_azure_model_router(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.contains("model-router") || lower.contains("model_router")
}

pub fn is_router_fee_entry(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let bare = lower.strip_prefix("azure_ai/").unwrap_or(&lower);
    matches!(bare, "model-router" | "model_router")
}

pub fn router_fee_entry_name(model: &str) -> &'static str {
    let lower = model.to_ascii_lowercase();
    let bare = lower.strip_prefix("azure_ai/").unwrap_or(&lower);
    if bare == "model-router" {
        "model-router"
    } else {
        "model_router"
    }
}

pub fn router_fee_name<'a>(model: &'a str, request_model: Option<&'a str>) -> Option<&'a str> {
    if is_router_fee_entry(model) {
        return None;
    }
    if is_azure_model_router(model) {
        return Some(model);
    }
    request_model.filter(|name| is_azure_model_router(name))
}

pub fn calculate_azure_model_router_flat_cost(
    model: &str,
    prompt_tokens: u64,
    router_model_info: &Value,
) -> f64 {
    if !is_azure_model_router(model) {
        return 0.0;
    }
    router_model_info
        .get("input_cost_per_token")
        .and_then(Value::as_f64)
        .filter(|rate| *rate > 0.0)
        .map_or(0.0, |rate| prompt_tokens as f64 * rate)
}
