use serde_json::Value;

pub const DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND: f64 = 0.0014;

pub fn get_replicate_completion_pricing(
    total_time_ms: f64,
    created_seconds: Option<f64>,
    ended_seconds: Option<f64>,
    now_seconds: f64,
    price_per_second: f64,
) -> f64 {
    let duration_ms = if total_time_ms == 0.0 {
        ended_seconds.unwrap_or(now_seconds) - created_seconds.unwrap_or(now_seconds)
    } else {
        total_time_ms
    };
    price_per_second * duration_ms / 1000.0
}

pub fn has_token_or_tiered_pricing(model_info: &Value) -> bool {
    model_info
        .get("input_cost_per_token")
        .and_then(Value::as_f64)
        .is_some_and(|rate| rate > 0.0)
        || model_info
            .get("output_cost_per_token")
            .and_then(Value::as_f64)
            .is_some_and(|rate| rate > 0.0)
        || model_info
            .get("tiered_pricing")
            .is_some_and(|tiers| !tiers.is_null())
}

pub fn bills_wall_clock_seconds(model_info: &Value) -> bool {
    match model_info.get("mode") {
        None | Some(Value::Null) => true,
        Some(Value::String(mode)) => matches!(
            mode.as_str(),
            "chat" | "completion" | "embedding" | "responses"
        ),
        Some(_) => false,
    }
}

pub fn per_second_pricing_cost(
    model_info: &Value,
    response_time_ms: Option<f64>,
) -> Option<(f64, f64)> {
    if has_token_or_tiered_pricing(model_info) || !bills_wall_clock_seconds(model_info) {
        return None;
    }
    let input = model_info
        .get("input_cost_per_second")
        .and_then(Value::as_f64);
    let output = model_info
        .get("output_cost_per_second")
        .and_then(Value::as_f64);
    if input.is_none() && output.is_none() {
        return None;
    }
    let seconds = response_time_ms.unwrap_or(0.0) / 1000.0;
    Some((
        input.unwrap_or(0.0) * seconds,
        output.unwrap_or(0.0) * seconds,
    ))
}
