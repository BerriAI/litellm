use serde_json::Value;

pub fn output_per_second_cost(
    model_info: &Value,
    response_time_ms: Option<f64>,
) -> Option<(f64, f64)> {
    let rate = model_info
        .get("output_cost_per_second")
        .and_then(Value::as_f64)?;
    Some((0.0, rate * response_time_ms? / 1000.0))
}
