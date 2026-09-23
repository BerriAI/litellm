use serde_json::Value;

fn range(tier: &Value) -> Option<(i64, i64)> {
    let bounds = tier.get("range")?.as_array()?;
    match bounds.as_slice() {
        [start, end] => Some((start.as_i64()?, end.as_i64()?)),
        _ => None,
    }
}

pub fn select_tier_for_input(tiers: &[Value], input_tokens: i64) -> Option<&Value> {
    if input_tokens <= 0 {
        return None;
    }
    let valid = tiers
        .iter()
        .filter_map(|tier| range(tier).map(|bounds| (tier, bounds)));
    valid
        .clone()
        .filter(|(_, (start, end))| *start < input_tokens && input_tokens <= *end)
        .min_by_key(|(_, (start, _))| *start)
        .or_else(|| valid.max_by_key(|(_, (start, _))| *start))
        .map(|(tier, _)| tier)
}

fn rate(value: Option<&Value>) -> f64 {
    match value {
        Some(Value::Bool(value)) => u8::from(*value) as f64,
        Some(Value::Number(value)) => value.as_f64().unwrap_or(0.0),
        Some(Value::String(value)) => value.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

pub fn tier_rate(tier: &Value, cost_key: &str, fallback_cost_key: Option<&str>) -> f64 {
    let primary = tier.get(cost_key).filter(|value| !value.is_null());
    rate(primary.or_else(|| fallback_cost_key.and_then(|key| tier.get(key))))
}
