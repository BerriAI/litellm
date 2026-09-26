use crate::wire::py_float;
use serde_json::Value;

fn range(tier: &Value) -> Option<(f64, f64)> {
    let bounds = tier.get("range")?.as_array()?;
    match bounds.as_slice() {
        [start, end] => Some((start.as_f64()?, end.as_f64()?)),
        _ => None,
    }
}

pub fn select_tier_for_input(tiers: &[Value], input_tokens: i64) -> Option<&Value> {
    if input_tokens <= 0 {
        return None;
    }
    let tokens = input_tokens as f64;
    let valid = tiers
        .iter()
        .filter_map(|tier| range(tier).map(|bounds| (tier, bounds)));
    valid
        .clone()
        .filter(|(_, (start, end))| *start < tokens && tokens <= *end)
        .min_by(|(_, (start, _)), (_, (other, _))| {
            start.partial_cmp(other).expect("tier starts are finite")
        })
        .or_else(|| {
            valid.max_by(|(_, (start, _)), (_, (other, _))| {
                start.partial_cmp(other).expect("tier starts are finite")
            })
        })
        .map(|(tier, _)| tier)
}

fn coerce(value: &Value) -> f64 {
    py_float(value).unwrap_or(0.0)
}

pub fn tier_rate(tier: &Value, cost_key: &str, fallback_cost_key: Option<&str>) -> f64 {
    match tier.get(cost_key).filter(|value| !value.is_null()) {
        Some(primary) => coerce(primary),
        None => fallback_cost_key
            .and_then(|key| tier.get(key).filter(|value| !value.is_null()))
            .map_or(0.0, coerce),
    }
}
