use serde_json::{Number, Value};

use crate::provider::LlmProviders;
use crate::wire::py_float_unless_bool;

fn rate(value: Option<&Value>) -> Option<f64> {
    value.and_then(py_float_unless_bool)
}

fn number(value: f64) -> Option<Value> {
    Number::from_f64(value).map(Value::Number)
}

pub fn with_default_cache_read_rate(model_info: &Value) -> Value {
    let Some(model) = model_info.as_object() else {
        return model_info.clone();
    };
    if model
        .get("cache_read_input_token_cost")
        .is_some_and(|value| !value.is_null())
    {
        return model_info.clone();
    }
    let Some(base_read_rate) =
        rate(model.get("input_cost_per_token")).and_then(|input| number(input * 0.5))
    else {
        return model_info.clone();
    };
    let off_peak = model.get("off_peak_pricing").and_then(Value::as_object);
    let adjusted_off_peak = off_peak
        .filter(|off_peak| !off_peak.contains_key("cache_read_input_token_cost"))
        .map(|off_peak| {
            let off_peak_read_rate = rate(off_peak.get("input_cost_per_token"))
                .and_then(|input| number(input * 0.5))
                .unwrap_or_else(|| base_read_rate.clone());
            Value::Object(
                off_peak
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .chain([("cache_read_input_token_cost".to_owned(), off_peak_read_rate)])
                    .collect(),
            )
        });
    Value::Object(
        model
            .iter()
            .filter(|(key, _)| {
                key.as_str() != "cache_read_input_token_cost" && key.as_str() != "off_peak_pricing"
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .chain([("cache_read_input_token_cost".to_owned(), base_read_rate)])
            .chain(model.get("off_peak_pricing").map(|off_peak| {
                (
                    "off_peak_pricing".to_owned(),
                    adjusted_off_peak.unwrap_or_else(|| off_peak.clone()),
                )
            }))
            .collect(),
    )
}

pub fn apply_provider_cache_read_default(model_info: &Value, provider: Option<&str>) -> Value {
    if LlmProviders::FIREWORKS_AI.matches(provider) {
        with_default_cache_read_rate(model_info)
    } else {
        model_info.clone()
    }
}
