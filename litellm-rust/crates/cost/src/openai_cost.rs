use crate::error::CostError;
use serde_json::Value;

use crate::call_type::CallTypes;
use crate::non_token::{Charge, Unit, calculate};
use crate::wire::py_float;

pub use crate::non_token::video_resolution_to_cost_field_suffix;

pub fn cost_router(call_type: &str) -> &'static str {
    if matches!(
        call_type.parse::<CallTypes>(),
        Ok(CallTypes::transcription | CallTypes::atranscription)
    ) {
        "cost_per_second"
    } else {
        "cost_per_token"
    }
}

pub fn video_output_cost_per_second(
    model_info: &Value,
    video_resolution: Option<&str>,
) -> Option<f64> {
    let rate = |value: &Value| py_float(value);
    let tier_rate = video_resolution
        .and_then(video_resolution_to_cost_field_suffix)
        .and_then(|suffix| model_info.get(format!("output_cost_per_second_{suffix}")))
        .and_then(rate);
    tier_rate.or_else(|| model_info.get("output_cost_per_second").and_then(rate))
}

pub fn video_generation_cost(
    model_info: &Value,
    duration_seconds: f64,
    video_resolution: Option<&str>,
) -> Result<f64, CostError> {
    let rate = model_info
        .get("output_cost_per_video_per_second")
        .and_then(Value::as_f64)
        .or_else(|| video_output_cost_per_second(model_info, video_resolution));
    let Some(rate) = rate else {
        if !duration_seconds.is_finite() || duration_seconds < 0.0 {
            return Err(CostError::InvalidQuantity);
        }
        return Ok(0.0);
    };
    calculate(&[Charge {
        unit: Unit::Second,
        quantity: duration_seconds,
        rate,
        units_per_rate: 1.0,
    }])
    .map(|cost| cost.total)
}
