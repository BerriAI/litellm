use serde_json::Value;

use crate::generic_usage::ParsedPromptDetails;
use crate::responses_usage::CacheCreationTokenDetails;

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse().ok(),
        Value::Bool(value) => Some(u8::from(*value) as f64),
        _ => None,
    }
}

pub fn get_cost_per_unit(
    model_info: &Value,
    cost_key: &str,
    default_value: Option<f64>,
) -> Option<f64> {
    let direct = model_info.get(cost_key);
    if let Some(parsed) = direct.and_then(number) {
        return Some(parsed);
    }
    if direct.is_none_or(Value::is_null) {
        for suffix in ["_ultrafast", "_priority", "_standard", "_flex", "_fast"] {
            if cost_key.contains(suffix) {
                let base = cost_key.replace(suffix, "");
                return model_info.get(&base).and_then(number).or(default_value);
            }
        }
    }
    default_value
}

pub fn calculate_cost_component(model_info: &Value, cost_key: &str, usage_value: f64) -> f64 {
    if usage_value.is_nan() || usage_value <= 0.0 {
        return 0.0;
    }
    get_cost_per_unit(model_info, cost_key, Some(0.0)).unwrap_or(0.0) * usage_value
}

pub fn calculate_cache_writing_cost(
    cache_creation_tokens: u64,
    cache_creation_token_details: Option<&CacheCreationTokenDetails>,
    cache_creation_cost_above_1hr: f64,
    cache_creation_cost: f64,
) -> f64 {
    match cache_creation_token_details {
        Some(details) => {
            details.ephemeral_5m_input_tokens.unwrap_or(0) as f64 * cache_creation_cost
                + details.ephemeral_1h_input_tokens.unwrap_or(0) as f64
                    * cache_creation_cost_above_1hr
        }
        None => cache_creation_tokens as f64 * cache_creation_cost,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputBaseRates {
    pub prompt: f64,
    pub cache_read: f64,
    pub cache_creation: f64,
    pub cache_creation_above_1hr: f64,
}

fn tier_key(base: &str, service_tier: Option<&str>) -> String {
    match service_tier.map(str::to_ascii_lowercase).as_deref() {
        Some("flex") => format!("{base}_flex"),
        Some("priority" | "fast") => format!("{base}_priority"),
        Some("ultrafast") => format!("{base}_ultrafast"),
        _ => base.to_owned(),
    }
}

pub fn calculate_input_cost(
    details: &ParsedPromptDetails,
    model_info: &Value,
    rates: InputBaseRates,
    service_tier: Option<&str>,
) -> f64 {
    let cached_audio_rate = get_cost_per_unit(
        model_info,
        &tier_key("cache_read_input_audio_token_cost", service_tier),
        None,
    )
    .unwrap_or(rates.cache_read);
    let audio_rate_key = tier_key("input_cost_per_audio_token", service_tier);
    let audio_cost = if details.audio_tokens > 0
        && !(details.audio_length_seconds > 0.0
            && model_info
                .get("input_cost_per_audio_per_second")
                .is_some_and(|value| !value.is_null()))
    {
        calculate_cost_component(model_info, &audio_rate_key, details.audio_tokens as f64)
    } else {
        0.0
    };
    let image_cost = if details.image_tokens > 0
        && !(details.image_count > 0
            && model_info
                .get("input_cost_per_image")
                .is_some_and(|value| !value.is_null()))
    {
        let rate_key = if model_info
            .get("input_cost_per_image_token")
            .is_some_and(|value| !value.is_null())
        {
            "input_cost_per_image_token"
        } else {
            "input_cost_per_token"
        };
        calculate_cost_component(model_info, rate_key, details.image_tokens as f64)
    } else {
        0.0
    };
    let video_cost = if details.video_tokens > 0
        && !(details.video_length_seconds > 0.0
            && model_info
                .get("input_cost_per_video_per_second")
                .is_some_and(|value| !value.is_null()))
    {
        let rate_key = if model_info
            .get("input_cost_per_video_token")
            .is_some_and(|value| !value.is_null())
        {
            "input_cost_per_video_token"
        } else {
            "input_cost_per_token"
        };
        calculate_cost_component(model_info, rate_key, details.video_tokens as f64)
    } else {
        0.0
    };
    details.text_tokens as f64 * rates.prompt
        + details
            .cache_hit_tokens
            .saturating_sub(details.cache_hit_audio_tokens) as f64
            * rates.cache_read
        + details.cache_hit_audio_tokens as f64 * cached_audio_rate
        + audio_cost
        + image_cost
        + video_cost
        + calculate_cache_writing_cost(
            details.cache_creation_tokens,
            details.cache_creation_token_details.as_ref(),
            rates.cache_creation_above_1hr,
            rates.cache_creation,
        )
        + calculate_cost_component(
            model_info,
            "input_cost_per_character",
            details.character_count as f64,
        )
        + calculate_cost_component(
            model_info,
            "input_cost_per_image",
            details.image_count as f64,
        )
        + calculate_cost_component(
            model_info,
            "input_cost_per_video_per_second",
            details.video_length_seconds,
        )
        + calculate_cost_component(
            model_info,
            "input_cost_per_audio_per_second",
            details.audio_length_seconds,
        )
        + calculate_cost_component(
            model_info,
            "input_cost_per_query",
            details.query_count as f64,
        )
}
