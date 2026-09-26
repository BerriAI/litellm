use serde_json::Value;

use crate::base_rate_selection::tier_key;
use crate::generic_input::get_cost_per_unit;
use crate::generic_usage::parse_completion_tokens_details;
use crate::responses_usage::ChatUsage;

pub fn resolve_reasoning_token_cost(
    model_info: &Value,
    service_tier: Option<&str>,
    completion_base_cost: f64,
) -> f64 {
    let reasoning_key = tier_key("output_cost_per_reasoning_token", service_tier);
    if model_info
        .get(&reasoning_key)
        .is_some_and(|value| !value.is_null())
        && let Some(rate) = get_cost_per_unit(model_info, &reasoning_key, None)
    {
        return rate;
    }
    let output_key = tier_key("output_cost_per_token", service_tier);
    if output_key != "output_cost_per_token"
        && model_info
            .get(&output_key)
            .is_some_and(|value| !value.is_null())
    {
        return completion_base_cost;
    }
    get_cost_per_unit(model_info, "output_cost_per_reasoning_token", None)
        .unwrap_or(completion_base_cost)
}

pub fn calculate_output_cost(
    usage: &ChatUsage,
    model_info: &Value,
    completion_base_cost: f64,
    service_tier: Option<&str>,
    reasoning_rate_override: Option<f64>,
) -> f64 {
    let details = parse_completion_tokens_details(usage);
    let breakdown = details.audio_tokens > 0
        || details.reasoning_tokens > 0
        || details.image_tokens > 0
        || details.video_tokens > 0;
    let text_tokens = if details.text_tokens > 0 {
        details.text_tokens
    } else if breakdown {
        usage
            .completion_tokens
            .saturating_sub(details.reasoning_tokens)
            .saturating_sub(details.audio_tokens)
            .saturating_sub(details.image_tokens)
            .saturating_sub(details.video_tokens)
    } else {
        usage.completion_tokens
    };
    let modality_rate =
        |key| get_cost_per_unit(model_info, key, None).unwrap_or(completion_base_cost);
    let reasoning_rate = reasoning_rate_override.unwrap_or_else(|| {
        resolve_reasoning_token_cost(model_info, service_tier, completion_base_cost)
    });
    text_tokens as f64 * completion_base_cost
        + details.audio_tokens as f64 * modality_rate("output_cost_per_audio_token")
        + details.reasoning_tokens as f64 * reasoning_rate
        + details.image_tokens as f64 * modality_rate("output_cost_per_image_token")
        + details.video_tokens as f64 * modality_rate("output_cost_per_video_token")
}
