use serde_json::Value;

use crate::generic_input::{InputBaseRates, calculate_input_cost};
use crate::generic_output::calculate_output_cost;
use crate::generic_usage::{ParsedPromptDetails, parse_prompt_tokens_details};
use crate::responses_usage::ChatUsage;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedTokenRates {
    pub input: InputBaseRates,
    pub output: f64,
    pub reasoning: Option<f64>,
    pub multiplier: f64,
}

pub fn billable_prompt_details(usage: &ChatUsage) -> ParsedPromptDetails {
    let details = if usage.prompt_tokens_details.is_some() {
        parse_prompt_tokens_details(usage)
    } else {
        ParsedPromptDetails {
            text_tokens: usage.prompt_tokens,
            ..ParsedPromptDetails::default()
        }
    };
    let total_details = details.text_tokens as u128
        + details.cache_hit_tokens as u128
        + details.audio_tokens as u128
        + details.cache_creation_tokens as u128
        + details.image_tokens as u128
        + details.video_tokens as u128;
    let overlapping_cache = (details.cache_hit_tokens > 0 || details.cache_creation_tokens > 0)
        && total_details > usage.prompt_tokens as u128;
    if overlapping_cache {
        let budget = usage
            .prompt_tokens
            .saturating_sub(details.cache_hit_tokens)
            .saturating_sub(details.cache_creation_tokens);
        let audio = details.audio_tokens.min(budget);
        let image = details.image_tokens.min(budget - audio);
        let video = details.video_tokens.min(budget - audio - image);
        return ParsedPromptDetails {
            text_tokens: budget - audio - image - video,
            audio_tokens: audio,
            image_tokens: image,
            video_tokens: video,
            ..details
        };
    }
    if details.text_tokens == 0 && details.image_count == 0 {
        return ParsedPromptDetails {
            text_tokens: usage
                .prompt_tokens
                .saturating_sub(details.cache_hit_tokens)
                .saturating_sub(details.audio_tokens)
                .saturating_sub(details.cache_creation_tokens)
                .saturating_sub(details.image_tokens)
                .saturating_sub(details.video_tokens),
            ..details
        };
    }
    details
}

pub fn calculate_generic_cost_with_resolved_rates(
    usage: &ChatUsage,
    model_info: &Value,
    rates: ResolvedTokenRates,
    service_tier: Option<&str>,
) -> (f64, f64) {
    let prompt = calculate_input_cost(
        &billable_prompt_details(usage),
        model_info,
        rates.input,
        service_tier,
    );
    let completion = calculate_output_cost(
        usage,
        model_info,
        rates.output,
        service_tier,
        rates.reasoning,
    );
    (prompt * rates.multiplier, completion * rates.multiplier)
}
