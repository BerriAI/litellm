use jiff::Timestamp;
use serde_json::Value;

use crate::base_rate_selection::{
    TokenBaseRates, get_tiered_reasoning_rate, get_token_base_cost,
    get_token_base_cost_without_off_peak, uses_inclusive_token_thresholds,
};
use crate::generic_input::{InputBaseRates, calculate_input_cost};
use crate::generic_output::{calculate_output_cost, resolve_reasoning_token_cost};
use crate::generic_usage::{ParsedPromptDetails, parse_prompt_tokens_details};
use crate::off_peak::{open_off_peak_block, parse_off_peak_rate};
use crate::provider_cache::apply_provider_cache_read_default;
use crate::regional_uplift::apply_regional_totals_uplift;
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

fn calculate_with_base_rates(
    usage: &ChatUsage,
    model_info: &Value,
    base: TokenBaseRates,
    service_tier: Option<&str>,
    reasoning: f64,
    multiplier: f64,
) -> (f64, f64) {
    calculate_generic_cost_with_resolved_rates(
        usage,
        model_info,
        ResolvedTokenRates {
            input: InputBaseRates {
                prompt: base.input,
                cache_read: base.cache_read,
                cache_creation: base.cache_creation,
                cache_creation_above_1hr: base.cache_creation_above_1hr,
            },
            output: base.output,
            reasoning: Some(reasoning),
            multiplier,
        },
        service_tier,
    )
}

pub fn calculate_generic_cost_from_model_info_without_off_peak(
    usage: &ChatUsage,
    model_info: &Value,
    service_tier: Option<&str>,
    threshold_inclusive: bool,
    multiplier: f64,
) -> (f64, f64) {
    let base = get_token_base_cost_without_off_peak(
        model_info,
        usage.prompt_tokens,
        service_tier,
        threshold_inclusive,
    );
    let reasoning = get_tiered_reasoning_rate(model_info, usage.prompt_tokens)
        .unwrap_or_else(|| resolve_reasoning_token_cost(model_info, service_tier, base.output));
    calculate_with_base_rates(usage, model_info, base, service_tier, reasoning, multiplier)
}

pub fn calculate_generic_cost_from_model_info(
    usage: &ChatUsage,
    model_info: &Value,
    service_tier: Option<&str>,
    threshold_inclusive: bool,
    multiplier: f64,
    at: Timestamp,
) -> (f64, f64) {
    let base = get_token_base_cost(
        model_info,
        usage.prompt_tokens,
        service_tier,
        threshold_inclusive,
        at,
    );
    let reasoning = resolve_billed_reasoning_rate(usage, model_info, service_tier, base.output, at);
    calculate_with_base_rates(usage, model_info, base, service_tier, reasoning, multiplier)
}

pub fn resolve_billed_reasoning_rate(
    usage: &ChatUsage,
    model_info: &Value,
    service_tier: Option<&str>,
    completion_base_cost: f64,
    at: Timestamp,
) -> f64 {
    open_off_peak_block(model_info, at)
        .and_then(|block| parse_off_peak_rate(block.get("output_cost_per_reasoning_token")))
        .or_else(|| get_tiered_reasoning_rate(model_info, usage.prompt_tokens))
        .unwrap_or_else(|| {
            resolve_reasoning_token_cost(model_info, service_tier, completion_base_cost)
        })
}

pub fn calculate_generic_cost_from_model_info_with_region(
    usage: &ChatUsage,
    model_info: &Value,
    service_tier: Option<&str>,
    threshold_inclusive: bool,
    data_residency: Option<&str>,
    vertex_location: Option<&str>,
    at: Timestamp,
) -> (f64, f64) {
    apply_regional_totals_uplift(
        calculate_generic_cost_from_model_info(
            usage,
            model_info,
            service_tier,
            threshold_inclusive,
            1.0,
            at,
        ),
        model_info,
        data_residency,
        vertex_location,
    )
}

#[derive(Clone, Copy, Debug)]
pub struct GenericCostRequest<'a> {
    pub model_info: &'a Value,
    pub usage: &'a ChatUsage,
    pub provider: Option<&'a str>,
    pub service_tier: Option<&'a str>,
    pub data_residency: Option<&'a str>,
    pub vertex_location: Option<&'a str>,
    pub at: Timestamp,
}

pub fn generic_cost_per_token(request: GenericCostRequest<'_>) -> (f64, f64) {
    let model_info = apply_provider_cache_read_default(request.model_info, request.provider);
    calculate_generic_cost_from_model_info_with_region(
        request.usage,
        &model_info,
        request.service_tier,
        uses_inclusive_token_thresholds(request.provider),
        request.data_residency,
        request.vertex_location,
        request.at,
    )
}
