use jiff::Timestamp;
use serde_json::Value;

use crate::base_rate_selection::{get_token_base_cost, tier_key, uses_inclusive_token_thresholds};
use crate::catalog::{ModelCostRequest, ModelInfoCatalog};
use crate::custom_pricing::CustomTokenRates;
use crate::generic_cost::resolve_billed_reasoning_rate;
use crate::generic_input::{calculate_cache_writing_cost, get_cost_per_unit};
use crate::generic_usage::{parse_completion_tokens_details, parse_prompt_tokens_details};
use crate::provider_cache::apply_provider_cache_read_default;
use crate::regional_uplift::combined_regional_multiplier;
use crate::responses_usage::ChatUsage;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilledTokenRates {
    pub input_cost_per_token: f64,
    pub output_cost_per_token: f64,
    pub cache_read_input_token_cost: f64,
    pub cache_read_input_audio_token_cost: f64,
    pub cache_creation_input_token_cost: f64,
    pub cache_creation_input_token_cost_above_1hr: f64,
    pub output_cost_per_reasoning_token: f64,
}

impl BilledTokenRates {
    pub fn scaled(self, multiplier: f64) -> Self {
        Self {
            input_cost_per_token: self.input_cost_per_token * multiplier,
            output_cost_per_token: self.output_cost_per_token * multiplier,
            cache_read_input_token_cost: self.cache_read_input_token_cost * multiplier,
            cache_read_input_audio_token_cost: self.cache_read_input_audio_token_cost * multiplier,
            cache_creation_input_token_cost: self.cache_creation_input_token_cost * multiplier,
            cache_creation_input_token_cost_above_1hr: self
                .cache_creation_input_token_cost_above_1hr
                * multiplier,
            output_cost_per_reasoning_token: self.output_cost_per_reasoning_token * multiplier,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenTypeCostBreakdown {
    pub reasoning_cost: f64,
    pub cache_read_cost: f64,
    pub cache_creation_cost: f64,
    pub rates: Option<BilledTokenRates>,
}

#[derive(Clone, Copy, Debug)]
pub struct BilledRatesRequest<'a> {
    pub model_info: Option<&'a Value>,
    pub usage: &'a ChatUsage,
    pub provider: Option<&'a str>,
    pub service_tier: Option<&'a str>,
    pub data_residency: Option<&'a str>,
    pub vertex_location: Option<&'a str>,
    pub at: Timestamp,
    pub custom_cost_per_token: Option<CustomTokenRates>,
}

fn custom_pricing_rates(custom: CustomTokenRates) -> BilledTokenRates {
    let read = custom.cache_read.unwrap_or(custom.input);
    let write = custom.cache_creation.unwrap_or(custom.input);
    BilledTokenRates {
        input_cost_per_token: custom.input,
        output_cost_per_token: custom.output,
        cache_read_input_token_cost: read,
        cache_read_input_audio_token_cost: read,
        cache_creation_input_token_cost: write,
        cache_creation_input_token_cost_above_1hr: write,
        output_cost_per_reasoning_token: custom.output,
    }
}

pub fn calculate_billed_token_rates(request: BilledRatesRequest<'_>) -> Option<BilledTokenRates> {
    if let Some(custom) = request.custom_cost_per_token {
        return Some(custom_pricing_rates(custom));
    }
    let model_info = apply_provider_cache_read_default(request.model_info?, request.provider);
    let base = get_token_base_cost(
        &model_info,
        request.usage.prompt_tokens,
        request.service_tier,
        uses_inclusive_token_thresholds(request.provider),
        request.at,
    );
    let reasoning = resolve_billed_reasoning_rate(
        request.usage,
        &model_info,
        request.service_tier,
        base.output,
        request.at,
    );
    let audio_cache_read = get_cost_per_unit(
        &model_info,
        &tier_key("cache_read_input_audio_token_cost", request.service_tier),
        None,
    )
    .unwrap_or(base.cache_read);
    let multiplier = combined_regional_multiplier(
        &model_info,
        request
            .usage
            .extra
            .get("inference_geo")
            .and_then(Value::as_str),
        request.data_residency,
        request.vertex_location,
    );
    Some(
        BilledTokenRates {
            input_cost_per_token: base.input,
            output_cost_per_token: base.output,
            cache_read_input_token_cost: base.cache_read,
            cache_read_input_audio_token_cost: audio_cache_read,
            cache_creation_input_token_cost: base.cache_creation,
            cache_creation_input_token_cost_above_1hr: base.cache_creation_above_1hr,
            output_cost_per_reasoning_token: reasoning,
        }
        .scaled(multiplier),
    )
}

fn coerce_token_count(value: Option<&Value>) -> u64 {
    match value {
        Some(Value::Bool(true)) => 1,
        Some(Value::Number(number)) => number.as_u64().unwrap_or(0),
        _ => 0,
    }
}

pub fn calculate_token_type_cost_breakdown(
    request: BilledRatesRequest<'_>,
) -> TokenTypeCostBreakdown {
    let Some(rates) = calculate_billed_token_rates(request) else {
        return TokenTypeCostBreakdown {
            reasoning_cost: 0.0,
            cache_read_cost: 0.0,
            cache_creation_cost: 0.0,
            rates: None,
        };
    };
    let prompt = request
        .usage
        .prompt_tokens_details
        .as_ref()
        .map(|_| parse_prompt_tokens_details(request.usage));
    let cache_read = prompt
        .as_ref()
        .map(|details| details.cache_hit_tokens)
        .filter(|tokens| *tokens > 0)
        .unwrap_or_else(|| coerce_token_count(request.usage.extra.get("_cache_read_input_tokens")));
    let cached_audio = prompt
        .as_ref()
        .map_or(0, |details| details.cache_hit_audio_tokens);
    let cache_creation = prompt
        .as_ref()
        .map(|details| details.cache_creation_tokens)
        .filter(|tokens| *tokens > 0)
        .unwrap_or_else(|| {
            coerce_token_count(request.usage.extra.get("_cache_creation_input_tokens"))
        });
    let reasoning = request
        .usage
        .completion_tokens_details
        .as_ref()
        .map(|_| parse_completion_tokens_details(request.usage).reasoning_tokens)
        .filter(|tokens| *tokens > 0)
        .unwrap_or_else(|| coerce_token_count(request.usage.extra.get("reasoning_tokens")));
    let cache_creation_cost = if request.custom_cost_per_token.is_some() {
        cache_creation as f64 * rates.cache_creation_input_token_cost
    } else {
        calculate_cache_writing_cost(
            cache_creation,
            prompt
                .as_ref()
                .and_then(|details| details.cache_creation_token_details.as_ref()),
            rates.cache_creation_input_token_cost_above_1hr,
            rates.cache_creation_input_token_cost,
        )
    };
    TokenTypeCostBreakdown {
        reasoning_cost: reasoning as f64 * rates.output_cost_per_reasoning_token,
        cache_read_cost: cache_read.saturating_sub(cached_audio) as f64
            * rates.cache_read_input_token_cost
            + cached_audio as f64 * rates.cache_read_input_audio_token_cost,
        cache_creation_cost,
        rates: Some(rates),
    }
}

fn billed_rates_request<'a>(
    model_info: Option<&'a Value>,
    request: ModelCostRequest<'a>,
    custom_cost_per_token: Option<CustomTokenRates>,
) -> BilledRatesRequest<'a> {
    BilledRatesRequest {
        model_info,
        usage: request.usage,
        provider: request.provider,
        service_tier: request.service_tier,
        data_residency: request.data_residency,
        vertex_location: request.vertex_location,
        at: request.at,
        custom_cost_per_token,
    }
}

pub fn get_billed_token_rates(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    custom_cost_per_token: Option<CustomTokenRates>,
) -> Option<BilledTokenRates> {
    let model_info = catalog.entry(request.model, request.provider, request.region);
    calculate_billed_token_rates(billed_rates_request(
        model_info.as_deref(),
        request,
        custom_cost_per_token,
    ))
}

pub fn get_token_type_cost_breakdown(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    custom_cost_per_token: Option<CustomTokenRates>,
) -> TokenTypeCostBreakdown {
    let model_info = catalog.entry(request.model, request.provider, request.region);
    calculate_token_type_cost_breakdown(billed_rates_request(
        model_info.as_deref(),
        request,
        custom_cost_per_token,
    ))
}
