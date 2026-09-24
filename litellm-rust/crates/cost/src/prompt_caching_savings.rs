use jiff::Timestamp;
use serde_json::Value;

use crate::base_rate_selection::get_token_base_cost;
use crate::base_rate_selection::uses_inclusive_token_thresholds;
use crate::catalog::{ModelCostRequest, ModelInfoCatalog};
use crate::generic_input::calculate_cache_writing_cost;
use crate::generic_usage::parse_prompt_tokens_details;
use crate::provider_cache::apply_provider_cache_read_default;
use crate::regional_uplift::{get_regional_uplift_multiplier, get_vertex_regional_endpoint_uplift};
use crate::responses_usage::ChatUsage;

#[derive(Clone, Copy, Debug)]
pub struct PromptCachingSavingsRequest<'a> {
    pub model_info: &'a Value,
    pub usage: &'a ChatUsage,
    pub provider: Option<&'a str>,
    pub service_tier: Option<&'a str>,
    pub data_residency: Option<&'a str>,
    pub vertex_location: Option<&'a str>,
    pub at: Timestamp,
}

pub fn calculate_prompt_caching_savings(request: PromptCachingSavingsRequest<'_>) -> f64 {
    let model_info = apply_provider_cache_read_default(request.model_info, request.provider);
    let base = get_token_base_cost(
        &model_info,
        request.usage.prompt_tokens,
        request.service_tier,
        uses_inclusive_token_thresholds(request.provider),
        request.at,
    );
    let details = parse_prompt_tokens_details(request.usage);
    let write_rate = if base.cache_creation == 0.0 {
        base.input
    } else {
        base.cache_creation
    };
    let one_hour_rate = if base.cache_creation_above_1hr == 0.0 {
        write_rate
    } else {
        base.cache_creation_above_1hr
    };
    let read_discount = details.cache_hit_tokens as f64 * (base.input - base.cache_read).max(0.0);
    let write_premium = calculate_cache_writing_cost(
        details.cache_creation_tokens,
        details.cache_creation_token_details.as_ref(),
        one_hour_rate - base.input,
        write_rate - base.input,
    );
    let uplift = get_regional_uplift_multiplier(request.model_info, request.data_residency)
        * get_vertex_regional_endpoint_uplift(request.model_info, request.vertex_location);
    (read_discount - write_premium) * uplift
}

pub fn prompt_caching_savings_for_model(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
) -> Option<f64> {
    let model_info = catalog.entry(request.model, request.provider, request.region)?;
    Some(calculate_prompt_caching_savings(
        PromptCachingSavingsRequest {
            model_info: &model_info,
            usage: request.usage,
            provider: request.provider,
            service_tier: request.service_tier,
            data_residency: request.data_residency,
            vertex_location: request.vertex_location,
            at: request.at,
        },
    ))
}
