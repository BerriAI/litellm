use super::pricing::lookup_model_pricing;
use super::types::{CostRequest, CostResponse, ModelPricing, Usage};
use crate::CoreResult;

pub fn calculate_cost(request: &CostRequest<'_>) -> CoreResult<CostResponse> {
    let pricing = lookup_model_pricing(request.model).ok_or_else(|| {
        crate::CoreError::InvalidRequest(format!("No pricing found for model: {}", request.model))
    })?;

    let (prompt_cost, completion_cost) =
        generic_cost_per_token(pricing, &request.usage, request.service_tier);

    Ok(CostResponse {
        prompt_cost_usd: prompt_cost,
        completion_cost_usd: completion_cost,
    })
}

fn generic_cost_per_token(
    pricing: &ModelPricing,
    usage: &Usage,
    service_tier: Option<&str>,
) -> (f64, f64) {
    let prompt_cost = calculate_input_cost(pricing, usage, service_tier);
    let completion_cost = calculate_output_cost(pricing, usage, service_tier);
    (prompt_cost, completion_cost)
}

fn calculate_input_cost(pricing: &ModelPricing, usage: &Usage, service_tier: Option<&str>) -> f64 {
    let (input_rate, cache_read_rate, cache_creation_rate) =
        get_input_rates(pricing, usage.prompt_tokens, service_tier);

    let details = usage.prompt_tokens_details.as_ref();

    let cache_hit_tokens = details.map(|d| d.cache_hit_tokens).unwrap_or(0);
    let cache_creation_tokens = details.map(|d| d.cache_creation_tokens).unwrap_or(0);
    let audio_tokens = details.map(|d| d.audio_tokens).unwrap_or(0);
    let image_tokens = details.map(|d| d.image_tokens).unwrap_or(0);
    let text_tokens = usage
        .prompt_tokens
        .saturating_sub(cache_hit_tokens + cache_creation_tokens + audio_tokens + image_tokens);

    let mut cost = 0.0;

    if text_tokens > 0 && input_rate > 0.0 {
        cost += (text_tokens as f64) * input_rate;
    }

    if cache_hit_tokens > 0 && cache_read_rate > 0.0 {
        cost += (cache_hit_tokens as f64) * cache_read_rate;
    }

    if cache_creation_tokens > 0 && cache_creation_rate > 0.0 {
        cost += (cache_creation_tokens as f64) * cache_creation_rate;
    }

    if audio_tokens > 0
        && let Some(audio_rate) = pricing.input_cost_per_audio_token
    {
        cost += (audio_tokens as f64) * audio_rate;
    }

    if image_tokens > 0
        && let Some(image_rate) = pricing.input_cost_per_token
    {
        cost += (image_tokens as f64) * image_rate;
    }

    cost
}

fn calculate_output_cost(pricing: &ModelPricing, usage: &Usage, service_tier: Option<&str>) -> f64 {
    let output_rate = get_output_rate(pricing, usage.prompt_tokens, service_tier);

    let details = usage.completion_tokens_details.as_ref();

    let reasoning_tokens = details.map(|d| d.reasoning_tokens).unwrap_or(0);
    let audio_tokens = details.map(|d| d.audio_tokens).unwrap_or(0);
    let text_tokens = usage
        .completion_tokens
        .saturating_sub(reasoning_tokens + audio_tokens);

    let mut cost = 0.0;

    if text_tokens > 0 && output_rate > 0.0 {
        cost += (text_tokens as f64) * output_rate;
    }

    if reasoning_tokens > 0 {
        if let Some(reasoning_rate) = pricing.output_cost_per_reasoning_token {
            cost += (reasoning_tokens as f64) * reasoning_rate;
        } else if output_rate > 0.0 {
            cost += (reasoning_tokens as f64) * output_rate;
        }
    }

    if audio_tokens > 0
        && let Some(audio_rate) = pricing.output_cost_per_audio_token
    {
        cost += (audio_tokens as f64) * audio_rate;
    }

    cost
}

fn get_input_rates(
    pricing: &ModelPricing,
    prompt_tokens: u64,
    service_tier: Option<&str>,
) -> (f64, f64, f64) {
    let base_input_rate = resolve_input_rate(pricing, prompt_tokens, service_tier);
    let cache_read_rate = resolve_cache_read_rate(pricing, prompt_tokens, service_tier);
    let cache_creation_rate = resolve_cache_creation_rate(pricing, prompt_tokens);

    (base_input_rate, cache_read_rate, cache_creation_rate)
}

fn resolve_input_rate(
    pricing: &ModelPricing,
    prompt_tokens: u64,
    service_tier: Option<&str>,
) -> f64 {
    if let Some(tier) = service_tier {
        match tier {
            "priority" => {
                if let Some(rate) = pricing.input_cost_per_token_priority {
                    return rate;
                }
            }
            "flex" => {
                if let Some(rate) = pricing.input_cost_per_token_flex {
                    return rate;
                }
            }
            _ => {}
        }
    }

    if prompt_tokens > 200_000
        && let Some(rate) = pricing.input_cost_per_token_above_200k_tokens
    {
        return rate;
    }

    if prompt_tokens > 128_000
        && let Some(rate) = pricing.input_cost_per_token_above_128k_tokens
    {
        return rate;
    }

    pricing.input_cost_per_token.unwrap_or(0.0)
}

fn resolve_cache_read_rate(
    pricing: &ModelPricing,
    prompt_tokens: u64,
    service_tier: Option<&str>,
) -> f64 {
    if let Some("priority") = service_tier
        && let Some(rate) = pricing.cache_read_input_tokens_cost
    {
        return rate;
    }

    if prompt_tokens > 200_000
        && let Some(rate) = pricing.cache_read_input_tokens_cost_above_200k_tokens
    {
        return rate;
    }

    pricing.cache_read_input_tokens_cost.unwrap_or(0.0)
}

fn resolve_cache_creation_rate(pricing: &ModelPricing, prompt_tokens: u64) -> f64 {
    if prompt_tokens > 200_000
        && let Some(rate) = pricing.cache_creation_input_token_cost_above_200k_tokens
    {
        return rate;
    }

    pricing.cache_creation_input_token_cost.unwrap_or(0.0)
}

fn get_output_rate(pricing: &ModelPricing, prompt_tokens: u64, service_tier: Option<&str>) -> f64 {
    if let Some(tier) = service_tier {
        match tier {
            "priority" => {
                if let Some(rate) = pricing.output_cost_per_token_priority {
                    return rate;
                }
            }
            "flex" => {
                if let Some(rate) = pricing.output_cost_per_token_flex {
                    return rate;
                }
            }
            _ => {}
        }
    }

    if prompt_tokens > 200_000
        && let Some(rate) = pricing.output_cost_per_token_above_200k_tokens
    {
        return rate;
    }

    if prompt_tokens > 128_000
        && let Some(rate) = pricing.output_cost_per_token_above_128k_tokens
    {
        return rate;
    }

    pricing.output_cost_per_token.unwrap_or(0.0)
}
