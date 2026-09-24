use jiff::Timestamp;
use serde_json::Value;

use crate::call_type::CallTypes;
use crate::catalog::{ModelCostRequest, ModelInfoCatalog};
use crate::cost_calculator::cost_per_token as catalog_cost_per_token;
use crate::error::CostError;
use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::per_second::per_second_pricing_cost;
use crate::provider::LlmProviders;
use crate::regional_uplift::get_vertex_regional_endpoint_uplift;
use crate::responses_usage::ChatUsage;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostRoute {
    PerCharacter,
    PerToken,
}

pub fn cost_router(model: &str, provider: &str, call_type: &str) -> CostRoute {
    let token_model = [
        "claude",
        "llama",
        "mistral",
        "jamba",
        "codestral",
        "gemma",
        "gemini-2",
    ]
    .iter()
    .any(|name| model.contains(name));
    let embedding = matches!(
        call_type.parse::<CallTypes>(),
        Ok(CallTypes::embedding | CallTypes::aembedding)
    );
    if LlmProviders::VERTEX_AI.matches(Some(provider)) && (token_model || embedding) {
        CostRoute::PerToken
    } else {
        CostRoute::PerCharacter
    }
}

fn rate(model_info: &Value, key: &str) -> Option<f64> {
    model_info.get(key).and_then(Value::as_f64)
}

pub fn handle_128k_pricing(model_info: &Value, usage: &ChatUsage) -> (f64, f64) {
    let input_key = if usage.prompt_tokens > 128_000
        && rate(model_info, "input_cost_per_token_above_128k_tokens").is_some()
    {
        "input_cost_per_token_above_128k_tokens"
    } else {
        "input_cost_per_token"
    };
    let output_key = if usage.completion_tokens > 128_000
        && rate(model_info, "output_cost_per_token_above_128k_tokens").is_some()
    {
        "output_cost_per_token_above_128k_tokens"
    } else {
        "output_cost_per_token"
    };
    (
        usage.prompt_tokens as f64 * rate(model_info, input_key).unwrap_or(0.0),
        usage.completion_tokens as f64 * rate(model_info, output_key).unwrap_or(0.0),
    )
}

pub fn cost_per_token(
    usage: &ChatUsage,
    model_info: &Value,
    service_tier: Option<&str>,
    vertex_location: Option<&str>,
    at: Timestamp,
) -> (f64, f64) {
    if rate(model_info, "input_cost_per_token_above_128k_tokens").is_some()
        || rate(model_info, "output_cost_per_token_above_128k_tokens").is_some()
    {
        let (input, output) = handle_128k_pricing(model_info, usage);
        let uplift = get_vertex_regional_endpoint_uplift(model_info, vertex_location);
        return (input * uplift, output * uplift);
    }
    calculate_generic_cost_from_model_info_with_region(
        usage,
        model_info,
        service_tier,
        false,
        None,
        vertex_location,
        at,
    )
}

pub fn cost_per_character(
    model: &str,
    usage: &ChatUsage,
    model_info: &Value,
    characters: (Option<f64>, Option<f64>),
    service_tier: Option<&str>,
    vertex_location: Option<&str>,
    at: Timestamp,
) -> (f64, f64) {
    let (prompt_characters, completion_characters) = characters;
    let fallback = || cost_per_token(usage, model_info, service_tier, None, at);
    let dynamic = !matches!(model, "gemini-1.0-pro" | "gemini-pro" | "gemini-2");
    let input_key = if prompt_characters.is_some_and(|count| count * 4.0 > 128_000.0) && dynamic {
        "input_cost_per_character_above_128k_tokens"
    } else {
        "input_cost_per_character"
    };
    let output_key =
        if completion_characters.is_some_and(|count| count * 4.0 > 128_000.0) && dynamic {
            "output_cost_per_character_above_128k_tokens"
        } else {
            "output_cost_per_character"
        };
    let input = prompt_characters
        .zip(rate(model_info, input_key))
        .map_or_else(|| fallback().0, |(count, price)| count * price);
    let output = completion_characters
        .zip(rate(model_info, output_key))
        .map_or_else(
            || fallback().1,
            |(count, price)| {
                let quantity = if output_key == "output_cost_per_character_above_128k_tokens" {
                    usage.completion_tokens as f64
                } else {
                    count
                };
                quantity * price
            },
        );
    let uplift = get_vertex_regional_endpoint_uplift(model_info, vertex_location);
    (input * uplift, output * uplift)
}

pub fn vertex_cost(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    call_type: &str,
    prompt_characters: Option<f64>,
    completion_characters: Option<f64>,
) -> Result<(f64, f64), CostError> {
    if let Some(cost) = catalog
        .entry(request.model, request.provider, request.region)
        .and_then(|info| per_second_pricing_cost(&info, request.response_time_ms))
    {
        return Ok(cost);
    }
    let model_without_prefix = request
        .model
        .split_once('/')
        .map_or(request.model, |(_, model)| model);
    if cost_router(
        model_without_prefix,
        request.provider.unwrap_or(""),
        call_type,
    ) == CostRoute::PerToken
    {
        return catalog_cost_per_token(catalog, request);
    }
    let entry = catalog
        .entry(request.model, request.provider, request.region)
        .ok_or(CostError::ModelNotFound)?;
    Ok(cost_per_character(
        model_without_prefix,
        request.usage,
        &entry,
        (prompt_characters, completion_characters),
        request.service_tier,
        request.vertex_location,
        request.at,
    ))
}
