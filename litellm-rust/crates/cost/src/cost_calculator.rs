//! The public cost entry points mirroring litellm/cost_calculator.py.

use serde_json::Value;

use crate::anthropic_cost::fast_speed_multiplier;
use crate::azure_ai_cost::azure_ai_cost_per_token;
use crate::azure_cost::output_per_second_cost;
use crate::base_rate_selection::uses_inclusive_token_thresholds;
use crate::batch::batch_cost_from_model_info;
use crate::catalog::{CostCall, ModelCostRequest, ModelInfoCatalog};
use crate::custom_pricing::{CustomPricing, cost_from_chat_usage};
use crate::dashscope_cost::cost_per_token as dashscope_cost_per_token;
use crate::databricks_cost::databricks_cost_per_token;
use crate::error::CostError;
use crate::fireworks_cost::{
    FireworksThresholds, cost_per_token as fireworks_cost_per_token, get_base_model_for_pricing,
};
use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::lemonade_cost::lemonade_cost_per_token;
use crate::ocr_cost::ocr_cost;
use crate::per_second::{has_token_or_tiered_pricing, per_second_pricing_cost};
use crate::perplexity_cost::cost_per_token as perplexity_cost_per_token;
use crate::provider::LlmProviders;
use crate::provider_cache::apply_provider_cache_read_default;
use crate::regional_uplift::get_provider_specific_geo_multiplier;
use crate::together_cost::{
    TogetherThresholds, get_model_params_and_category, has_together_registry_pricing,
    together_ai_cost_per_token,
};
use crate::vertex_cost::{cost_per_token as vertex_cost_per_token, vertex_cost};
use crate::xai_cost::{cost_per_token as xai_cost_per_token, reported_cost as xai_reported_cost};

pub fn cost_per_token_with_custom(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    pricing: CustomPricing,
) -> Result<(f64, f64), CostError> {
    match cost_from_chat_usage(request.usage, pricing, request.response_time_ms)? {
        Some(cost) => Ok((cost.input, cost.output)),
        None => cost_per_token(catalog, request),
    }
}

pub fn cost_per_token_for_call_with_custom(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    call: CostCall<'_>,
    pricing: CustomPricing,
) -> Result<(f64, f64), CostError> {
    match cost_from_chat_usage(request.usage, pricing, request.response_time_ms)? {
        Some(cost) => Ok((cost.input, cost.output)),
        None => cost_per_token_for_call(catalog, request, call),
    }
}

pub fn cost_per_token_for_call(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    call: CostCall<'_>,
) -> Result<(f64, f64), CostError> {
    let provider = request.provider.and_then(LlmProviders::parse);
    match call {
        CostCall::Token {
            call_type,
            prompt_characters,
            completion_characters,
            request_model,
        } => {
            if provider == Some(LlmProviders::VERTEX_AI) {
                return vertex_cost(
                    catalog,
                    request,
                    call_type,
                    prompt_characters,
                    completion_characters,
                );
            }
            if provider == Some(LlmProviders::TOGETHER_AI)
                && matches!(
                    crate::call_type::CallTypes::parse(call_type),
                    Some(crate::call_type::CallTypes::embedding)
                        | Some(crate::call_type::CallTypes::aembedding)
                )
            {
                return together_ai_cost_per_token(catalog, request, call_type);
            }
            if provider == Some(LlmProviders::AZURE_AI) {
                return azure_ai_cost_per_token(catalog, request, request_model);
            }
            Ok(cost_per_token(catalog, request)?)
        }
        CostCall::Speech { prompt_characters } => {
            Ok(catalog.speech_cost(request, prompt_characters)?)
        }
        CostCall::Transcription { duration_seconds } => {
            Ok(catalog.transcription_cost(request, duration_seconds)?)
        }
        CostCall::Rerank { billed_units } => {
            let provider = request.provider.ok_or(CostError::MissingProvider)?;
            Ok(catalog.rerank_cost(request.model, provider, request.region, billed_units))
        }
        CostCall::VectorStoreSearch { api_type } => {
            let provider = request.provider.ok_or(CostError::MissingProvider)?;
            Ok(catalog.vector_store_search_cost(provider, api_type))
        }
        CostCall::Search {
            number_of_queries,
            optional_params,
        } => Ok(catalog.search_provider_cost_per_query(
            request.model,
            request.provider,
            number_of_queries.filter(|count| *count > 0).unwrap_or(1),
            optional_params,
        )?),
        CostCall::Ocr {
            response,
            deployment_info,
        } => {
            let published = catalog.entry(request.model, request.provider, request.region);
            Ok(ocr_cost(response, deployment_info, published)?)
        }
        CostCall::Batch { deployment_info } => {
            let published = catalog.entry(request.model, request.provider, request.region);
            let has_deployment_rates = deployment_info.is_some_and(|info| {
                [
                    "input_cost_per_token_batches",
                    "input_cost_per_token",
                    "output_cost_per_token_batches",
                    "output_cost_per_token",
                ]
                .into_iter()
                .any(|key| info.get(key).is_some_and(|value| !value.is_null()))
            });
            let info = if has_deployment_rates {
                deployment_info
            } else {
                published.or(deployment_info)
            };
            let Some(info) = info else {
                return Ok((0.0, 0.0));
            };
            let cost = batch_cost_from_model_info(
                info,
                request.usage,
                request.provider,
                request.data_residency,
            )?;
            Ok((cost.prompt, cost.completion))
        }
    }
}

pub fn cost_per_token(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
) -> Result<(f64, f64), CostError> {
    if let Some(cost) = catalog
        .entry(request.model, request.provider, request.region)
        .and_then(|info| per_second_pricing_cost(info, request.response_time_ms))
    {
        return Ok(cost);
    }
    let provider = request.provider.and_then(LlmProviders::parse);
    if provider == Some(LlmProviders::AZURE_AI) {
        return azure_ai_cost_per_token(catalog, request, None);
    }
    if provider == Some(LlmProviders::DATABRICKS) {
        return databricks_cost_per_token(catalog, request);
    }
    if provider == Some(LlmProviders::LEMONADE) {
        return Ok(lemonade_cost_per_token(catalog, request));
    }
    if provider == Some(LlmProviders::PERPLEXITY)
        && let Some(cost) = request.usage.cost
    {
        return Ok((0.0, cost));
    }
    if provider == Some(LlmProviders::XAI)
        && let Some(cost) = xai_reported_cost(request.usage)
    {
        return Ok((0.0, cost));
    }
    let needs_together_fallback = (provider == Some(LlmProviders::TOGETHER_AI)
        || request.model.contains("togethercomputer")
        || request.model.contains("together_ai"))
        && !has_together_registry_pricing(request.model, catalog.entries());
    let together_fallback = needs_together_fallback.then(|| {
        get_model_params_and_category(request.model, "completion", TogetherThresholds::default())
    });
    let key = match together_fallback.as_deref() {
        Some(category) => catalog.select_model_key(category, None, request.region),
        None => catalog.select_model_key(request.model, request.provider, request.region),
    }
    .or_else(|| {
        (request.provider == Some("fireworks_ai"))
            .then(|| get_base_model_for_pricing(request.model, FireworksThresholds::default()))
            .and_then(|category| {
                catalog.select_model_key(category, request.provider, request.region)
            })
    })
    .ok_or(CostError::ModelNotFound)?;
    let model_info =
        apply_provider_cache_read_default(catalog.entry_for_key(key), request.provider);
    if let Some(cost) = per_second_pricing_cost(&model_info, request.response_time_ms) {
        return Ok(cost);
    }
    if provider == Some(LlmProviders::AZURE)
        && let Some(cost) = output_per_second_cost(&model_info, request.response_time_ms)
    {
        return Ok(cost);
    }
    match provider {
        Some(LlmProviders::PERPLEXITY) => {
            return Ok(perplexity_cost_per_token(
                request.usage,
                &model_info,
                request.at,
            ));
        }
        Some(LlmProviders::XAI) => {
            return Ok(xai_cost_per_token(request.usage, &model_info, request.at));
        }
        Some(
            LlmProviders::DASHSCOPE | LlmProviders::QWENCLOUD | LlmProviders::QWEN_AI_PLATFORM,
        ) => {
            return Ok(dashscope_cost_per_token(
                request.usage,
                &model_info,
                request.at,
            ));
        }
        Some(LlmProviders::FIREWORKS_AI) => {
            return Ok(fireworks_cost_per_token(
                request.usage,
                catalog.entry_for_key(key),
                request.at,
            ));
        }
        Some(LlmProviders::VERTEX_AI) => {
            return Ok(vertex_cost_per_token(
                request.usage,
                &model_info,
                request.service_tier,
                request.vertex_location,
                request.at,
            ));
        }
        _ => {}
    }
    let dispatches_before_the_gate = matches!(
        provider,
        Some(
            LlmProviders::VERTEX_AI
                | LlmProviders::ANTHROPIC
                | LlmProviders::BEDROCK
                | LlmProviders::OPENAI
                | LlmProviders::DATABRICKS
                | LlmProviders::FIREWORKS_AI
                | LlmProviders::AZURE
                | LlmProviders::GEMINI
                | LlmProviders::DEEPSEEK
                | LlmProviders::TENCENT
                | LlmProviders::PERPLEXITY
                | LlmProviders::XAI
                | LlmProviders::LEMONADE
                | LlmProviders::DASHSCOPE
                | LlmProviders::QWENCLOUD
                | LlmProviders::QWEN_AI_PLATFORM
                | LlmProviders::AZURE_AI
        )
    );
    if !dispatches_before_the_gate && !has_token_or_tiered_pricing(&model_info) {
        return Ok((0.0, 0.0));
    }
    let cost = calculate_generic_cost_from_model_info_with_region(
        request.usage,
        &model_info,
        request.service_tier,
        uses_inclusive_token_thresholds(request.provider),
        request.data_residency,
        request.vertex_location,
        request.at,
    );
    let speed = if provider == Some(LlmProviders::ANTHROPIC) {
        fast_speed_multiplier(&model_info, request.usage)
            * get_provider_specific_geo_multiplier(
                &model_info,
                request
                    .usage
                    .extra
                    .get("inference_geo")
                    .and_then(Value::as_str),
            )
    } else {
        1.0
    };
    Ok((cost.0 * speed, cost.1 * speed))
}
