use jiff::Timestamp;
use serde_json::Value;

use crate::azure_ai_image_cost::{
    AzureAiImageRequest, cost_calculator as azure_ai_image_cost_calculator,
};
use crate::bedrock_image_cost::cost_calculator as bedrock_image_cost_calculator;
use crate::call_type::{CallTypes, PassthroughCallTypes};
use crate::catalog::ModelInfoCatalog;
use crate::cost_calculator::{DefaultImageCostRequest, default_image_cost_calculator};
use crate::error::CostError;
use crate::fal_ai_image_cost::cost_calculator as fal_ai_image_cost_calculator;
use crate::generic_input::get_cost_per_unit;
use crate::image_response_cost::{
    flat_image_cost, gemini_image_edit_cost, gemini_image_generation_cost,
    resolve_image_model_info, vertex_image_edit_cost, vertex_image_generation_cost,
};
use crate::openai_image_cost::cost_calculator as openai_image_cost_calculator;
use crate::provider::LlmProviders;

#[derive(Clone, Copy, Debug)]
pub struct ImageCostRouteRequest<'a> {
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub image_response: &'a Value,
    pub call_type: Option<CallTypes>,
    pub quality: Option<&'a str>,
    pub size: Option<&'a str>,
    pub n: Option<u64>,
    pub optional_params: &'a Value,
    pub supplied_model_info: Option<&'a Value>,
    pub at: Timestamp,
}

#[derive(Clone, Copy, Debug)]
pub struct AzureAiImageCatalogRequest<'a> {
    pub model: &'a str,
    pub image_response: &'a Value,
    pub size: Option<&'a str>,
    pub n: Option<u64>,
    pub optional_params: &'a Value,
    pub supplied_model_info: Option<&'a Value>,
    pub at: Timestamp,
}

pub fn call_type_has_image_response(call_type: &str) -> bool {
    matches!(
        call_type.parse::<CallTypes>(),
        Ok(CallTypes::image_generation
            | CallTypes::aimage_generation
            | CallTypes::image_edit
            | CallTypes::aimage_edit)
    ) || call_type.parse::<PassthroughCallTypes>().is_ok()
}

// CustomPricingLiteLLMParams.model_fields, litellm/types/utils.py as of 2026-09-22.
const DEPLOYMENT_PRICING_KEYS: &[&str] = &[
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_character",
    "output_cost_per_character",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "tiered_pricing",
    "input_cost_per_second",
    "output_cost_per_second",
    "output_cost_per_second_1080p",
    "output_cost_per_second_480p",
    "output_cost_per_second_720p",
    "output_cost_per_second_768p",
    "output_cost_per_second_2k",
    "output_cost_per_second_4k",
    "output_cost_per_image_512",
    "output_cost_per_image_1024",
    "output_cost_per_image_1536",
    "input_cost_per_pixel",
    "output_cost_per_pixel",
    "input_cost_per_token_flex",
    "input_cost_per_token_priority",
    "input_cost_per_token_ultrafast",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens_priority",
    "cache_creation_input_token_cost_above_272k_tokens_flex",
    "cache_creation_input_token_cost_flex",
    "cache_creation_input_token_cost_priority",
    "cache_creation_input_token_cost_ultrafast",
    "cache_creation_input_audio_token_cost",
    "cache_read_input_token_cost_flex",
    "cache_read_input_token_cost_priority",
    "cache_read_input_token_cost_ultrafast",
    "cache_read_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens_priority",
    "cache_read_input_token_cost_above_272k_tokens_priority",
    "cache_read_input_token_cost_above_272k_tokens_flex",
    "cache_read_input_token_cost_batches",
    "cache_read_input_token_cost_above_272k_tokens_batches",
    "cache_creation_input_token_cost_batches",
    "cache_creation_input_token_cost_above_272k_tokens_batches",
    "cache_read_input_audio_token_cost",
    "cache_read_input_image_token_cost",
    "input_cost_per_character_above_128k_tokens",
    "input_cost_per_audio_token",
    "input_cost_per_token_cache_hit",
    "input_cost_per_token_above_128k_tokens",
    "input_cost_per_token_above_200k_tokens",
    "input_cost_per_token_above_200k_tokens_priority",
    "input_cost_per_token_above_272k_tokens_priority",
    "input_cost_per_token_above_272k_tokens_flex",
    "input_cost_per_token_above_272k_tokens_batches",
    "input_cost_per_query",
    "input_cost_per_image",
    "input_cost_per_image_above_128k_tokens",
    "input_cost_per_audio_per_second",
    "input_cost_per_audio_per_second_above_128k_tokens",
    "input_cost_per_video_per_second",
    "input_cost_per_video_per_second_above_128k_tokens",
    "input_cost_per_video_per_second_above_15s_interval",
    "input_cost_per_video_per_second_above_8s_interval",
    "input_cost_per_audio_token_batches",
    "input_cost_per_image_token_batches",
    "input_cost_per_token_batches",
    "input_cost_per_video_token_batches",
    "output_cost_per_token_batches",
    "output_cost_per_token_flex",
    "output_cost_per_token_priority",
    "output_cost_per_token_ultrafast",
    "output_cost_per_audio_token",
    "output_cost_per_token_above_128k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "output_cost_per_token_above_200k_tokens_priority",
    "output_cost_per_token_above_272k_tokens_priority",
    "output_cost_per_token_above_272k_tokens_flex",
    "output_cost_per_token_above_272k_tokens_batches",
    "output_cost_per_character_above_128k_tokens",
    "output_cost_per_image",
    "output_cost_per_image_token",
    "output_cost_per_video_token",
    "output_cost_per_reasoning_token",
    "output_cost_per_reasoning_token_flex",
    "output_cost_per_reasoning_token_priority",
    "output_cost_per_video_per_second",
    "output_cost_per_audio_per_second",
    "search_context_cost_per_query",
    "google_maps_grounding_cost_per_query",
    "citation_cost_per_token",
    "cache_read_input_token_cost_above_272k_tokens",
    "cache_read_input_token_cost_above_512k_tokens",
    "input_cost_per_image_token",
    "input_cost_per_video_token",
    "input_cost_per_token_above_272k_tokens",
    "input_cost_per_token_above_512k_tokens",
    "output_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_512k_tokens",
    "output_vector_size",
    "ocr_cost_per_page",
    "ocr_cost_per_page_batches",
    "ocr_cost_per_credit",
    "annotation_cost_per_page",
    "annotation_cost_per_page_batches",
    "regional_processing_uplift_multiplier_eu",
    "regional_processing_uplift_multiplier_us",
    "regional_endpoint_uplift_multiplier",
];

pub fn deployment_pricing(model_info: Option<&Value>) -> Option<Value> {
    let model_info = model_info?;
    let prices = DEPLOYMENT_PRICING_KEYS
        .iter()
        .filter(|key| model_info.get(**key).is_some_and(|value| !value.is_null()))
        .filter_map(|key| {
            get_cost_per_unit(model_info, key, None)
                .map(|price| ((*key).to_string(), Value::from(price)))
        })
        .collect::<serde_json::Map<String, Value>>();
    (!prices.is_empty()).then_some(Value::Object(prices))
}

fn requested_image_size(optional_params: &Value) -> Option<&str> {
    let size = optional_params.get("size")?.as_str()?;
    let (width, height) = size.split_once("-x-").or_else(|| size.split_once('x'))?;
    (!width.is_empty()
        && !height.is_empty()
        && width.bytes().all(|byte| byte.is_ascii_digit())
        && height.bytes().all(|byte| byte.is_ascii_digit()))
    .then_some(size)
}

pub fn route_image_generation_cost_calculator<'a>(
    catalog: &ModelInfoCatalog,
    request: ImageCostRouteRequest<'a>,
) -> Result<f64, CostError> {
    let pricing = deployment_pricing(request.supplied_model_info);
    let non_empty = |value: Option<&'a str>| value.filter(|value| !value.is_empty());
    let resolved_size = non_empty(request.size)
        .or_else(|| non_empty(request.image_response.get("size").and_then(Value::as_str)))
        .or_else(|| requested_image_size(request.optional_params))
        .unwrap_or("1024-x-1024");
    let resolved_quality = non_empty(request.quality)
        .or_else(|| {
            non_empty(
                request
                    .image_response
                    .get("quality")
                    .and_then(Value::as_str),
            )
        })
        .or_else(|| {
            non_empty(
                request
                    .optional_params
                    .get("quality")
                    .and_then(Value::as_str),
            )
        })
        .unwrap_or("standard");
    let resolved_n = request.n.unwrap_or_else(|| {
        request
            .image_response
            .get("data")
            .and_then(Value::as_array)
            .map_or(0, |data| data.len() as u64)
    });
    let provider = request
        .provider
        .and_then(|provider| provider.parse::<LlmProviders>().ok());
    match provider {
        Some(LlmProviders::VERTEX_AI) => google_image_generation_cost(
            catalog,
            request.model,
            "vertex_ai",
            request.image_response,
            pricing.as_ref(),
            request.at,
        ),
        Some(LlmProviders::GEMINI)
            if matches!(
                request.call_type,
                Some(CallTypes::image_edit | CallTypes::aimage_edit)
            ) =>
        {
            google_image_edit_cost(
                catalog,
                request.model,
                "gemini",
                request.image_response,
                pricing.as_ref(),
                request.at,
            )
        }
        Some(LlmProviders::GEMINI) => google_image_generation_cost(
            catalog,
            request.model,
            "gemini",
            request.image_response,
            pricing.as_ref(),
            request.at,
        ),
        Some(LlmProviders::AZURE_AI) => azure_ai_image_generation_cost(
            catalog,
            AzureAiImageCatalogRequest {
                model: request.model,
                image_response: request.image_response,
                size: Some(resolved_size),
                n: Some(resolved_n),
                optional_params: request.optional_params,
                supplied_model_info: pricing.as_ref(),
                at: request.at,
            },
        ),
        Some(
            provider @ (LlmProviders::RECRAFT
            | LlmProviders::AIML
            | LlmProviders::COMETAPI
            | LlmProviders::RUNWAYML),
        ) => flat_image_generation_cost(
            catalog,
            request.model,
            provider.as_str(),
            request.image_response,
            pricing.as_ref(),
        ),
        Some(LlmProviders::FAL_AI) => fal_ai_image_cost_calculator(
            catalog,
            request.model,
            request.image_response,
            request.optional_params,
            pricing.as_ref(),
        )
        .ok_or(CostError::ModelNotFound),
        Some(LlmProviders::BEDROCK) => bedrock_image_cost_calculator(
            catalog,
            request.model,
            request.image_response,
            Some(resolved_size),
            request.optional_params,
        ),
        Some(provider @ (LlmProviders::OPENAI | LlmProviders::AZURE))
            if request.model.to_ascii_lowercase().contains("gpt-image") =>
        {
            openai_image_generation_cost(
                catalog,
                request.model,
                provider.as_str(),
                request.image_response,
                pricing.as_ref(),
                request.at,
            )
        }
        _ => default_image_cost_calculator(
            catalog,
            DefaultImageCostRequest {
                model: request.model,
                provider: request.provider,
                quality: Some(resolved_quality),
                n: Some(resolved_n),
                size: Some(resolved_size),
                supplied_model_info: pricing.as_ref(),
            },
        ),
    }
}

fn resolved_model_info(
    catalog: &ModelInfoCatalog,
    model: &str,
    provider: &str,
    supplied_model_info: Option<&Value>,
) -> Result<Value, CostError> {
    resolve_image_model_info(
        catalog
            .get_model_info(model, Some(provider))
            .ok()
            .map(|model_info| model_info.info)
            .as_deref(),
        supplied_model_info,
    )
    .ok_or(CostError::ModelNotFound)
}

pub fn google_image_generation_cost(
    catalog: &ModelInfoCatalog,
    model: &str,
    provider: &str,
    image_response: &Value,
    supplied_model_info: Option<&Value>,
    at: Timestamp,
) -> Result<f64, CostError> {
    let model_info = resolved_model_info(catalog, model, provider, supplied_model_info)?;
    match provider {
        "gemini" => Ok(gemini_image_generation_cost(
            image_response,
            &model_info,
            at,
        )),
        "vertex_ai" => Ok(vertex_image_generation_cost(
            image_response,
            &model_info,
            at,
        )),
        _ => Err(CostError::ModelNotFound),
    }
}

pub fn google_image_edit_cost(
    catalog: &ModelInfoCatalog,
    model: &str,
    provider: &str,
    image_response: &Value,
    supplied_model_info: Option<&Value>,
    at: Timestamp,
) -> Result<f64, CostError> {
    match provider {
        "gemini" => {
            let model_info = resolved_model_info(catalog, model, provider, supplied_model_info)?;
            Ok(gemini_image_edit_cost(image_response, &model_info, at))
        }
        "vertex_ai" => catalog
            .get_model_info(model, Some(provider))
            .map(|model_info| vertex_image_edit_cost(image_response, &model_info.info)),
        _ => Err(CostError::ModelNotFound),
    }
}

pub fn azure_ai_image_generation_cost(
    catalog: &ModelInfoCatalog,
    request: AzureAiImageCatalogRequest<'_>,
) -> Result<f64, CostError> {
    let selected = catalog.get_model_info(request.model, Some("azure_ai")).ok();
    let shared = selected.as_ref().map(|selected| &*selected.info);
    let model_info = resolve_image_model_info(shared, request.supplied_model_info)
        .ok_or(CostError::ModelNotFound)?;
    let shared_pricing = model_info
        .get("key")
        .and_then(Value::as_str)
        .and_then(|key| catalog.entries().get(key))
        .or(shared);
    azure_ai_image_cost_calculator(AzureAiImageRequest {
        catalog,
        model: selected
            .as_ref()
            .map_or(request.model, |selected| &selected.key),
        image_response: request.image_response,
        model_info: &model_info,
        supplied_model_info: request.supplied_model_info,
        shared_pricing,
        size: request.size,
        n: request.n,
        optional_params: request.optional_params,
        at: request.at,
    })
}

pub fn flat_image_generation_cost(
    catalog: &ModelInfoCatalog,
    model: &str,
    provider: &str,
    image_response: &Value,
    supplied_model_info: Option<&Value>,
) -> Result<f64, CostError> {
    let model_info = resolved_model_info(catalog, model, provider, supplied_model_info)?;
    Ok(flat_image_cost(image_response, &model_info))
}

pub fn openai_image_generation_cost(
    catalog: &ModelInfoCatalog,
    model: &str,
    provider: &str,
    image_response: &Value,
    supplied_model_info: Option<&Value>,
    at: Timestamp,
) -> Result<f64, CostError> {
    let model_info = resolved_model_info(catalog, model, provider, supplied_model_info)?;
    openai_image_cost_calculator(image_response, &model_info, provider, at)
}
