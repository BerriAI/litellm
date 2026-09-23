use jiff::Timestamp;
use serde_json::Value;

use crate::catalog::{
    AzureAiImageCatalogRequest, CatalogError, CatalogImageError, DefaultImageCostRequest,
    ModelInfoCatalog,
};
use crate::generic_input::get_cost_per_unit;

#[derive(Clone, Copy, Debug)]
pub struct ImageCostRouteRequest<'a> {
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub image_response: &'a Value,
    pub call_type: Option<&'a str>,
    pub quality: Option<&'a str>,
    pub size: Option<&'a str>,
    pub n: Option<u64>,
    pub optional_params: &'a Value,
    pub supplied_model_info: Option<&'a Value>,
    pub at: Timestamp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageCostRouteError {
    Catalog(CatalogError),
    Image(CatalogImageError),
    UnsupportedProvider,
}

impl From<CatalogError> for ImageCostRouteError {
    fn from(value: CatalogError) -> Self {
        Self::Catalog(value)
    }
}

impl From<CatalogImageError> for ImageCostRouteError {
    fn from(value: CatalogImageError) -> Self {
        Self::Image(value)
    }
}

pub fn call_type_has_image_response(call_type: &str) -> bool {
    matches!(
        call_type,
        "image_generation"
            | "aimage_generation"
            | "passthrough-image-generation"
            | "image_edit"
            | "aimage_edit"
    )
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

pub fn route_image_generation_cost_calculator(
    catalog: &ModelInfoCatalog,
    request: ImageCostRouteRequest<'_>,
) -> Result<f64, ImageCostRouteError> {
    let pricing = deployment_pricing(request.supplied_model_info);
    let resolved_size = request
        .size
        .or_else(|| request.image_response.get("size").and_then(Value::as_str))
        .or_else(|| requested_image_size(request.optional_params))
        .or(Some("1024-x-1024"));
    let resolved_quality = request
        .quality
        .or_else(|| {
            request
                .image_response
                .get("quality")
                .and_then(Value::as_str)
        })
        .or_else(|| {
            request
                .optional_params
                .get("quality")
                .and_then(Value::as_str)
        })
        .or(Some("standard"));
    let resolved_n = request.n.or_else(|| {
        request
            .image_response
            .get("data")
            .and_then(Value::as_array)
            .map(|data| data.len() as u64)
    });
    match request.provider {
        Some("vertex_ai") => Ok(catalog.google_image_generation_cost(
            request.model,
            "vertex_ai",
            request.image_response,
            pricing.as_ref(),
            request.at,
        )?),
        Some("gemini") if matches!(request.call_type, Some("image_edit" | "aimage_edit")) => {
            Ok(catalog.google_image_edit_cost(
                request.model,
                "gemini",
                request.image_response,
                pricing.as_ref(),
                request.at,
            )?)
        }
        Some("gemini") => Ok(catalog.google_image_generation_cost(
            request.model,
            "gemini",
            request.image_response,
            pricing.as_ref(),
            request.at,
        )?),
        Some("azure_ai") => Ok(catalog.azure_ai_image_generation_cost(
            AzureAiImageCatalogRequest {
                model: request.model,
                image_response: request.image_response,
                size: resolved_size,
                n: resolved_n,
                optional_params: request.optional_params,
                supplied_model_info: pricing.as_ref(),
                at: request.at,
            },
        )?),
        Some(provider @ ("recraft" | "aiml" | "cometapi" | "runwayml")) => Ok(catalog
            .flat_image_generation_cost(
                request.model,
                provider,
                request.image_response,
                pricing.as_ref(),
            )?),
        Some("fal_ai") => Ok(catalog.fal_ai_image_generation_cost(
            request.model,
            request.image_response,
            request.optional_params,
            pricing.as_ref(),
        )?),
        Some("bedrock") => Ok(catalog.bedrock_image_generation_cost(
            request.model,
            request.image_response,
            resolved_size,
            request.optional_params,
        )?),
        Some(provider @ ("openai" | "azure"))
            if request.model.to_ascii_lowercase().contains("gpt-image") =>
        {
            Ok(catalog.openai_image_generation_cost(
                request.model,
                provider,
                request.image_response,
                pricing.as_ref(),
                request.at,
            )?)
        }
        _ => Ok(
            catalog.default_image_cost_calculator(DefaultImageCostRequest {
                model: request.model,
                provider: request.provider,
                quality: resolved_quality,
                n: resolved_n,
                size: resolved_size,
                supplied_model_info: pricing.as_ref(),
            })?,
        ),
    }
}
