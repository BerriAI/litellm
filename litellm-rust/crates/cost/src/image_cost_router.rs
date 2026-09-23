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

pub fn deployment_pricing(model_info: Option<&Value>) -> Option<Value> {
    let model_info = model_info?;
    let prices = model_info
        .as_object()?
        .iter()
        .filter(|(key, value)| key.contains("cost_per") && !value.is_null())
        .filter_map(|(key, _)| {
            get_cost_per_unit(model_info, key, None).map(|price| (key.clone(), Value::from(price)))
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
        Some("bedrock") => Err(ImageCostRouteError::UnsupportedProvider),
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
