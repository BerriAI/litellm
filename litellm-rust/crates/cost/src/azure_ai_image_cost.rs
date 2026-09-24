use jiff::Timestamp;
use serde_json::Value;

use crate::catalog::ModelInfoCatalog;
use crate::cost_calculator::{DefaultImageCostRequest, default_image_cost_calculator};
use crate::error::CostError;
use crate::generic_input::get_cost_per_unit;
use crate::image_response_cost::calculate_image_response_cost_from_usage;

#[derive(Clone, Copy, Debug)]
pub struct AzureAiImageRequest<'a> {
    pub catalog: &'a ModelInfoCatalog,
    pub model: &'a str,
    pub image_response: &'a Value,
    pub model_info: &'a Value,
    pub supplied_model_info: Option<&'a Value>,
    pub shared_pricing: Option<&'a Value>,
    pub size: Option<&'a str>,
    pub n: Option<u64>,
    pub optional_params: &'a Value,
    pub at: Timestamp,
}

pub fn input_cost_per_pixel(resolved: &Value, shared_entry: Option<&Value>) -> f64 {
    get_cost_per_unit(resolved, "input_cost_per_pixel", None)
        .or_else(|| {
            shared_entry.and_then(|info| get_cost_per_unit(info, "input_cost_per_pixel", None))
        })
        .unwrap_or(0.0)
}

pub fn cost_calculator(request: AzureAiImageRequest<'_>) -> Result<f64, CostError> {
    if let Some(cost) = calculate_image_response_cost_from_usage(
        request.image_response,
        request.model_info,
        Some("azure_ai"),
        request.at,
    ) {
        return Ok(cost);
    }
    let count = request.n.unwrap_or_else(|| {
        request
            .image_response
            .get("data")
            .and_then(Value::as_array)
            .map_or(0, |images| images.len() as u64)
    });
    let output_rate = request
        .model_info
        .get("output_cost_per_image")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    if output_rate != 0.0 {
        return Ok(output_rate * count as f64);
    }
    if input_cost_per_pixel(request.model_info, request.shared_pricing) == 0.0 {
        return Ok(0.0);
    }
    let width = request.optional_params.get("width").and_then(Value::as_u64);
    let height = request
        .optional_params
        .get("height")
        .and_then(Value::as_u64);
    let size = match (width, height) {
        (Some(width), Some(height)) if width > 0 && height > 0 => Some(format!("{width}x{height}")),
        _ => request
            .size
            .or_else(|| request.image_response.get("size").and_then(Value::as_str))
            .map(str::to_owned),
    };
    default_image_cost_calculator(
        request.catalog,
        DefaultImageCostRequest {
            model: request
                .model_info
                .get("key")
                .and_then(Value::as_str)
                .unwrap_or(request.model),
            provider: Some("azure_ai"),
            quality: None,
            n: Some(count),
            size: size.as_deref(),
            supplied_model_info: request.supplied_model_info,
        },
    )
}
