use std::collections::HashMap;

use serde_json::Value;

use crate::catalog::ModelInfoCatalog;
use crate::image_response_cost::resolve_image_model_info;
use crate::wire::py_real;

pub const PIXELS_PER_MEGAPIXEL: u64 = 1_048_576;

#[derive(Clone, Debug, PartialEq)]
pub struct KeyedRow {
    pub width: u64,
    pub height: u64,
    pub cost: f64,
    pub key: String,
}

pub fn parse_keyed_dimensions(size: &str) -> Option<(u64, u64)> {
    let (width, height) = size.split_once("-x-")?;
    let width = width.parse::<u64>().ok()?;
    let height = height.parse::<u64>().ok()?;
    (width > 0 && height > 0).then_some((width, height))
}

pub fn keyed_size(params: &Value) -> Option<(u64, u64)> {
    match params.get("image_size") {
        None | Some(Value::Null) => Some((1024, 768)),
        Some(Value::String(size)) if size == "auto" => Some((1024, 768)),
        Some(Value::Object(size)) => {
            Some((size.get("width")?.as_u64()?, size.get("height")?.as_u64()?))
        }
        Some(Value::String(size)) => match size.as_str() {
            "square_hd" => Some((1024, 1024)),
            "square" => Some((512, 512)),
            "portrait_4_3" => Some((768, 1024)),
            "portrait_16_9" => Some((576, 1024)),
            "landscape_4_3" => Some((1024, 768)),
            "landscape_16_9" => Some((1024, 576)),
            _ => None,
        },
        _ => None,
    }
    .filter(|(width, height)| *width > 0 && *height > 0)
}

pub fn image_dimensions(image: &Value) -> Option<(u64, u64)> {
    let fields = image.get("provider_specific_fields")?;
    let width = fields.get("width")?.as_u64()?;
    let height = fields.get("height")?.as_u64()?;
    (width > 0 && height > 0).then_some((width, height))
}

pub fn keyed_quality(params: &Value) -> &str {
    match params.get("quality").and_then(Value::as_str) {
        Some(quality) if quality != "auto" => quality,
        _ => "high",
    }
}

pub fn keyed_rows(entries: &HashMap<String, Value>, model: &str, quality: &str) -> Vec<KeyedRow> {
    let prefix = format!("fal_ai/{quality}/");
    let suffix = format!("/{model}");
    entries
        .iter()
        .filter_map(|(key, info)| {
            let size = key.strip_prefix(&prefix)?.strip_suffix(&suffix)?;
            let (width, height) = parse_keyed_dimensions(size)?;
            let cost = info.get("output_cost_per_image")?.as_f64()?;
            Some(KeyedRow {
                width,
                height,
                cost,
                key: key.clone(),
            })
        })
        .collect()
}

pub fn keyed_cost_per_image(rows: &[KeyedRow], image: &Value, params: &Value) -> Option<f64> {
    let (width, height) = image_dimensions(image)
        .or_else(|| keyed_size(params))
        .unwrap_or((1024, 768));
    let target_pixels = u128::from(width) * u128::from(height);
    rows.iter()
        .min_by_key(|row| {
            let pixels = u128::from(row.width) * u128::from(row.height);
            (pixels.abs_diff(target_pixels), pixels, row.key.as_str())
        })
        .map(|row| row.cost)
}

pub fn flat_cost_per_image(
    image: &Value,
    output_cost_per_image: f64,
    output_cost_per_pixel: Option<f64>,
) -> f64 {
    let Some(((width, height), pixel_rate)) = image_dimensions(image).zip(output_cost_per_pixel)
    else {
        return output_cost_per_image;
    };
    let pixels = u128::from(width) * u128::from(height);
    let megapixels = pixels.div_ceil(u128::from(PIXELS_PER_MEGAPIXEL));
    pixel_rate * PIXELS_PER_MEGAPIXEL as f64 * megapixels as f64
}

pub fn fal_ai_passthrough_cost(
    catalog: &ModelInfoCatalog,
    model: &str,
    request_body: &Value,
) -> Option<f64> {
    fal_ai_passthrough_cost_from_model_info(
        catalog.entries().get(&format!("fal_ai/{model}")),
        request_body,
    )
}

pub fn fal_ai_passthrough_cost_from_model_info(
    model_info: Option<&Value>,
    request_body: &Value,
) -> Option<f64> {
    let info = model_info?;
    let resolution = request_body
        .get("resolution")
        .and_then(|value| match value {
            Value::String(value) => Some(value.clone()),
            Value::Number(value) if value.as_i64().is_some() || value.as_u64().is_some() => {
                Some(value.to_string())
            }
            _ => None,
        });
    resolution
        .as_deref()
        .and_then(|resolution| info.get(format!("output_cost_per_image_{resolution}")))
        .and_then(Value::as_f64)
        .or_else(|| info.get("output_cost_per_image").and_then(Value::as_f64))
}

pub fn cost_calculator(
    catalog: &ModelInfoCatalog,
    model: &str,
    image_response: &Value,
    optional_params: &Value,
    deployment_prices: Option<&Value>,
) -> Option<f64> {
    let entries = catalog.entries();
    let images = image_response.get("data").and_then(Value::as_array);
    let count = images.map_or(0, Vec::len);
    if let Some(rate) = deployment_prices
        .and_then(|prices| prices.get("output_cost_per_image"))
        .and_then(Value::as_f64)
    {
        return Some(rate * count as f64);
    }
    let normalized_model = model.strip_prefix("fal_ai/").unwrap_or(model);
    let rows = keyed_rows(entries, normalized_model, keyed_quality(optional_params));
    let keyed = images
        .into_iter()
        .flatten()
        .map(|image| keyed_cost_per_image(&rows, image, optional_params))
        .collect::<Vec<_>>();
    if keyed.iter().all(Option::is_some) {
        return Some(keyed.into_iter().flatten().sum());
    }
    let shared = catalog
        .get_model_info(normalized_model, Some("fal_ai"))
        .ok()
        .map(|model_info| model_info.info);
    let resolved = resolve_image_model_info(shared.as_deref(), deployment_prices)?;
    let flat_rate = resolved
        .get("output_cost_per_image")
        .and_then(py_real)
        .unwrap_or(0.0);
    let pixel_rate = resolved.get("output_cost_per_pixel").and_then(py_real);
    Some(
        images
            .into_iter()
            .flatten()
            .zip(keyed)
            .map(|(image, keyed)| {
                keyed.unwrap_or_else(|| flat_cost_per_image(image, flat_rate, pixel_rate))
            })
            .sum(),
    )
}
