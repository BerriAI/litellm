use serde_json::Value;

use crate::catalog::ModelInfoCatalog;
use crate::error::CostError;
use crate::wire::py_real;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BedrockImageFamily {
    Titan,
    NovaCanvas,
    Stability3,
    Stability1,
}

pub fn get_config_class(model: &str) -> BedrockImageFamily {
    if model.contains("amazon.titan") {
        BedrockImageFamily::Titan
    } else if model.contains("amazon.nova-canvas") {
        BedrockImageFamily::NovaCanvas
    } else if model.contains("sd3") || model.contains("stable-image") {
        BedrockImageFamily::Stability3
    } else {
        BedrockImageFamily::Stability1
    }
}

pub fn stability1_pricing_key(
    model: &str,
    size: Option<&str>,
    optional_params: &Value,
) -> Result<String, CostError> {
    let steps = match optional_params.get("steps") {
        None => 50.0,
        Some(steps) => py_real(steps).ok_or(CostError::InvalidSteps)?,
    };
    let tier = if steps > 50.0 {
        "max-steps"
    } else {
        "50-steps"
    };
    Ok(format!("{}/{tier}/{model}", size.unwrap_or("1024-x-1024")))
}

pub fn cost_calculator(
    catalog: &ModelInfoCatalog,
    model: &str,
    image_response: &Value,
    size: Option<&str>,
    optional_params: &Value,
) -> Result<f64, CostError> {
    let model_info = match get_config_class(model) {
        BedrockImageFamily::Titan => catalog.get_model_info(model, None)?,
        BedrockImageFamily::NovaCanvas | BedrockImageFamily::Stability3 => {
            catalog.get_model_info(model, Some("bedrock"))?
        }
        BedrockImageFamily::Stability1 => catalog.get_model_info(
            &stability1_pricing_key(model, size.filter(|size| !size.is_empty()), optional_params)?,
            Some("bedrock"),
        )?,
    };
    let rate = model_info
        .info
        .get("output_cost_per_image")
        .and_then(py_real)
        .unwrap_or(0.0);
    let images = image_response
        .get("data")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    Ok(rate * images as f64)
}
