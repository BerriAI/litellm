use std::collections::HashMap;

use serde_json::Value;

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

pub fn stability1_pricing_key(model: &str, size: Option<&str>, optional_params: &Value) -> String {
    let steps = optional_params
        .get("steps")
        .and_then(Value::as_f64)
        .unwrap_or(50.0);
    let tier = if steps > 50.0 {
        "max-steps"
    } else {
        "50-steps"
    };
    format!("{}/{tier}/{model}", size.unwrap_or("1024-x-1024"))
}

fn model_info<'a>(
    entries: &'a HashMap<String, Value>,
    model: &str,
    family: BedrockImageFamily,
    size: Option<&str>,
    optional_params: &Value,
) -> Option<&'a Value> {
    let bare = model.strip_prefix("bedrock/").unwrap_or(model);
    match family {
        BedrockImageFamily::Stability1 => {
            let key = stability1_pricing_key(model, size, optional_params);
            let bare_key = stability1_pricing_key(bare, size, optional_params);
            [key, bare_key]
                .into_iter()
                .find_map(|candidate| entries.get(&candidate))
        }
        _ => [model.to_owned(), format!("bedrock/{bare}"), bare.to_owned()]
            .into_iter()
            .find_map(|candidate| entries.get(&candidate)),
    }
}

pub fn cost_calculator(
    model: &str,
    image_response: &Value,
    size: Option<&str>,
    optional_params: &Value,
    entries: &HashMap<String, Value>,
) -> Option<f64> {
    let family = get_config_class(model);
    let info = model_info(entries, model, family, size, optional_params)?;
    let rate = info
        .get("output_cost_per_image")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let images = image_response
        .get("data")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    Some(rate * images as f64)
}
