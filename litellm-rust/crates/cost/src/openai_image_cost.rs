use crate::error::CostError;
use jiff::Timestamp;
use serde_json::Value;

use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::image_response_cost::{calculate_image_response_cost_from_usage, flat_image_cost};

use crate::usage_dispatch::chat_usage;

fn chat_token_cost(usage: &Value, model_info: &Value, at: Timestamp) -> Result<f64, CostError> {
    let usage = chat_usage(usage)?;
    let (input, output) = calculate_generic_cost_from_model_info_with_region(
        &usage, model_info, None, false, None, None, at,
    );
    Ok(input + output)
}

pub fn cost_calculator(
    image_response: &Value,
    model_info: &Value,
    provider: &str,
    at: Timestamp,
) -> Result<f64, CostError> {
    let Some(usage) = image_response.get("usage").filter(|usage| !usage.is_null()) else {
        return Ok(flat_image_cost(image_response, model_info));
    };
    if usage.get("prompt_tokens").is_some()
        && usage
            .get("completion_tokens_details")
            .is_some_and(|details| !details.is_null())
    {
        return chat_token_cost(usage, model_info, at);
    }
    if usage.get("input_tokens").is_some()
        && let Some(cost) =
            calculate_image_response_cost_from_usage(image_response, model_info, Some(provider), at)
    {
        return Ok(cost);
    }
    if usage.get("prompt_tokens").is_some() {
        return chat_token_cost(usage, model_info, at);
    }
    Ok(flat_image_cost(image_response, model_info))
}
