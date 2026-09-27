use std::collections::BTreeMap;

use jiff::Timestamp;
use serde_json::Value;

use crate::base_rate_selection::uses_inclusive_token_thresholds;
use crate::gemini_cost::cost_per_web_search_request;
use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::generic_input::get_cost_per_unit;
use crate::provider_cache::apply_provider_cache_read_default;
use crate::responses_usage::{ChatUsage, CompletionTokenDetails, PromptTokenDetails};

fn token_detail(details: Option<&Value>, key: &str) -> Option<u64> {
    details?.get(key)?.as_u64()
}

pub fn prices_tokens(model_info: &Value) -> bool {
    [
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_image_token",
        "output_cost_per_image_token",
    ]
    .iter()
    .any(|key| model_info.get(key).is_some_and(|value| !value.is_null()))
}

pub fn calculate_image_response_cost_from_usage(
    image_response: &Value,
    model_info: &Value,
    provider: Option<&str>,
    at: Timestamp,
) -> Option<f64> {
    let source = image_response.get("usage")?;
    let input_tokens = source.get("input_tokens")?.as_u64()?;
    let output_tokens = source.get("output_tokens")?.as_u64()?;
    let total_tokens = source.get("total_tokens")?.as_u64()?;
    if input_tokens == 0 && output_tokens == 0 && total_tokens == 0 {
        return None;
    }
    if !prices_tokens(model_info) {
        return None;
    }
    let input_details = source
        .get("input_tokens_details")
        .filter(|value| !value.is_null());
    let output_details = source
        .get("completion_tokens_details")
        .filter(|value| !value.is_null())
        .or_else(|| {
            source
                .get("output_tokens_details")
                .filter(|value| !value.is_null())
        });
    let output = output_details.map_or_else(
        || CompletionTokenDetails {
            text_tokens: Some(0),
            image_tokens: Some(output_tokens),
            reasoning_tokens: Some(0),
            audio_tokens: Some(0),
            video_tokens: None,
        },
        |details| {
            let text_tokens = token_detail(Some(details), "text_tokens").unwrap_or(0);
            let image_tokens = token_detail(Some(details), "image_tokens").unwrap_or(0);
            let audio_tokens = token_detail(Some(details), "audio_tokens").unwrap_or(0);
            let reasoning_tokens = token_detail(Some(details), "reasoning_tokens").unwrap_or(0);
            let known = text_tokens
                .saturating_add(image_tokens)
                .saturating_add(audio_tokens)
                .saturating_add(reasoning_tokens);
            CompletionTokenDetails {
                text_tokens: Some(text_tokens.saturating_add(output_tokens.saturating_sub(known))),
                image_tokens: Some(image_tokens),
                reasoning_tokens: Some(reasoning_tokens),
                audio_tokens: Some(audio_tokens),
                video_tokens: None,
            }
        },
    );
    let usage = ChatUsage {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        total_tokens,
        prompt_tokens_details: input_details.map(|details| PromptTokenDetails {
            text_tokens: token_detail(Some(details), "text_tokens"),
            image_tokens: token_detail(Some(details), "image_tokens"),
            ..PromptTokenDetails::default()
        }),
        completion_tokens_details: Some(output),
        cost: None,
        extra: BTreeMap::new(),
    };
    let model_info = apply_provider_cache_read_default(model_info, provider);
    let (input, output) = calculate_generic_cost_from_model_info_with_region(
        &usage,
        &model_info,
        None,
        uses_inclusive_token_thresholds(provider),
        None,
        None,
        at,
    );
    Some(input + output)
}

pub fn flat_image_cost(image_response: &Value, model_info: &Value) -> f64 {
    let count = image_response
        .get("data")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    count as f64 * get_cost_per_unit(model_info, "output_cost_per_image", Some(0.0)).unwrap_or(0.0)
}

pub fn resolve_image_model_info(shared: Option<&Value>, supplied: Option<&Value>) -> Option<Value> {
    match (shared, supplied) {
        (None, None) => None,
        (Some(info), None) | (None, Some(info)) => Some(info.clone()),
        (Some(shared), Some(supplied)) => match (shared.as_object(), supplied.as_object()) {
            (Some(shared), Some(supplied)) => Some(Value::Object(
                shared
                    .iter()
                    .chain(supplied.iter())
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            )),
            _ => Some(supplied.clone()),
        },
    }
}

pub fn calculate_image_response_web_search_cost(
    image_response: &Value,
    model_info: &Value,
    provider: &str,
) -> f64 {
    if !matches!(provider, "gemini" | "vertex_ai") {
        return 0.0;
    }
    let requests = image_response
        .pointer("/usage/web_search_requests")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if requests == 0 {
        return 0.0;
    }
    let usage = ChatUsage {
        prompt_tokens_details: Some(PromptTokenDetails {
            web_search_requests: Some(requests),
            ..PromptTokenDetails::default()
        }),
        ..ChatUsage::default()
    };
    cost_per_web_search_request(&usage, model_info)
}

fn google_image_generation_cost(
    image_response: &Value,
    model_info: &Value,
    provider: &str,
    at: Timestamp,
) -> f64 {
    let search = calculate_image_response_web_search_cost(image_response, model_info, provider);
    let image =
        calculate_image_response_cost_from_usage(image_response, model_info, Some(provider), at)
            .unwrap_or_else(|| flat_image_cost(image_response, model_info));
    image + search
}

pub fn vertex_image_generation_cost(
    image_response: &Value,
    model_info: &Value,
    at: Timestamp,
) -> f64 {
    google_image_generation_cost(image_response, model_info, "vertex_ai", at)
}

pub fn gemini_image_generation_cost(
    image_response: &Value,
    model_info: &Value,
    at: Timestamp,
) -> f64 {
    google_image_generation_cost(image_response, model_info, "gemini", at)
}

pub fn gemini_image_edit_cost(image_response: &Value, model_info: &Value, at: Timestamp) -> f64 {
    gemini_image_generation_cost(image_response, model_info, at)
}

pub fn vertex_image_edit_cost(image_response: &Value, model_info: &Value) -> f64 {
    flat_image_cost(image_response, model_info)
}
