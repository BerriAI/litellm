use std::collections::BTreeMap;

use jiff::Timestamp;
use serde_json::Value;

use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::provider_cache::apply_provider_cache_read_default;
use crate::responses_usage::{ChatUsage, CompletionTokenDetails, PromptTokenDetails};

fn token_detail(details: Option<&Value>, key: &str) -> Option<u64> {
    details?.get(key)?.as_u64()
}

fn prices_tokens(model_info: &Value) -> bool {
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
        provider == Some("xai"),
        None,
        None,
        at,
    );
    Some(input + output)
}
