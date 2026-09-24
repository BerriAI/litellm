use crate::error::CostError;
use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use crate::wire::{deserialize_lax_count, deserialize_lax_count_option, lax_count};

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct CachedTokenDetails {
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub text_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub audio_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub image_tokens: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct CacheCreationTokenDetails {
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub ephemeral_5m_input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct PromptTokenDetails {
    #[serde(default, deserialize_with = "deserialize_lax_count")]
    pub cached_tokens: u64,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub audio_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub text_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub image_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub video_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub cache_write_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub cache_creation_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_creation_token_details: Option<CacheCreationTokenDetails>,
    pub cached_tokens_details: Option<CachedTokenDetails>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub web_search_requests: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub google_maps_grounding_requests: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub character_count: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub image_count: Option<u64>,
    pub video_length_seconds: Option<f64>,
    pub audio_length_seconds: Option<f64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub query_count: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct CompletionTokenDetails {
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub reasoning_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub text_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub image_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub audio_tokens: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_lax_count_option")]
    pub video_tokens: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: Option<PromptTokenDetails>,
    pub completion_tokens_details: Option<CompletionTokenDetails>,
    pub cost: Option<f64>,
    pub extra: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct RawResponseUsage {
    input_tokens: Value,
    output_tokens: Value,
    #[serde(default)]
    total_tokens: Option<Value>,
    #[serde(default)]
    input_tokens_details: Option<PromptTokenDetails>,
    #[serde(default)]
    output_tokens_details: Option<CompletionTokenDetails>,
    #[serde(default)]
    input_token_details: Option<PromptTokenDetails>,
    #[serde(default)]
    output_token_details: Option<CompletionTokenDetails>,
    #[serde(default)]
    cost: Option<Value>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

pub fn is_response_api_usage(usage: &Value) -> bool {
    usage.as_object().is_some_and(|object| {
        object.contains_key("input_tokens") && object.contains_key("output_tokens")
    })
}

pub fn text_tokens_without_nested_reasoning(
    completion_tokens: u64,
    text_tokens: u64,
    reasoning_tokens: u64,
    other_modality_tokens: u64,
) -> u64 {
    let reported_total =
        text_tokens as u128 + reasoning_tokens as u128 + other_modality_tokens as u128;
    let nested = (reasoning_tokens as u128)
        .min(text_tokens as u128)
        .min(reported_total.saturating_sub(completion_tokens as u128));
    text_tokens - nested as u64
}

pub fn transform_response_api_usage_to_chat_usage(usage: &Value) -> Result<ChatUsage, CostError> {
    if !is_response_api_usage(usage) {
        return Err(CostError::InvalidShape);
    }
    let raw: RawResponseUsage =
        serde_json::from_value(usage.clone()).map_err(|_| CostError::InvalidUsage)?;
    let is_python_int = |value: &Value| value.is_i64() || value.is_u64() || value.is_boolean();
    let total_is_derivable = is_python_int(&raw.input_tokens) && is_python_int(&raw.output_tokens);
    if raw.total_tokens.as_ref().is_none_or(Value::is_null) && !total_is_derivable {
        return Err(CostError::InvalidUsage);
    }
    let input_tokens = lax_count(&raw.input_tokens).ok_or(CostError::InvalidUsage)?;
    let output_tokens = lax_count(&raw.output_tokens).ok_or(CostError::InvalidUsage)?;
    let total_tokens = input_tokens
        .checked_add(output_tokens)
        .ok_or(CostError::TokenCountOverflow)?;
    let prompt_tokens_details =
        raw.input_tokens_details
            .or(raw.input_token_details)
            .map(|details| PromptTokenDetails {
                cached_tokens: details.cached_tokens,
                audio_tokens: details.audio_tokens,
                text_tokens: details.text_tokens,
                image_tokens: details.image_tokens,
                cached_tokens_details: details.cached_tokens_details,
                video_tokens: details.video_tokens,
                cache_write_tokens: details.cache_write_tokens,
                cache_creation_tokens: details.cache_write_tokens,
                web_search_requests: details.web_search_requests,
                google_maps_grounding_requests: details.google_maps_grounding_requests,
                ..PromptTokenDetails::default()
            });
    let completion_tokens_details =
        raw.output_tokens_details
            .or(raw.output_token_details)
            .map(|details| CompletionTokenDetails {
                reasoning_tokens: details.reasoning_tokens,
                image_tokens: details.image_tokens,
                text_tokens: details.text_tokens.map(|text| {
                    text_tokens_without_nested_reasoning(
                        output_tokens,
                        text,
                        details.reasoning_tokens.unwrap_or(0),
                        details
                            .audio_tokens
                            .unwrap_or(0)
                            .saturating_add(details.image_tokens.unwrap_or(0)),
                    )
                }),
                audio_tokens: details.audio_tokens,
                video_tokens: None,
            });
    let extra = raw
        .extra
        .into_iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "prompt_tokens"
                    | "completion_tokens"
                    | "total_tokens"
                    | "prompt_tokens_details"
                    | "completion_tokens_details"
            )
        })
        .collect();
    Ok(ChatUsage {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        total_tokens,
        prompt_tokens_details,
        completion_tokens_details,
        cost: reported_cost(raw.cost.as_ref()),
        extra,
    })
}

pub(crate) fn reported_cost(value: Option<&Value>) -> Option<f64> {
    let value = value?;
    value.as_f64().or_else(|| value.get("total_cost")?.as_f64())
}
