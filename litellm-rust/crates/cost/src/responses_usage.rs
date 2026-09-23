use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use crate::{PromptConvention, Usage};

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct CachedTokenDetails {
    pub text_tokens: Option<u64>,
    pub audio_tokens: Option<u64>,
    pub image_tokens: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct CacheCreationTokenDetails {
    pub ephemeral_5m_input_tokens: Option<u64>,
    pub ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct PromptTokenDetails {
    #[serde(default)]
    pub cached_tokens: u64,
    pub audio_tokens: Option<u64>,
    pub text_tokens: Option<u64>,
    pub image_tokens: Option<u64>,
    pub video_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    pub cache_creation_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_creation_token_details: Option<CacheCreationTokenDetails>,
    pub cached_tokens_details: Option<CachedTokenDetails>,
    pub web_search_requests: Option<u64>,
    pub google_maps_grounding_requests: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct CompletionTokenDetails {
    pub reasoning_tokens: Option<u64>,
    pub text_tokens: Option<u64>,
    pub image_tokens: Option<u64>,
    pub audio_tokens: Option<u64>,
    pub video_tokens: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: Option<PromptTokenDetails>,
    pub completion_tokens_details: Option<CompletionTokenDetails>,
    pub cost: Option<f64>,
    pub extra: BTreeMap<String, Value>,
}

impl ChatUsage {
    pub fn token_usage(&self) -> Usage {
        let cache_creation_details = self
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cache_creation_token_details.as_ref());
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            cache_read_tokens: self
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |details| details.cached_tokens),
            cache_write_tokens: self
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cache_write_tokens)
                .unwrap_or(0),
            cache_write_5m_tokens: cache_creation_details
                .map(|details| details.ephemeral_5m_input_tokens.unwrap_or(0)),
            cache_write_1h_tokens: cache_creation_details
                .map(|details| details.ephemeral_1h_input_tokens.unwrap_or(0)),
            prompt_convention: PromptConvention::IncludesCache,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UsageError {
    InvalidShape,
    InvalidUsage,
    TokenCountOverflow,
}

#[derive(Deserialize)]
struct RawResponseUsage {
    input_tokens: u64,
    output_tokens: u64,
    #[serde(default)]
    input_tokens_details: Option<PromptTokenDetails>,
    #[serde(default)]
    output_tokens_details: Option<CompletionTokenDetails>,
    #[serde(default)]
    input_token_details: Option<PromptTokenDetails>,
    #[serde(default)]
    output_token_details: Option<CompletionTokenDetails>,
    #[serde(default)]
    cost: Option<f64>,
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

pub fn transform_response_api_usage_to_chat_usage(usage: &Value) -> Result<ChatUsage, UsageError> {
    if !is_response_api_usage(usage) {
        return Err(UsageError::InvalidShape);
    }
    let raw: RawResponseUsage =
        serde_json::from_value(usage.clone()).map_err(|_| UsageError::InvalidUsage)?;
    let total_tokens = raw
        .input_tokens
        .checked_add(raw.output_tokens)
        .ok_or(UsageError::TokenCountOverflow)?;
    let prompt_tokens_details =
        raw.input_tokens_details
            .or(raw.input_token_details)
            .map(|details| {
                let creation = details
                    .cache_write_tokens
                    .or(details.cache_creation_tokens)
                    .or(details.cache_creation_input_tokens);
                PromptTokenDetails {
                    cache_write_tokens: creation,
                    cache_creation_tokens: creation,
                    ..details
                }
            });
    let completion_tokens_details =
        raw.output_tokens_details
            .or(raw.output_token_details)
            .map(|details| {
                let text_tokens = details.text_tokens.map(|text| {
                    text_tokens_without_nested_reasoning(
                        raw.output_tokens,
                        text,
                        details.reasoning_tokens.unwrap_or(0),
                        details
                            .audio_tokens
                            .unwrap_or(0)
                            .saturating_add(details.image_tokens.unwrap_or(0)),
                    )
                });
                CompletionTokenDetails {
                    text_tokens,
                    ..details
                }
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
        prompt_tokens: raw.input_tokens,
        completion_tokens: raw.output_tokens,
        total_tokens,
        prompt_tokens_details,
        completion_tokens_details,
        cost: raw.cost,
        extra,
    })
}
