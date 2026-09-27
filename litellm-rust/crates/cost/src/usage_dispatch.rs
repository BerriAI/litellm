use crate::error::CostError;
use std::collections::BTreeMap;

use serde_json::Value;

use crate::wire::lax_count;

use crate::anthropic_usage::{is_anthropic_usage_object, transform_anthropic_usage_to_chat_usage};
use crate::interactions_usage::{
    is_interactions_usage_object, transform_interactions_usage_object,
};
use crate::responses_usage::{
    ChatUsage, CompletionTokenDetails, PromptTokenDetails, is_response_api_usage,
    transform_response_api_usage_to_chat_usage,
};
use crate::transcription_usage::{
    is_transcription_usage_object, transform_transcription_usage_object,
};

fn int_count(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_bool().map(u64::from))
}

pub fn chat_usage(raw: &Value) -> Result<ChatUsage, CostError> {
    let object = raw.as_object().ok_or(CostError::InvalidShape)?;
    let count = |name: &str| {
        object
            .get(name)
            .filter(|value| !value.is_null())
            .map_or(Ok(0), |value| {
                lax_count(value).ok_or(CostError::InvalidUsage)
            })
    };
    let prompt_tokens = count("prompt_tokens")?;
    let completion_tokens = count("completion_tokens")?;
    let total_tokens = count("total_tokens")?;
    let prompt_cache_hit = object.get("prompt_cache_hit_tokens").and_then(int_count);
    let read = object
        .get("cache_read_input_tokens")
        .and_then(int_count)
        .or(prompt_cache_hit);
    let creation = object
        .get("cache_creation_input_tokens")
        .and_then(int_count);
    let private_read =
        prompt_cache_hit.or(object.get("cache_read_input_tokens").and_then(int_count));
    let prompt_details: Option<PromptTokenDetails> = object
        .get("prompt_tokens_details")
        .filter(|value| value.as_object().is_some_and(|details| !details.is_empty()))
        .map(|value| serde_json::from_value(value.clone()).map_err(|_| CostError::InvalidUsage))
        .transpose()?;
    let prompt_tokens_details = match (prompt_details, read, creation) {
        (None, None, None) => None,
        (details, read, creation) => {
            let details = details.unwrap_or_default();
            let cache_write_tokens = creation
                .or(details.cache_write_tokens)
                .or(details.cache_creation_tokens)
                .or(details.cache_creation_input_tokens);
            Some(PromptTokenDetails {
                cached_tokens: read.unwrap_or(details.cached_tokens),
                cache_write_tokens,
                cache_creation_tokens: cache_write_tokens,
                ..details
            })
        }
    };
    let completion_tokens_details: Option<CompletionTokenDetails> = object
        .get("completion_tokens_details")
        .filter(|value| value.as_object().is_some_and(|details| !details.is_empty()))
        .map(|value| serde_json::from_value(value.clone()).map_err(|_| CostError::InvalidUsage))
        .transpose()?;
    let reasoning = object.get("reasoning_tokens").and_then(lax_count);
    let completion_tokens_details = match (completion_tokens_details, reasoning) {
        (None, None | Some(0)) => None,
        (details, reasoning) => {
            let details = details.unwrap_or_default();
            let reported_reasoning = details.reasoning_tokens.or(reasoning);
            let text_tokens = details.text_tokens.or_else(|| {
                reasoning.map(|tokens| {
                    completion_tokens
                        .saturating_sub(tokens)
                        .saturating_sub(details.image_tokens.unwrap_or(0))
                        .saturating_sub(details.audio_tokens.unwrap_or(0))
                })
            });
            Some(CompletionTokenDetails {
                reasoning_tokens: reported_reasoning,
                text_tokens,
                ..details
            })
        }
    };
    let mut extra: BTreeMap<_, _> = object
        .iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "prompt_tokens"
                    | "completion_tokens"
                    | "total_tokens"
                    | "prompt_tokens_details"
                    | "completion_tokens_details"
                    | "cost"
                    | "reasoning_tokens"
            )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    if let Some(private_read) = private_read {
        extra.insert("_cache_read_input_tokens".to_string(), private_read.into());
    }
    if let Some(creation) = creation {
        extra.insert("_cache_creation_input_tokens".to_string(), creation.into());
    }
    Ok(ChatUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        prompt_tokens_details,
        completion_tokens_details,
        cost: crate::responses_usage::reported_cost(object.get("cost")),
        extra,
    })
}

pub fn get_usage_object(response: &Value) -> Result<Option<ChatUsage>, CostError> {
    let Some(usage) = response.as_object().and_then(|object| object.get("usage")) else {
        return Ok(None);
    };
    if usage.is_null() {
        return Ok(None);
    }
    if is_anthropic_usage_object(usage) {
        return Ok(Some(transform_anthropic_usage_to_chat_usage(
            usage, None, false,
        )?));
    }
    if is_response_api_usage(usage) {
        return Ok(Some(transform_response_api_usage_to_chat_usage(usage)?));
    }
    if is_transcription_usage_object(usage) {
        return transform_transcription_usage_object(usage);
    }
    if is_interactions_usage_object(usage) {
        return Ok(Some(transform_interactions_usage_object(usage)?));
    }
    if usage.is_object() {
        return Ok(Some(chat_usage(usage)?));
    }
    Err(CostError::InvalidShape)
}
