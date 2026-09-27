use crate::error::CostError;
use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::wire::lax_count;

use crate::responses_usage::{
    CacheCreationTokenDetails, ChatUsage, CompletionTokenDetails, PromptTokenDetails,
};

pub fn is_anthropic_usage_object(usage: &Value) -> bool {
    usage.as_object().is_some_and(|object| {
        !object.contains_key("prompt_tokens")
            && object.contains_key("input_tokens")
            && (object.contains_key("cache_read_input_tokens")
                || object.contains_key("cache_creation_input_tokens"))
    })
}

fn count(value: Option<&Value>) -> Result<u64, CostError> {
    match value {
        None | Some(Value::Null) => Ok(0),
        Some(Value::Number(number)) => number
            .as_u64()
            .or_else(|| {
                number
                    .as_f64()
                    .filter(|value| value.is_finite() && *value >= 0.0 && *value < u64::MAX as f64)
                    .map(|value| value as u64)
            })
            .ok_or(CostError::InvalidUsage),
        Some(Value::Bool(value)) => Ok(u64::from(*value)),
        _ => Ok(0),
    }
}

fn sum_field(entries: &[&Map<String, Value>], field: &str) -> Result<u64, CostError> {
    entries.iter().try_fold(0_u64, |sum, entry| {
        sum.checked_add(count(entry.get(field))?)
            .ok_or(CostError::TokenCountOverflow)
    })
}

fn cache_creation_details(
    usage: &Map<String, Value>,
    iterations: &[&Map<String, Value>],
    total_creation: u64,
) -> Result<Option<CacheCreationTokenDetails>, CostError> {
    let breakdowns: Vec<_> = iterations
        .iter()
        .filter_map(|entry| entry.get("cache_creation").and_then(Value::as_object))
        .collect();
    if !breakdowns.is_empty() {
        let five = sum_field(&breakdowns, "ephemeral_5m_input_tokens")?;
        let one = sum_field(&breakdowns, "ephemeral_1h_input_tokens")?;
        let undetailed = total_creation.saturating_sub(five.saturating_add(one));
        return Ok(Some(CacheCreationTokenDetails {
            ephemeral_5m_input_tokens: Some(
                five.checked_add(undetailed)
                    .ok_or(CostError::TokenCountOverflow)?,
            ),
            ephemeral_1h_input_tokens: Some(one),
        }));
    }
    let Some(details) = usage.get("cache_creation").and_then(Value::as_object) else {
        return Ok(None);
    };
    let pydantic_count = |field: &str| {
        details
            .get(field)
            .filter(|value| !value.is_null())
            .map(|value| lax_count(value).ok_or(CostError::InvalidUsage))
            .transpose()
    };
    Ok(Some(CacheCreationTokenDetails {
        ephemeral_5m_input_tokens: pydantic_count("ephemeral_5m_input_tokens")?,
        ephemeral_1h_input_tokens: pydantic_count("ephemeral_1h_input_tokens")?,
    }))
}

fn thinking_tokens(usage: &Map<String, Value>) -> Option<u64> {
    usage
        .get("output_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("thinking_tokens"))
        .and_then(lax_count)
}

pub fn transform_anthropic_usage_to_chat_usage(
    usage: &Value,
    estimated_reasoning_tokens: Option<u64>,
    response_has_thinking_block: bool,
) -> Result<ChatUsage, CostError> {
    let object = usage.as_object().ok_or(CostError::InvalidShape)?;
    let iterations: Vec<_> =
        object
            .get("iterations")
            .and_then(Value::as_array)
            .map_or(Ok(Vec::new()), |entries| {
                entries
                    .iter()
                    .map(|entry| entry.as_object().ok_or(CostError::InvalidUsage))
                    .collect()
            })?;
    let fresh = if iterations.is_empty() {
        count(object.get("input_tokens"))?
    } else {
        sum_field(&iterations, "input_tokens")?
    };
    let output = if iterations.is_empty() {
        count(object.get("output_tokens"))?
    } else {
        sum_field(&iterations, "output_tokens")?
    };
    let cache_read = if iterations.is_empty() {
        count(object.get("cache_read_input_tokens"))?
    } else {
        sum_field(&iterations, "cache_read_input_tokens")?
    };
    let cache_creation = if iterations.is_empty() {
        count(object.get("cache_creation_input_tokens"))?
    } else {
        sum_field(&iterations, "cache_creation_input_tokens")?
    };
    let prompt = fresh
        .checked_add(cache_read)
        .and_then(|total| total.checked_add(cache_creation))
        .ok_or(CostError::TokenCountOverflow)?;
    let total = prompt
        .checked_add(output)
        .ok_or(CostError::TokenCountOverflow)?;
    let reported = if iterations.is_empty() {
        thinking_tokens(object)
    } else {
        iterations
            .iter()
            .map(|entry| thinking_tokens(entry))
            .collect::<Option<Vec<_>>>()
            .and_then(|tokens| tokens.into_iter().try_fold(0_u64, u64::checked_add))
            .or_else(|| thinking_tokens(object))
    };
    let reasoning = reported
        .or(estimated_reasoning_tokens)
        .map(|tokens| tokens.min(output))
        .or_else(|| (!response_has_thinking_block).then_some(0));
    let completion_tokens_details = Some(CompletionTokenDetails {
        reasoning_tokens: reasoning,
        text_tokens: reasoning.map(|tokens| output - tokens),
        ..CompletionTokenDetails::default()
    });
    let prompt_tokens_details = Some(PromptTokenDetails {
        cached_tokens: cache_read,
        text_tokens: Some(fresh),
        cache_write_tokens: Some(cache_creation),
        cache_creation_tokens: Some(cache_creation),
        cache_creation_token_details: cache_creation_details(object, &iterations, cache_creation)?,
        ..PromptTokenDetails::default()
    });
    let mut extra: BTreeMap<_, _> = object
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "input_tokens" | "output_tokens"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    extra.insert("cache_read_input_tokens".to_string(), cache_read.into());
    extra.insert(
        "cache_creation_input_tokens".to_string(),
        cache_creation.into(),
    );
    extra.insert("_cache_read_input_tokens".to_string(), cache_read.into());
    extra.insert(
        "_cache_creation_input_tokens".to_string(),
        cache_creation.into(),
    );
    Ok(ChatUsage {
        prompt_tokens: prompt,
        completion_tokens: output,
        total_tokens: total,
        prompt_tokens_details,
        completion_tokens_details,
        cost: None,
        extra,
    })
}
