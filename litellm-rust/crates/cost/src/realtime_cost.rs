use crate::error::CostError;
use serde_json::{Value, json};

use crate::responses_usage::{
    CacheCreationTokenDetails, CachedTokenDetails, ChatUsage, CompletionTokenDetails,
    PromptTokenDetails,
};
use crate::usage_dispatch::get_usage_object;

fn add_optional(first: Option<u64>, second: Option<u64>) -> Result<Option<u64>, CostError> {
    match (first, second) {
        (None, None) => Ok(None),
        (first, second) => first
            .unwrap_or(0)
            .checked_add(second.unwrap_or(0))
            .map(Some)
            .ok_or(CostError::TokenCountOverflow),
    }
}

fn add_optional_f64(first: Option<f64>, second: Option<f64>) -> Option<f64> {
    (first.is_some() || second.is_some()).then(|| first.unwrap_or(0.0) + second.unwrap_or(0.0))
}

fn combine_cached_details(
    first: Option<CachedTokenDetails>,
    second: Option<CachedTokenDetails>,
) -> Result<Option<CachedTokenDetails>, CostError> {
    match (first, second) {
        (None, None) => Ok(None),
        (first, second) => {
            let first = first.unwrap_or_default();
            let second = second.unwrap_or_default();
            Ok(Some(CachedTokenDetails {
                text_tokens: add_optional(first.text_tokens, second.text_tokens)?,
                audio_tokens: add_optional(first.audio_tokens, second.audio_tokens)?,
                image_tokens: add_optional(first.image_tokens, second.image_tokens)?,
            }))
        }
    }
}

fn combine_cache_creation_details(
    first: Option<CacheCreationTokenDetails>,
    second: Option<CacheCreationTokenDetails>,
) -> Result<Option<CacheCreationTokenDetails>, CostError> {
    match (first, second) {
        (None, None) => Ok(None),
        (first, second) => {
            let first = first.unwrap_or_default();
            let second = second.unwrap_or_default();
            Ok(Some(CacheCreationTokenDetails {
                ephemeral_5m_input_tokens: add_optional(
                    first.ephemeral_5m_input_tokens,
                    second.ephemeral_5m_input_tokens,
                )?,
                ephemeral_1h_input_tokens: add_optional(
                    first.ephemeral_1h_input_tokens,
                    second.ephemeral_1h_input_tokens,
                )?,
            }))
        }
    }
}

fn combine_prompt_details(
    first: Option<PromptTokenDetails>,
    second: Option<PromptTokenDetails>,
) -> Result<Option<PromptTokenDetails>, CostError> {
    match (first, second) {
        (None, None) => Ok(None),
        (first, second) => {
            let first = first.unwrap_or_default();
            let second = second.unwrap_or_default();
            Ok(Some(PromptTokenDetails {
                cached_tokens: first
                    .cached_tokens
                    .checked_add(second.cached_tokens)
                    .ok_or(CostError::TokenCountOverflow)?,
                audio_tokens: add_optional(first.audio_tokens, second.audio_tokens)?,
                text_tokens: add_optional(first.text_tokens, second.text_tokens)?,
                image_tokens: add_optional(first.image_tokens, second.image_tokens)?,
                video_tokens: add_optional(first.video_tokens, second.video_tokens)?,
                cache_write_tokens: add_optional(
                    first.cache_write_tokens,
                    second.cache_write_tokens,
                )?,
                cache_creation_tokens: add_optional(
                    first.cache_creation_tokens,
                    second.cache_creation_tokens,
                )?,
                cache_creation_input_tokens: add_optional(
                    first.cache_creation_input_tokens,
                    second.cache_creation_input_tokens,
                )?,
                cache_creation_token_details: combine_cache_creation_details(
                    first.cache_creation_token_details,
                    second.cache_creation_token_details,
                )?,
                cached_tokens_details: combine_cached_details(
                    first.cached_tokens_details,
                    second.cached_tokens_details,
                )?,
                web_search_requests: add_optional(
                    first.web_search_requests,
                    second.web_search_requests,
                )?,
                google_maps_grounding_requests: add_optional(
                    first.google_maps_grounding_requests,
                    second.google_maps_grounding_requests,
                )?,
                character_count: add_optional(first.character_count, second.character_count)?,
                image_count: add_optional(first.image_count, second.image_count)?,
                video_length_seconds: add_optional_f64(
                    first.video_length_seconds,
                    second.video_length_seconds,
                ),
                audio_length_seconds: add_optional_f64(
                    first.audio_length_seconds,
                    second.audio_length_seconds,
                ),
                query_count: add_optional(first.query_count, second.query_count)?,
            }))
        }
    }
}

fn combine_completion_details(
    first: Option<CompletionTokenDetails>,
    second: Option<CompletionTokenDetails>,
) -> Result<Option<CompletionTokenDetails>, CostError> {
    match (first, second) {
        (None, None) => Ok(None),
        (first, second) => {
            let first = first.unwrap_or_default();
            let second = second.unwrap_or_default();
            Ok(Some(CompletionTokenDetails {
                reasoning_tokens: add_optional(first.reasoning_tokens, second.reasoning_tokens)?,
                text_tokens: add_optional(first.text_tokens, second.text_tokens)?,
                image_tokens: add_optional(first.image_tokens, second.image_tokens)?,
                audio_tokens: add_optional(first.audio_tokens, second.audio_tokens)?,
                video_tokens: add_optional(first.video_tokens, second.video_tokens)?,
            }))
        }
    }
}

pub fn combine_usage_objects(
    usages: impl IntoIterator<Item = ChatUsage>,
) -> Result<ChatUsage, CostError> {
    usages
        .into_iter()
        .try_fold(ChatUsage::default(), |combined, usage| {
            Ok(ChatUsage {
                prompt_tokens: combined
                    .prompt_tokens
                    .checked_add(usage.prompt_tokens)
                    .ok_or(CostError::TokenCountOverflow)?,
                completion_tokens: combined
                    .completion_tokens
                    .checked_add(usage.completion_tokens)
                    .ok_or(CostError::TokenCountOverflow)?,
                total_tokens: combined
                    .total_tokens
                    .checked_add(usage.total_tokens)
                    .ok_or(CostError::TokenCountOverflow)?,
                prompt_tokens_details: combine_prompt_details(
                    combined.prompt_tokens_details,
                    usage.prompt_tokens_details,
                )?,
                completion_tokens_details: combine_completion_details(
                    combined.completion_tokens_details,
                    usage.completion_tokens_details,
                )?,
                cost: add_optional_f64(combined.cost, usage.cost),
                extra: Default::default(),
            })
        })
}

pub(crate) fn event_usage(event: &Value) -> Result<ChatUsage, CostError> {
    let usage = event.pointer("/response/usage").unwrap_or(&Value::Null);
    Ok(get_usage_object(&json!({"usage": usage}))?.unwrap_or_default())
}

pub fn collect_usage_from_realtime_stream_results(
    results: &[Value],
) -> Result<Vec<ChatUsage>, CostError> {
    results
        .iter()
        .filter(|event| event.get("type").and_then(Value::as_str) == Some("response.done"))
        .map(event_usage)
        .collect()
}

pub fn collect_and_combine_usage_from_realtime_stream_results(
    results: &[Value],
) -> Result<ChatUsage, CostError> {
    combine_usage_objects(collect_usage_from_realtime_stream_results(results)?)
}

pub fn billable_responses_ws_events(results: &[Value]) -> Vec<&Value> {
    results
        .iter()
        .filter(|event| {
            matches!(
                event.get("type").and_then(Value::as_str),
                Some("response.completed" | "response.incomplete")
            ) && event
                .pointer("/response/usage")
                .is_some_and(|usage| !usage.is_null())
        })
        .collect()
}

pub fn collect_usage_from_responses_ws_results(
    results: &[Value],
) -> Result<Vec<ChatUsage>, CostError> {
    billable_responses_ws_events(results)
        .into_iter()
        .map(event_usage)
        .collect()
}

pub fn collect_and_combine_usage_from_responses_ws_results(
    results: &[Value],
) -> Result<ChatUsage, CostError> {
    combine_usage_objects(collect_usage_from_responses_ws_results(results)?)
}

pub fn partition_results_by_service_tier(results: &[Value]) -> Vec<(Option<&str>, Vec<&Value>)> {
    let billable = billable_responses_ws_events(results);
    let tiers = billable
        .iter()
        .map(|event| {
            event
                .pointer("/response/service_tier")
                .and_then(Value::as_str)
        })
        .collect::<Vec<_>>();
    tiers
        .into_iter()
        .enumerate()
        .filter(|(index, tier)| {
            !billable[..*index].iter().any(|event| {
                event
                    .pointer("/response/service_tier")
                    .and_then(Value::as_str)
                    == *tier
            })
        })
        .map(|(_, tier)| {
            let group = billable
                .iter()
                .copied()
                .filter(|event| {
                    event
                        .pointer("/response/service_tier")
                        .and_then(Value::as_str)
                        == tier
                })
                .collect();
            (tier, group)
        })
        .collect()
}

pub fn get_transcription_model_name_from_results(results: &[Value]) -> Option<&str> {
    results.iter().find_map(|event| {
        let event_type = event.get("type")?.as_str()?;
        if !matches!(
            event_type,
            "transcription_session.created"
                | "transcription_session.updated"
                | "session.created"
                | "session.updated"
        ) {
            return None;
        }
        let session = event.get("session")?;
        let nested_transcription = session.pointer("/audio/input/transcription");
        let transcription = nested_transcription
            .filter(|value| value.as_object().is_some_and(|object| !object.is_empty()))
            .or_else(|| session.get("input_audio_transcription"));
        [
            transcription.and_then(|value| value.get("model")),
            session.get("model"),
        ]
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|model| !model.is_empty())
    })
}

pub fn transcription_usage_cost(usage: &Value, model_info: Option<&Value>) -> f64 {
    let Some(model_info) = model_info else {
        return 0.0;
    };
    let number = |value: Option<&Value>| value.and_then(Value::as_f64).unwrap_or(0.0);
    match usage.get("type").and_then(Value::as_str) {
        Some("duration") => {
            number(usage.get("seconds")) * number(model_info.get("input_cost_per_second"))
        }
        Some("tokens") => {
            let audio_tokens = number(usage.pointer("/input_token_details/audio_tokens"));
            let text_tokens = number(usage.pointer("/input_token_details/text_tokens"));
            let output_tokens = number(usage.get("output_tokens"));
            let input_rate = number(model_info.get("input_cost_per_token"));
            let audio_rate = model_info
                .get("input_cost_per_audio_token")
                .and_then(Value::as_f64)
                .filter(|rate| *rate != 0.0)
                .unwrap_or(input_rate);
            let output_rate = number(model_info.get("output_cost_per_token"));
            audio_tokens * audio_rate + text_tokens * input_rate + output_tokens * output_rate
        }
        _ => 0.0,
    }
}
