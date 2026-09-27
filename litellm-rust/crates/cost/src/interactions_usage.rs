use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::error::CostError;
use crate::responses_usage::{ChatUsage, CompletionTokenDetails, PromptTokenDetails};

#[derive(Clone, Copy, Debug, Default)]
struct Modalities {
    text: Option<u64>,
    audio: Option<u64>,
    image: Option<u64>,
    video: Option<u64>,
}

impl Modalities {
    fn any(self) -> bool {
        self.text.is_some() || self.audio.is_some() || self.image.is_some() || self.video.is_some()
    }

    fn subtract(self, cached: Self, total_cached: u64) -> Self {
        if cached.any() {
            return Self {
                text: self
                    .text
                    .map(|value| value.saturating_sub(cached.text.unwrap_or(0))),
                audio: self
                    .audio
                    .map(|value| value.saturating_sub(cached.audio.unwrap_or(0))),
                image: self
                    .image
                    .map(|value| value.saturating_sub(cached.image.unwrap_or(0))),
                video: self
                    .video
                    .map(|value| value.saturating_sub(cached.video.unwrap_or(0))),
            };
        }
        Self {
            text: self.text.map(|value| value.saturating_sub(total_cached)),
            ..self
        }
    }
}

pub fn is_interactions_usage_object(usage: &Value) -> bool {
    usage.as_object().is_some_and(|object| {
        !object.contains_key("prompt_tokens")
            && !object.contains_key("input_tokens")
            && (object.contains_key("total_input_tokens")
                || object.contains_key("total_output_tokens"))
    })
}

fn count(value: Option<&Value>) -> u64 {
    match value {
        Some(Value::Bool(value)) => u64::from(*value),
        Some(value) => value.as_u64().unwrap_or(0),
        None => 0,
    }
}

fn entries<'a>(usage: &'a Map<String, Value>, key: &str) -> Vec<&'a Map<String, Value>> {
    usage
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .collect()
}

fn add(existing: Option<u64>, value: u64) -> Result<Option<u64>, CostError> {
    existing
        .unwrap_or(0)
        .checked_add(value)
        .map(Some)
        .ok_or(CostError::TokenCountOverflow)
}

fn sums<'a>(
    entries: impl IntoIterator<Item = &'a Map<String, Value>>,
) -> Result<Modalities, CostError> {
    entries
        .into_iter()
        .try_fold(Modalities::default(), |counts, entry| {
            let tokens = count(entry.get("tokens"));
            let modality = entry
                .get("modality")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_ascii_lowercase();
            match modality.as_str() {
                "text" | "document" => Ok(Modalities {
                    text: add(counts.text, tokens)?,
                    ..counts
                }),
                "audio" => Ok(Modalities {
                    audio: add(counts.audio, tokens)?,
                    ..counts
                }),
                "image" => Ok(Modalities {
                    image: add(counts.image, tokens)?,
                    ..counts
                }),
                "video" => Ok(Modalities {
                    video: add(counts.video, tokens)?,
                    ..counts
                }),
                _ => Ok(counts),
            }
        })
}

pub fn transform_interactions_usage_object(usage: &Value) -> Result<ChatUsage, CostError> {
    if !is_interactions_usage_object(usage) {
        return Err(CostError::InvalidShape);
    }
    let object = usage.as_object().ok_or(CostError::InvalidShape)?;
    let input = sums(
        entries(object, "input_tokens_by_modality")
            .into_iter()
            .chain(entries(object, "tool_use_tokens_by_modality")),
    )?;
    let cached = sums(entries(object, "cached_tokens_by_modality"))?;
    let output = sums(entries(object, "output_tokens_by_modality"))?;
    let total_cached = count(object.get("total_cached_tokens"));
    let input = input.subtract(cached, total_cached);
    let reported_reasoning = count(object.get("total_reasoning_tokens"));
    let reasoning = if reported_reasoning > 0 {
        reported_reasoning
    } else {
        count(object.get("total_thought_tokens"))
    };
    let prompt_tokens = count(object.get("total_input_tokens"))
        .checked_add(count(object.get("total_tool_use_tokens")))
        .ok_or(CostError::TokenCountOverflow)?;
    let completion_tokens = count(object.get("total_output_tokens"))
        .checked_add(reasoning)
        .ok_or(CostError::TokenCountOverflow)?;
    let derived_total = prompt_tokens
        .checked_add(completion_tokens)
        .ok_or(CostError::TokenCountOverflow)?;
    let total_tokens = match count(object.get("total_tokens")) {
        0 => derived_total,
        reported => reported,
    };
    let web_search_requests = entries(object, "grounding_tool_count")
        .iter()
        .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("google_search"))
        .try_fold(0_u64, |sum, entry| {
            sum.checked_add(count(entry.get("count")))
                .ok_or(CostError::TokenCountOverflow)
        })?;
    let prompt_tokens_details = (input.any() || total_cached > 0 || web_search_requests > 0)
        .then_some(PromptTokenDetails {
            cached_tokens: total_cached,
            text_tokens: input.text,
            audio_tokens: input.audio,
            image_tokens: input.image,
            video_tokens: input.video,
            web_search_requests: (web_search_requests > 0).then_some(web_search_requests),
            ..PromptTokenDetails::default()
        });
    let completion_tokens_details =
        (output.any() || reasoning > 0).then_some(CompletionTokenDetails {
            reasoning_tokens: (reasoning > 0).then_some(reasoning),
            text_tokens: output.text,
            audio_tokens: output.audio,
            image_tokens: output.image,
            video_tokens: output.video,
        });
    let mut extra = BTreeMap::new();
    if total_cached > 0 {
        extra.insert(
            "cache_read_input_tokens".to_string(),
            Value::from(total_cached),
        );
        extra.insert(
            "_cache_read_input_tokens".to_string(),
            Value::from(total_cached),
        );
    }
    Ok(ChatUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        prompt_tokens_details,
        completion_tokens_details,
        cost: None,
        extra,
    })
}
