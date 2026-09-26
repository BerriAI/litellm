use crate::error::CostError;
use serde_json::Value;

use crate::responses_usage::ChatUsage;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeechCostMetric {
    PerCharacter,
    PerToken,
}

pub fn count_characters(text: &str) -> usize {
    text.chars()
        .filter(|character| {
            !character.is_whitespace() && !matches!(character, '\u{001c}'..='\u{001f}')
        })
        .count()
}

pub fn select_cost_metric_for_model(model_info: &Value) -> Result<SpeechCostMetric, CostError> {
    if model_info
        .get("input_cost_per_character")
        .and_then(Value::as_f64)
        .is_some_and(|rate| rate != 0.0)
    {
        return Ok(SpeechCostMetric::PerCharacter);
    }
    if model_info
        .get("input_cost_per_token")
        .and_then(Value::as_f64)
        .is_some_and(|rate| rate != 0.0)
    {
        return Ok(SpeechCostMetric::PerToken);
    }
    Err(CostError::MissingMetric)
}

pub fn generic_cost_per_character(
    model_info: &Value,
    prompt_characters: f64,
    completion_characters: f64,
    custom_prompt_rate: Option<f64>,
    custom_completion_rate: Option<f64>,
) -> (Option<f64>, Option<f64>) {
    let prompt_rate = custom_prompt_rate.or_else(|| {
        model_info
            .get("input_cost_per_character")
            .and_then(Value::as_f64)
    });
    let completion_rate = custom_completion_rate.or_else(|| {
        model_info
            .get("output_cost_per_character")
            .and_then(Value::as_f64)
    });
    (
        prompt_rate.map(|rate| prompt_characters * rate),
        completion_rate.map(|rate| completion_characters * rate),
    )
}

pub fn cost_per_second(model_info: &Value, duration_seconds: f64) -> (f64, f64) {
    let output_rate = model_info
        .get("output_cost_per_second")
        .and_then(Value::as_f64);
    if let Some(rate) = output_rate.filter(|rate| *rate > 0.0) {
        return (0.0, rate * duration_seconds);
    }
    let input_rate = model_info
        .get("input_cost_per_second")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    (input_rate * duration_seconds, 0.0)
}

pub fn transcription_usage_has_token_details(usage: &ChatUsage) -> bool {
    usage.prompt_tokens > 0
        || usage.completion_tokens > 0
        || usage.prompt_tokens_details.as_ref().is_some_and(|details| {
            details.audio_tokens.unwrap_or(0) > 0 || details.text_tokens.unwrap_or(0) > 0
        })
}

pub fn lyria_generation_cost(model_info: &Value) -> Option<f64> {
    let audio_api = model_info.get("vertex_ai_audio_api")?.as_str()?;
    if !matches!(audio_api, "lyria_predict" | "lyria_interactions")
        || !model_info
            .get("supported_audio_formats")
            .is_some_and(Value::is_array)
    {
        return None;
    }
    model_info
        .get("output_cost_per_image")
        .and_then(Value::as_f64)
}
