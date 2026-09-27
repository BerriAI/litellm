use crate::error::CostError;
use serde::Deserialize;
use serde_json::Value;

use crate::responses_usage::{ChatUsage, PromptTokenDetails};

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TranscriptionUsage {
    Duration {
        #[serde(rename = "seconds")]
        _seconds: f64,
    },
    Tokens {
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
        input_token_details: InputTokenDetails,
    },
}

#[derive(Deserialize)]
struct InputTokenDetails {
    text_tokens: u64,
    audio_tokens: u64,
}

pub fn is_transcription_usage_object(usage: &Value) -> bool {
    serde_json::from_value::<TranscriptionUsage>(usage.clone()).is_ok()
}

pub fn transform_transcription_usage_object(usage: &Value) -> Result<Option<ChatUsage>, CostError> {
    let parsed: TranscriptionUsage =
        serde_json::from_value(usage.clone()).map_err(|_| CostError::InvalidShape)?;
    match parsed {
        TranscriptionUsage::Duration { .. } => Ok(None),
        TranscriptionUsage::Tokens {
            input_tokens,
            output_tokens,
            total_tokens,
            input_token_details,
        } => Ok(Some(ChatUsage {
            prompt_tokens: input_tokens,
            completion_tokens: output_tokens,
            total_tokens,
            prompt_tokens_details: Some(PromptTokenDetails {
                text_tokens: Some(input_token_details.text_tokens),
                audio_tokens: Some(input_token_details.audio_tokens),
                ..PromptTokenDetails::default()
            }),
            completion_tokens_details: None,
            cost: None,
            extra: Default::default(),
        })),
    }
}
