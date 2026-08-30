//! Audio provider transformation logic.

use serde_json::Value;

use crate::error::{CoreError, CoreResult};

use super::types::{SpeechRequest, TranscriptionRequest};

/// Provider configuration for audio endpoints.
pub trait AudioProviderConfig: Send + Sync {
    /// Get the base URL for the speech endpoint.
    fn speech_url(&self, api_base: Option<&str>) -> String;

    /// Get the base URL for the transcription endpoint.
    fn transcription_url(&self, api_base: Option<&str>) -> String;

    /// Transform a speech request into the provider's format.
    fn transform_speech_request(&self, request: &SpeechRequest) -> CoreResult<Value>;

    /// Transform a transcription request into the provider's format.
    fn transform_transcription_request(&self, request: &TranscriptionRequest) -> CoreResult<Value>;

    /// Get the content type for the speech response.
    fn speech_response_content_type(&self, response_format: Option<&str>) -> String;
}

/// OpenAI provider configuration for audio endpoints.
pub struct OpenAiAudioConfig;

impl AudioProviderConfig for OpenAiAudioConfig {
    fn speech_url(&self, api_base: Option<&str>) -> String {
        let base = api_base.unwrap_or("https://api.openai.com/v1");
        format!("{}/audio/speech", base.trim_end_matches('/'))
    }

    fn transcription_url(&self, api_base: Option<&str>) -> String {
        let base = api_base.unwrap_or("https://api.openai.com/v1");
        format!("{}/audio/transcriptions", base.trim_end_matches('/'))
    }

    fn transform_speech_request(&self, request: &SpeechRequest) -> CoreResult<Value> {
        let mut body = serde_json::json!({
            "model": request.model,
            "input": request.input,
            "voice": request.voice,
        });

        if let Some(ref format) = request.response_format {
            body["response_format"] = Value::String(format.clone());
        }

        if let Some(speed) = request.speed {
            body["speed"] = Value::Number(serde_json::Number::from_f64(speed as f64).unwrap());
        }

        Ok(body)
    }

    fn transform_transcription_request(&self, request: &TranscriptionRequest) -> CoreResult<Value> {
        let mut body = serde_json::json!({
            "model": request.model,
        });

        if let Some(ref language) = request.language {
            body["language"] = Value::String(language.clone());
        }

        if let Some(ref prompt) = request.prompt {
            body["prompt"] = Value::String(prompt.clone());
        }

        if let Some(ref format) = request.response_format {
            body["response_format"] = Value::String(format.clone());
        }

        if let Some(temp) = request.temperature {
            body["temperature"] = Value::Number(serde_json::Number::from_f64(temp as f64).unwrap());
        }

        Ok(body)
    }

    fn speech_response_content_type(&self, response_format: Option<&str>) -> String {
        match response_format {
            Some("mp3") => "audio/mpeg".to_string(),
            Some("opus") => "audio/ogg".to_string(),
            Some("aac") => "audio/aac".to_string(),
            Some("flac") => "audio/flac".to_string(),
            _ => "audio/mpeg".to_string(),
        }
    }
}

/// Get the audio provider config for a given provider name.
pub fn get_audio_provider_config(provider: &str) -> CoreResult<Box<dyn AudioProviderConfig>> {
    match provider {
        "openai" => Ok(Box::new(OpenAiAudioConfig)),
        _ => Err(CoreError::InvalidRequest(format!(
            "Unsupported audio provider: {}",
            provider
        ))),
    }
}
