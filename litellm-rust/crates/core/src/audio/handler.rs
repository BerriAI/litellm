//! Audio request execution logic.

use reqwest::multipart::{Form, Part};
use serde_json::Value;

use crate::error::{CoreError, CoreResult};

use super::client::get_http_client;
use super::prepare::{prepare_speech_request, prepare_transcription_request};
use super::types::{SpeechRequest, SpeechResponse, TranscriptionRequest, TranscriptionResponse};

/// Execute a speech (TTS) request.
pub async fn speech(
    request: &SpeechRequest,
    provider: &str,
    api_base: Option<&str>,
    api_key: Option<String>,
) -> CoreResult<SpeechResponse> {
    let prepared = prepare_speech_request(request, provider, api_base, api_key)?;

    let client = get_http_client();
    let mut req_builder = client.post(&prepared.url).json(&prepared.body);

    if let Some(api_key) = prepared.api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = req_builder.send().await.map_err(|e| {
        CoreError::Network(format!("Failed to send speech request: {}", e))
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: error_text,
        });
    }

    let audio_data = response.bytes().await.map_err(|e| {
        CoreError::Network(format!("Failed to read speech response: {}", e))
    })?;

    Ok(SpeechResponse {
        audio_data: audio_data.to_vec(),
        content_type: prepared.content_type,
    })
}

/// Execute a transcription (STT) request.
pub async fn transcription(
    request: &TranscriptionRequest,
    provider: &str,
    api_base: Option<&str>,
    api_key: Option<String>,
) -> CoreResult<TranscriptionResponse> {
    let prepared = prepare_transcription_request(request, provider, api_base, api_key)?;

    let client = get_http_client();

    // Build multipart form data
    let mut form = Form::new();

    // Add the audio file
    let file_part = Part::bytes(prepared.file)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| CoreError::InvalidRequest(format!("Failed to create file part: {}", e)))?;
    form = form.part("file", file_part);

    // Add other fields from the body
    if let Value::Object(map) = &prepared.body {
        for (key, value) in map {
            if key != "file" {
                let text_value = match value {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => continue,
                };
                form = form.text(key.clone(), text_value);
            }
        }
    }

    let mut req_builder = client.post(&prepared.url).multipart(form);

    if let Some(api_key) = prepared.api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = req_builder.send().await.map_err(|e| {
        CoreError::Network(format!("Failed to send transcription request: {}", e))
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: error_text,
        });
    }

    let response_text = response.text().await.map_err(|e| {
        CoreError::Network(format!("Failed to read transcription response: {}", e))
    })?;

    // Try to parse as JSON first
    if let Ok(json_response) = serde_json::from_str::<Value>(&response_text) {
        if let Some(text) = json_response.get("text").and_then(|v| v.as_str()) {
            return Ok(TranscriptionResponse {
                text: text.to_string(),
            });
        }
    }

    // If not JSON, treat the entire response as the text
    Ok(TranscriptionResponse {
        text: response_text,
    })
}
