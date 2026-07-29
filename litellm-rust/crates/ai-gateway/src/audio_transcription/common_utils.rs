use std::collections::BTreeMap;

use litellm_core::CoreResult;
use litellm_core::audio_transcription::transformation::AudioTranscriptionProviderConfig;
use litellm_core::error::CoreError;
use litellm_core::providers::bedrock::audio_transcription::BEDROCK_AUDIO_TRANSCRIPTION_CONFIG;
use serde_json::{Map, Value};

pub(super) fn audio_transcription_provider_config(
    provider: &str,
) -> Option<&'static dyn AudioTranscriptionProviderConfig> {
    match provider {
        "bedrock" => Some(&BEDROCK_AUDIO_TRANSCRIPTION_CONFIG),
        _ => None,
    }
}

pub(super) fn string_headers(
    headers: Option<Map<String, Value>>,
) -> CoreResult<BTreeMap<String, String>> {
    headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    CoreError::InvalidRequest(format!(
                        "audio transcription extra_headers.{key} must be a string"
                    ))
                })
        })
        .collect()
}

pub(super) fn has_header(headers: &BTreeMap<String, String>, name: &str) -> bool {
    headers.keys().any(|key| key.eq_ignore_ascii_case(name))
}

pub(super) fn truncate_error_body(body: &str) -> String {
    let truncated: String = body.chars().take(256).collect();
    if truncated.chars().count() == body.chars().count() {
        truncated
    } else {
        format!("{truncated}... (truncated)")
    }
}
