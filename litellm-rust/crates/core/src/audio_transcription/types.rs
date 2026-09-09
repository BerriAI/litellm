use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::integrations::types::RequestMetadata;

use super::transformation::{AudioTranscriptionAuth, AudioTranscriptionProviderConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
}

impl AsRef<str> for AudioFormat {
    fn as_ref(&self) -> &str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::Ogg => "ogg",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioInput {
    pub data: String,
    pub format: AudioFormat,
    #[serde(default)]
    pub filename: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AudioPreCallGuardrailRequest {
    model: String,
    custom_llm_provider: String,
    audio: AudioInput,
    optional_params: Map<String, Value>,
}

impl AudioPreCallGuardrailRequest {
    pub(crate) fn new(
        model: String,
        custom_llm_provider: String,
        audio: AudioInput,
        optional_params: Map<String, Value>,
    ) -> Self {
        Self {
            model,
            custom_llm_provider,
            audio,
            optional_params,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn custom_llm_provider(&self) -> &str {
        &self.custom_llm_provider
    }

    pub fn audio(&self) -> &AudioInput {
        &self.audio
    }

    pub fn optional_params(&self) -> &Map<String, Value> {
        &self.optional_params
    }

    pub fn with_audio(self, audio: AudioInput) -> Self {
        Self { audio, ..self }
    }

    pub fn with_optional_params(self, optional_params: Map<String, Value>) -> Self {
        Self {
            optional_params,
            ..self
        }
    }

    pub(crate) fn into_payload(self) -> (AudioInput, Map<String, Value>) {
        (self.audio, self.optional_params)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AudioDuringCallGuardrailRequest {
    model: String,
    custom_llm_provider: String,
    url: String,
    body: Map<String, Value>,
}

impl AudioDuringCallGuardrailRequest {
    pub(crate) fn new(
        model: String,
        custom_llm_provider: String,
        url: String,
        body: Map<String, Value>,
    ) -> Self {
        Self {
            model,
            custom_llm_provider,
            url,
            body,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn custom_llm_provider(&self) -> &str {
        &self.custom_llm_provider
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn body(&self) -> &Map<String, Value> {
        &self.body
    }

    pub fn with_body(self, body: Map<String, Value>) -> Self {
        Self { body, ..self }
    }

    pub(crate) fn into_body(self) -> Map<String, Value> {
        self.body
    }
}

pub struct AudioTranscriptionRequest<'a> {
    pub model: &'a str,
    pub audio: AudioInput,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
}

pub struct AudioRouteRequest<'a> {
    pub model: &'a str,
    pub audio: AudioInput,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub request_metadata: RequestMetadata,
    pub litellm_call_id: Option<&'a str>,
}

#[derive(Clone)]
pub struct ProviderAudioTranscriptionRequest {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
    pub(crate) config: &'static dyn AudioTranscriptionProviderConfig,
    pub(crate) url: String,
    pub(crate) body: Map<String, Value>,
    pub(crate) upstream_headers: Vec<(String, String)>,
    pub(crate) auth: AudioTranscriptionAuth,
    #[cfg(feature = "bedrock-auth")]
    pub(crate) optional_params: Map<String, Value>,
    pub(crate) timeout: Option<Duration>,
}

pub struct AudioAuthorizationContext<'a> {
    url: &'a str,
    upstream_headers: &'a [(String, String)],
    auth: &'a AudioTranscriptionAuth,
    #[cfg(feature = "bedrock-auth")]
    optional_params: &'a Map<String, Value>,
}

impl AudioAuthorizationContext<'_> {
    pub(crate) fn url(&self) -> &str {
        self.url
    }

    pub(crate) fn upstream_headers(&self) -> &[(String, String)] {
        self.upstream_headers
    }

    pub(crate) fn auth(&self) -> &AudioTranscriptionAuth {
        self.auth
    }

    #[cfg(feature = "bedrock-auth")]
    pub(crate) fn optional_params(&self) -> &Map<String, Value> {
        self.optional_params
    }
}

impl ProviderAudioTranscriptionRequest {
    pub(crate) fn authorization_context(&self) -> AudioAuthorizationContext<'_> {
        AudioAuthorizationContext {
            url: &self.url,
            upstream_headers: &self.upstream_headers,
            auth: &self.auth,
            #[cfg(feature = "bedrock-auth")]
            optional_params: &self.optional_params,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn custom_llm_provider(&self) -> &str {
        &self.custom_llm_provider
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn body(&self) -> &Map<String, Value> {
        &self.body
    }

    pub fn with_body(self, body: Map<String, Value>) -> Self {
        Self { body, ..self }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioTranscriptionRequestData {
    pub body: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioTranscriptionResponseData {
    pub text: String,
}

impl AudioTranscriptionResponseData {
    pub fn into_json(self) -> Value {
        serde_json::json!({
            "text": self.text,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{AudioFormat, AudioInput};

    #[test]
    fn audio_input_accepts_every_supported_format() {
        for (format, expected) in [
            (AudioFormat::Wav, "wav"),
            (AudioFormat::Mp3, "mp3"),
            (AudioFormat::Flac, "flac"),
            (AudioFormat::Ogg, "ogg"),
        ] {
            let input = AudioInput {
                data: "AQI=".to_string(),
                format,
                filename: None,
            };
            assert_eq!(serde_json::to_value(input).unwrap()["format"], expected);
        }
    }

    #[test]
    fn audio_input_requires_data() {
        let result = serde_json::from_value::<AudioInput>(json!({"format": "wav"}));
        assert!(result.is_err());
    }
}
