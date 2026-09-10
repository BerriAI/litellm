use serde::{Deserialize, Serialize};

use crate::chat_completions::conversation::TurnRole;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseMappedParams {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::http_utils::deserialize_optional_param"
    )]
    pub max_tokens: Option<Option<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::http_utils::deserialize_optional_param"
    )]
    pub temperature: Option<Option<serde_json::Number>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::http_utils::deserialize_optional_param"
    )]
    pub top_p: Option<Option<serde_json::Number>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::http_utils::deserialize_optional_param"
    )]
    pub stop_sequences: Option<Option<Vec<String>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseRequest {
    pub messages: Vec<ConverseMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub system: Vec<ConverseText>,
    pub inference_config: ConverseMappedParams,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConverseMessage {
    pub role: TurnRole,
    pub content: Vec<ConverseContent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ConverseContent {
    Text(String),
    Audio(ConverseAudio),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConverseText {
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConverseAudio {
    pub format: crate::audio_transcription::types::AudioFormat,
    pub source: ConverseAudioSource,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConverseAudioSource {
    pub bytes: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ConverseResponse {
    pub output: Option<ConverseOutput>,
    #[serde(rename = "stopReason")]
    pub stop_reason: Option<String>,
    pub usage: Option<ConverseUsage>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ConverseOutput {
    pub message: Option<ConverseResponseMessage>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ConverseResponseMessage {
    pub content: Option<Vec<ConverseResponseContent>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum ConverseResponseContent {
    Object(ConverseResponseObject),
    Other(serde::de::IgnoredAny),
}

#[derive(Clone, Debug, Deserialize)]
pub struct ConverseResponseObject {
    #[serde(
        default,
        deserialize_with = "crate::http_utils::deserialize_optional_param"
    )]
    pub text: Option<Option<String>>,
    #[serde(flatten)]
    pub other_fields: std::collections::BTreeMap<String, serde::de::IgnoredAny>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConverseUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_write_input_tokens: Option<u64>,
}
