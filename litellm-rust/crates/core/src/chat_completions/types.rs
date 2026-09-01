use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::transformation::{ChatCompletionsAuth, ChatCompletionsProviderConfig};
use crate::streaming::{JsonObject, ProviderCallContext};

/// A `/chat/completions` call as it crosses into the core.
///
/// `optional_params` arrives already mapped to the provider's own parameter
/// names by the host, exactly as the messages route receives an already
/// Anthropic-shaped body. The core owns the conversation translation, the
/// provider call, and the response normalization.
pub struct ChatCompletionsRequest<'a> {
    pub model: &'a str,
    pub messages: Value,
    pub optional_params: Map<String, Value>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

pub(super) struct ProviderChatCompletionsRequest {
    pub(super) model: String,
    pub(super) config: &'static dyn ChatCompletionsProviderConfig,
    pub(super) url: String,
    pub(super) body: Value,
    pub(super) upstream_headers: Vec<(String, String)>,
    pub(super) auth: ChatCompletionsAuth,
    #[cfg_attr(not(feature = "bedrock-auth"), allow(dead_code))]
    pub(super) optional_params: Map<String, Value>,
    pub(super) timeout: Option<Duration>,
}

/// The provider-shaped request body a config produces. Named rather than a bare
/// `Value` so the transform contract stays a typed one, mirroring
/// [`crate::audio_transcription::types::AudioTranscriptionRequestData`].
pub struct ProviderChatRequestData {
    pub body: Value,
}

/// The raw provider response body handed back to a config for normalization.
pub struct ProviderChatResponseData {
    pub body: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<Value>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// OpenAI `usage`, including the `prompt_tokens_details` split LiteLLM's Python
/// path reports so cost tracking sees the same numbers on either path.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub text_tokens: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsChoiceMessage {
    pub role: String,
    // Whether an empty turn is `None` or `""` is the provider's choice, not a
    // shared invariant: Anthropic's transform ends on `merged_text or None`
    // while Converse assigns the joined string unconditionally. Each config
    // mirrors its own, so keep this optional and serialize it even when None.
    pub content: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsChoice {
    pub index: u64,
    pub message: ChatCompletionsChoiceMessage,
    pub finish_reason: String,
}

/// The normalized response handed back to the host.
///
/// There is deliberately no `id`: Python mints the `chatcmpl-…` id on the
/// `ModelResponse` it already created, and echoing the provider's own id here
/// would change it. Pinned by `response_carries_no_id` in `tests.rs`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsResponse {
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionsChoice>,
    pub usage: ChatCompletionsUsage,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatStreamRole {
    Assistant,
    Developer,
    Function,
    System,
    Tool,
    User,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatStreamMessageContent {
    Text(String),
    Parts(Vec<JsonObject>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatStreamMessage {
    pub role: ChatStreamRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatStreamMessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatStreamStop {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatStreamStringOrObject {
    Name(String),
    Definition(JsonObject),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsStreamParameters {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<ChatStreamStop>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<JsonObject>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatStreamStringOrObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ChatStreamStringOrObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionsStreamRequestBody {
    pub model: String,
    pub messages: Vec<ChatStreamMessage>,
    #[serde(flatten)]
    pub parameters: ChatCompletionsStreamParameters,
}

pub struct ChatCompletionsStreamRequest {
    pub body: ChatCompletionsStreamRequestBody,
    pub context: ProviderCallContext,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatStreamToolFunctionChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<JsonObject>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatStreamToolCallChunk {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ChatStreamToolFunctionChunk,
    pub index: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatStreamUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<JsonObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<JsonObject>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatStreamEvent {
    pub text: String,
    pub tool_use: Option<ChatStreamToolCallChunk>,
    pub is_finished: bool,
    pub finish_reason: String,
    pub usage: Option<ChatStreamUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<JsonObject>,
}

#[cfg(test)]
mod stream_contract_tests {
    use super::*;

    #[test]
    fn request_uses_public_chat_completion_parameter_names() {
        let request: ChatCompletionsStreamRequestBody = serde_json::from_value(serde_json::json!({
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "stream": true,
            "tool_choice": "auto"
        }))
        .expect("public request shape");

        assert_eq!(request.parameters.max_tokens, Some(32));
        assert_eq!(request.parameters.stream, Some(true));
        assert!(matches!(
            request.parameters.tool_choice,
            Some(ChatStreamStringOrObject::Name(ref value)) if value == "auto"
        ));
    }

    #[test]
    fn event_matches_python_generic_streaming_chunk_shape() {
        let event = ChatStreamEvent {
            text: "hello".to_string(),
            tool_use: None,
            is_finished: false,
            finish_reason: String::new(),
            usage: None,
            index: Some(0),
            provider_specific_fields: None,
        };

        assert_eq!(
            serde_json::to_value(event).expect("serializable event"),
            serde_json::json!({
                "text": "hello",
                "tool_use": null,
                "is_finished": false,
                "finish_reason": "",
                "usage": null,
                "index": 0
            })
        );
    }
}
