use crate::auth::RequestAuth;
use crate::chat_completions::error::{ChatRequestError, ChatResponseError};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::chat_completions::conversation::{Conversation, build_conversation};
use crate::chat_completions::transformation::{
    ChatCompletionsProviderConfig, Unsupported, unsupported_message, unsupported_param,
};
use crate::chat_completions::types::{
    ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse, ChatMessage,
};
use crate::constants::ANTHROPIC_OAUTH_TOKEN_PREFIX;
use crate::error::Error;
use crate::providers::anthropic::messages::transformation::{
    complete_anthropic_url, resolve_anthropic_api_key,
};

use crate::chat_completions::response_utils::{finish_reason_for, unix_now, usage_from_parts};

/// Anthropic parameter names, post `map_openai_params`, that the Rust path can
/// place verbatim in the Messages body.
///
/// `top_k` is deliberately absent even though the Messages API takes it.
/// `temperature` and `top_p` reach this gate already resolved, because
/// `map_openai_params` runs first and applies `_apply_sampling_param` to them.
/// `top_k` bypasses `map_openai_params` entirely, so Python applies that same
/// per-model gate inside `transform_request`, the function this route replaces.
/// Forwarding it would send `top_k` to a model that removed sampling params and
/// take a 400 after the call, where Python drops it and succeeds.
const SUPPORTED_PARAMS: &[&str] = &["max_tokens", "temperature", "top_p", "stop_sequences"];

pub struct AnthropicChatCompletionsConfig;

pub const ANTHROPIC_CHAT_COMPLETIONS_CONFIG: AnthropicChatCompletionsConfig =
    AnthropicChatCompletionsConfig;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnthropicMappedParams {
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
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::http_utils::deserialize_optional_param"
    )]
    pub stream: Option<Option<bool>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnthropicChatRequest {
    pub model: String,
    pub messages: Vec<AnthropicTextMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub system: Vec<AnthropicTextBlock>,
    #[serde(flatten)]
    pub params: AnthropicMappedParams,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnthropicTextMessage {
    pub role: crate::chat_completions::conversation::TurnRole,
    pub content: Vec<AnthropicTextBlock>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicTextBlock {
    Text { text: String },
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicChatResponse {
    pub model: Option<String>,
    pub content: Option<Vec<AnthropicResponseBlock>>,
    pub stop_reason: Option<String>,
    pub usage: Option<AnthropicChatUsage>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicResponseBlock {
    Text {
        text: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct AnthropicChatUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
}

impl ChatCompletionsProviderConfig for AnthropicChatCompletionsConfig {
    type MappedParams = AnthropicMappedParams;
    type RequestBody = AnthropicChatRequest;
    type ResponseBody = AnthropicChatResponse;

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_anthropic_url(api_base, env_lookup))
    }

    fn auth(
        &self,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<RequestAuth, Error> {
        Ok(RequestAuth::Header {
            name: "x-api-key",
            value: resolve_anthropic_api_key(api_key, env_lookup)?,
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ]
    }

    /// An OAuth bearer is the whole credential: Python's `validate_environment`
    /// authenticates with it and drops `x-api-key` rather than resolving one, so
    /// the resolved key must not be applied over the top. Any other forwarded
    /// `authorization` is unrelated to this header and does not defer, which is
    /// also what Python does: it sends the deployment's `x-api-key` alongside.
    fn defers_to_forwarded_auth(&self, headers: &[(String, String)]) -> bool {
        headers.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case("authorization")
                && value
                    .strip_prefix("Bearer ")
                    .is_some_and(|token| token.starts_with(ANTHROPIC_OAUTH_TOKEN_PREFIX))
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn supported_provider_params(&self) -> &'static [&'static str] {
        SUPPORTED_PARAMS
    }

    fn decline_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        unsupported_param(self.supported_provider_params(), &[], optional_params)
            .or_else(|| messages.iter().find_map(unsupported_message))
            // Anthropic rejects a request whose first turn is not a user turn.
            // Python only repairs that under `litellm.modify_params`, which the
            // core cannot observe, so decline instead of guessing.
            .or_else(|| {
                (!build_conversation(messages).opens_on_user_turn())
                    .then_some(Unsupported("conversation does not open on a user turn"))
            })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_request(
        &self,
        model: &str,
        conversation: Conversation,
        params: Self::MappedParams,
    ) -> Result<Self::RequestBody, ChatRequestError> {
        Ok(AnthropicChatRequest {
            model: model.to_string(),
            messages: conversation
                .turns
                .into_iter()
                .map(|turn| AnthropicTextMessage {
                    role: turn.role,
                    content: turn
                        .texts
                        .into_iter()
                        .map(|text| AnthropicTextBlock::Text { text })
                        .collect(),
                })
                .collect(),
            system: conversation
                .system
                .into_iter()
                .map(|text| AnthropicTextBlock::Text { text })
                .collect(),
            params,
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_response(
        &self,
        _model: &str,
        response: Self::ResponseBody,
    ) -> Result<ChatCompletionsResponse, ChatResponseError> {
        let content = response
            .content
            .ok_or(ChatResponseError::MissingField("content"))?;
        let text = content
            .into_iter()
            .map(|block| match block {
                AnthropicResponseBlock::Text { text } => Ok(text.unwrap_or_default()),
                AnthropicResponseBlock::Other => Err(ChatResponseError::NonTextContent),
            })
            .collect::<Result<String, _>>()?;
        let usage = response
            .usage
            .ok_or(ChatResponseError::MissingField("usage"))?;

        Ok(ChatCompletionsResponse {
            created: unix_now(),
            model: response
                .model
                .ok_or(ChatResponseError::MissingField("model"))?,
            choices: vec![ChatCompletionsChoice {
                index: 0,
                message: ChatCompletionsChoiceMessage {
                    role: "assistant".to_string(),
                    content: (!text.is_empty()).then_some(text),
                },
                finish_reason: finish_reason_for(response.stop_reason.as_deref().unwrap_or(""))
                    .to_string(),
            }],
            usage: usage_from_parts(
                usage.input_tokens.unwrap_or(0),
                usage.output_tokens.unwrap_or(0),
                usage.cache_read_input_tokens.unwrap_or(0),
                usage.cache_creation_input_tokens.unwrap_or(0),
            ),
        })
    }
}

#[cfg(test)]
#[path = "../../../../tests/anthropic_chat_completions.rs"]
mod tests;
