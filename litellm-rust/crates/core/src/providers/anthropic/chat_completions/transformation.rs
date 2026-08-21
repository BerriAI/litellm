use serde_json::{Map, Value, json};

use crate::chat_completions::conversation::{Conversation, build_conversation};
use crate::chat_completions::transformation::{
    ChatCompletionsAuth, ChatCompletionsProviderConfig, Unsupported, unsupported_message,
    unsupported_param,
};
use crate::chat_completions::types::{
    ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse, ChatMessage,
    ProviderChatRequestData, ProviderChatResponseData,
};
use crate::constants::ANTHROPIC_OAUTH_TOKEN_PREFIX;
use crate::error::{CoreError, CoreResult};
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

fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

fn anthropic_body(model: &str, conversation: &Conversation, params: Map<String, Value>) -> Value {
    let messages: Vec<Value> = conversation
        .turns
        .iter()
        .map(|turn| {
            json!({
                "role": turn.role.as_str(),
                "content": turn.texts.iter().map(|text| text_block(text)).collect::<Vec<_>>(),
            })
        })
        .collect();

    let system: Vec<Value> = conversation.system.iter().map(|s| text_block(s)).collect();

    let body = Map::from_iter(
        [
            ("model".to_string(), json!(model)),
            ("messages".to_string(), json!(messages)),
        ]
        .into_iter()
        // Python builds `{"model", "messages", **optional_params}` with
        // `system` already folded into optional_params, so a caller-supplied
        // key of the same name wins here too.
        .chain((!system.is_empty()).then(|| ("system".to_string(), json!(system))))
        .chain(params),
    );
    Value::Object(body)
}

impl ChatCompletionsProviderConfig for AnthropicChatCompletionsConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(complete_anthropic_url(api_base, env_lookup))
    }

    fn auth(
        &self,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<ChatCompletionsAuth> {
        Ok(ChatCompletionsAuth::Header {
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

    fn supported_params(&self) -> &'static [&'static str] {
        SUPPORTED_PARAMS
    }

    fn unsupported_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        unsupported_param(SUPPORTED_PARAMS, &[], optional_params)
            .or_else(|| messages.iter().find_map(unsupported_message))
            // Anthropic rejects a request whose first turn is not a user turn.
            // Python only repairs that under `litellm.modify_params`, which the
            // core cannot observe, so decline instead of guessing.
            .or_else(|| {
                (!build_conversation(messages).opens_on_user_turn())
                    .then_some(Unsupported("conversation does not open on a user turn"))
            })
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> CoreResult<ProviderChatRequestData> {
        Ok(ProviderChatRequestData {
            body: anthropic_body(model, &build_conversation(&messages), optional_params),
        })
    }

    fn transform_response(
        &self,
        _model: &str,
        response: ProviderChatResponseData,
    ) -> CoreResult<ChatCompletionsResponse> {
        let body = response.body.as_object().ok_or_else(|| {
            CoreError::InvalidResponse("messages response is not an object".into())
        })?;

        let content = body
            .get("content")
            .and_then(Value::as_array)
            .ok_or(CoreError::MissingField("content"))?;
        // The route declines tool and thinking requests, so a non-text block
        // means the response carries something this path never asked for.
        // Decline rather than silently dropping it; the host falls back.
        if content
            .iter()
            .any(|block| block.get("type").and_then(Value::as_str) != Some("text"))
        {
            return Err(CoreError::Unsupported("non-text response content block"));
        }
        let text: String = content
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect();

        let usage = body
            .get("usage")
            .and_then(Value::as_object)
            .ok_or(CoreError::MissingField("usage"))?;
        let field = |name: &str| usage.get(name).and_then(Value::as_u64).unwrap_or(0);

        Ok(ChatCompletionsResponse {
            created: unix_now(),
            model: body
                .get("model")
                .and_then(Value::as_str)
                .ok_or(CoreError::MissingField("model"))?
                .to_string(),
            choices: vec![ChatCompletionsChoice {
                index: 0,
                message: ChatCompletionsChoiceMessage {
                    role: "assistant".to_string(),
                    content: (!text.is_empty()).then_some(text),
                },
                finish_reason: finish_reason_for(
                    body.get("stop_reason")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )
                .to_string(),
            }],
            usage: usage_from_parts(
                field("input_tokens"),
                field("output_tokens"),
                field("cache_read_input_tokens"),
                field("cache_creation_input_tokens"),
            ),
        })
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
