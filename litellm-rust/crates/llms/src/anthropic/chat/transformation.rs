use litellm_auth::SecretValue;
use litellm_core_utils::{
    core_helpers::{finish_reason_for, unix_now, usage_from_parts},
    prompt_templates::factory::{Conversation, build_conversation},
};
use litellm_types::{
    llms::openai::ChatMessage,
    utils::{ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse},
};
use serde_json::{Map, Value, json};

use crate::{
    Error,
    anthropic::{
        chat::handler::ModelResponseIterator,
        common_utils::{
            API_KEY_PLACEMENT, complete_anthropic_url, forwarded_oauth_bearer,
            resolve_anthropic_api_key,
        },
    },
    base_llm::{
        anthropic_messages::streaming::anthropic_sse_event_stream,
        auth::AuthScheme,
        chat::{
            streaming::{ChatStream, StreamShape},
            transformation::{
                BaseConfig, Headers, ProviderChatRequestData, ProviderChatResponseData,
                Unsupported, ValidatedEnvironment, unsupported_message, unsupported_param,
            },
        },
    },
};

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
const SUPPORTED_PARAMS: &[(&str, &str)] = &[
    ("max_tokens", "max_tokens"),
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("stop", "stop_sequences"),
];

pub struct AnthropicConfig;

pub const ANTHROPIC_CHAT_COMPLETIONS_CONFIG: AnthropicConfig = AnthropicConfig;

impl BaseConfig for AnthropicConfig {
    fn supported_openai_param_mappings(&self) -> &'static [(&'static str, &'static str)] {
        SUPPORTED_PARAMS
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_anthropic_url(api_base, env_lookup))
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> Result<ProviderChatRequestData, Error> {
        Ok(ProviderChatRequestData {
            body: anthropic_body(model, &build_conversation(&messages), optional_params),
            stream_shape: StreamShape::default(),
        })
    }

    fn transform_response(
        &self,
        _model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error> {
        let body = response
            .body
            .as_object()
            .ok_or_else(|| Error::InvalidResponse("messages response is not an object".into()))?;

        let content = body
            .get("content")
            .and_then(Value::as_array)
            .ok_or(Error::MissingField("content"))?;
        // The route declines tool and thinking requests, so a non-text block
        // means the response carries something this path never asked for.
        // Decline rather than silently dropping it; the host falls back.
        if content
            .iter()
            .any(|block| block.get("type").and_then(Value::as_str) != Some("text"))
        {
            return Err(Error::Unsupported("non-text response content block"));
        }
        let text: String = content
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect();

        let usage = body
            .get("usage")
            .and_then(Value::as_object)
            .ok_or(Error::MissingField("usage"))?;
        let field = |name: &str| usage.get(name).and_then(Value::as_u64).unwrap_or(0);

        Ok(ChatCompletionsResponse {
            created: unix_now(),
            model: body
                .get("model")
                .and_then(Value::as_str)
                .ok_or(Error::MissingField("model"))?
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

    /// A forwarded OAuth bearer is the whole credential: Python pops `x-api-key` for it,
    /// so the resolved key is not applied over it. Any other forwarded header loses to
    /// the deployment's key, which Python writes last.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if forwarded_oauth_bearer(&headers).is_some() {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let auth = AuthScheme::Credential {
            placement: API_KEY_PLACEMENT,
            secret: SecretValue::new(resolve_anthropic_api_key(api_key, env_lookup)?),
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn model_response_iterator(&self, shape: StreamShape) -> Option<ChatStream> {
        Some(ChatStream::new(
            anthropic_sse_event_stream,
            ModelResponseIterator::new(shape),
        ))
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
    fn unsupported_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        unsupported_param(self.supported_openai_param_mappings(), &[], optional_params)
            .or_else(|| messages.iter().find_map(unsupported_message))
            // Anthropic rejects a request whose first turn is not a user turn.
            // Python only repairs that under `litellm.modify_params`, which the
            // core cannot observe, so decline instead of guessing.
            .or_else(|| {
                (!build_conversation(messages).opens_on_user_turn())
                    .then_some(Unsupported("conversation does not open on a user turn"))
            })
    }
}

fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

fn anthropic_body(
    model: &str,
    conversation: &Conversation,
    optional_params: Map<String, Value>,
) -> Value {
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
        .chain(optional_params),
    );
    Value::Object(body)
}
