use litellm_auth::ConnectionArguments;
use litellm_auth::SecretValue;
use litellm_core_utils::{
    core_helpers::{finish_reason_for, unix_now, usage_from_parts},
    prompt_templates::factory::{Conversation, build_conversation},
};
use litellm_llms_types::formats::chat_completions::{
    ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse, ChatMessage,
};
use serde::Deserialize;
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
        auth::AuthScheme,
        chat::{
            streaming::{ChatStream, StreamShape},
            transformation::{
                BaseConfig, Headers, ProviderChatRequestData, ProviderChatResponseData,
                ValidatedEnvironment, reject_stream, validate_message,
            },
        },
        messages::streaming::anthropic_sse_event_stream,
    },
};

#[derive(Deserialize)]
struct TextResponseProjection {
    model: String,
    content: Vec<TextResponseBlock>,
    usage: ResponseUsageProjection,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TextResponseBlock {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct ResponseUsageProjection {
    input_tokens: u64,
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
}

pub struct AnthropicConfig;

pub const ANTHROPIC_CHAT_COMPLETIONS_CONFIG: AnthropicConfig = AnthropicConfig;

impl BaseConfig for AnthropicConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        use crate::anthropic::common_utils::{
            ANTHROPIC_API_BASE_ENV, ANTHROPIC_API_KEY_ENV, ANTHROPIC_BASE_URL_ENV,
        };
        vec![
            ANTHROPIC_API_KEY_ENV,
            ANTHROPIC_API_BASE_ENV,
            ANTHROPIC_BASE_URL_ENV,
        ]
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _connection: &ConnectionArguments,
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
        let body: TextResponseProjection =
            serde_json::from_value(response.body).map_err(|error| {
                Error::InvalidResponse(crate::ErrorDetail::invalid("messages response", error))
            })?;
        if body
            .content
            .iter()
            .any(|block| matches!(block, TextResponseBlock::Other))
        {
            return Err(Error::Unsupported("non-text response content block"));
        }
        let text: String = body
            .content
            .into_iter()
            .map(|block| match block {
                TextResponseBlock::Text { text } => text,
                TextResponseBlock::Other => String::new(),
            })
            .collect();

        Ok(ChatCompletionsResponse {
            created: unix_now(),
            model: body.model,
            choices: vec![ChatCompletionsChoice {
                index: 0,
                message: ChatCompletionsChoiceMessage {
                    role: "assistant".to_string(),
                    content: (!text.is_empty()).then_some(text),
                },
                finish_reason: finish_reason_for(body.stop_reason.as_deref().unwrap_or(""))
                    .to_string(),
            }],
            usage: usage_from_parts(
                body.usage.input_tokens,
                body.usage.output_tokens,
                body.usage.cache_read_input_tokens.unwrap_or(0),
                body.usage.cache_creation_input_tokens.unwrap_or(0),
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
        _connection: &ConnectionArguments,
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
    fn validate_request(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Result<(), Error> {
        reject_stream(optional_params)?;
        messages.iter().try_for_each(validate_message)?;
        if !build_conversation(messages).opens_on_user_turn() {
            return Err(Error::Unsupported(
                "conversation does not open on a user turn",
            ));
        }
        Ok(())
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
