use litellm_auth::SecretValue;
use litellm_core_utils::{
    core_helpers::{finish_reason_for, unix_now, usage_from_parts},
    prompt_templates::factory::{Conversation, build_conversation},
};
use litellm_types::{
    llms::{anthropic::AnthropicVersion, openai::ChatMessage},
    utils::{ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse},
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
                Unsupported, ValidatedEnvironment, unsupported_message, unsupported_param,
            },
        },
        messages::streaming::anthropic_sse_event_stream,
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

#[derive(Deserialize)]
struct MessageResponse {
    model: String,
    content: Vec<ContentBlock>,
    usage: MessageUsage,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct MessageUsage {
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
        let body: MessageResponse = serde_json::from_value(response.body).map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::invalid("messages response", error))
        })?;
        // The route declines tool and thinking requests, so a non-text block
        // means the response carries something this path never asked for.
        // Decline rather than silently dropping it; the host falls back.
        if body
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Other))
        {
            return Err(Error::Unsupported("non-text response content block"));
        }
        let text: String = body
            .content
            .into_iter()
            .map(|block| match block {
                ContentBlock::Text { text } => text,
                ContentBlock::Other => String::new(),
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
        const HEADERS: &[(&str, &str)] = &[
            (
                "anthropic-version",
                AnthropicVersion::Version20230601.into_str(),
            ),
            ("content-type", "application/json"),
        ];
        HEADERS
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

#[cfg(test)]
mod tests {
    use super::ANTHROPIC_CHAT_COMPLETIONS_CONFIG;
    use crate::{
        Error,
        base_llm::{
            auth::AuthScheme,
            chat::transformation::{BaseConfig, ProviderChatResponseData, Unsupported},
        },
    };
    use litellm_types::llms::anthropic::AnthropicVersion;
    use litellm_types::{llms::openai::ChatMessage, utils::ChatCompletionsResponse};
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    fn messages(value: Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).expect("valid messages")
    }

    fn params(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("params must be an object, got {other}"),
        }
    }

    fn transform(model: &str, msgs: Value, opts: Value) -> Value {
        ANTHROPIC_CHAT_COMPLETIONS_CONFIG
            .transform_request(model, messages(msgs), params(opts))
            .expect("request transforms")
            .body
    }

    fn transform_response(body: Value) -> Result<ChatCompletionsResponse, Error> {
        ANTHROPIC_CHAT_COMPLETIONS_CONFIG
            .transform_response("claude-sonnet-4-5", ProviderChatResponseData { body })
    }

    fn reason(msgs: Value, opts: Value) -> Option<Unsupported> {
        ANTHROPIC_CHAT_COMPLETIONS_CONFIG.unsupported_reason(&messages(msgs), &params(opts))
    }

    #[rstest]
    fn builds_the_messages_body_python_builds() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}
            ]),
            json!({"max_tokens": 128, "temperature": 0.2}),
        );
        assert_eq!(
            body,
            json!({
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ],
                "system": [{"type": "text", "text": "be terse"}],
                "max_tokens": 128,
                "temperature": 0.2
            })
        );
    }

    #[rstest]
    fn omits_system_when_no_system_message_is_present() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 16}),
        );
        assert!(body.get("system").is_none());
    }

    #[rstest]
    fn merges_consecutive_turns_and_wraps_every_text_in_a_block() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([
                {"role": "user", "content": "one"},
                {"role": "user", "content": [{"type": "text", "text": "two"}]},
                {"role": "assistant", "content": "ack"}
            ]),
            json!({"max_tokens": 16}),
        );
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "ack"}]}
            ])
        );
    }

    #[rstest]
    fn right_strips_a_trailing_assistant_prefill_like_python() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Argentina  "}
            ]),
            json!({"max_tokens": 16}),
        );
        assert_eq!(
            body["messages"][1]["content"][0]["text"],
            json!("Argentina")
        );
    }

    #[rstest]
    fn passes_every_supported_param_through_untouched() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([{"role": "user", "content": "hi"}]),
            json!({
                "max_tokens": 64,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop_sequences": ["STOP"]
            }),
        );
        assert_eq!(body["max_tokens"], json!(64));
        assert_eq!(body["temperature"], json!(0.1));
        assert_eq!(body["top_p"], json!(0.9));
        assert_eq!(body["stop_sequences"], json!(["STOP"]));
    }

    #[rstest]
    fn declines_top_k_because_python_gates_it_by_model_below_this_point() {
        // `temperature` and `top_p` arrive already resolved, because
        // `map_openai_params` applies `_apply_sampling_param` to them before the
        // gate runs. `top_k` bypasses that and is gated inside `transform_request`,
        // the function this route replaces, so forwarding it would send `top_k` to
        // a model that removed sampling params and take a 400 after the call, where
        // Python drops it and succeeds.
        assert_eq!(
            reason(
                json!([{"role": "user", "content": "hi"}]),
                json!({"top_k": 40})
            ),
            Some(Unsupported("unrecognized request parameter"))
        );
    }

    #[rstest]
    fn declines_streaming_before_anything_else() {
        assert_eq!(
            reason(
                json!([{"role": "user", "content": "hi"}]),
                json!({"stream": true, "max_tokens": 16})
            ),
            Some(Unsupported("streaming"))
        );
    }

    #[rstest]
    fn accepts_an_explicit_stream_false() {
        assert_eq!(
            reason(
                json!([{"role": "user", "content": "hi"}]),
                json!({"stream": false, "max_tokens": 16})
            ),
            None
        );
    }

    #[rstest]
    #[case::tools(json!({"tools": []}))]
    #[case::tool_choice(json!({"tool_choice": {"type": "auto"}}))]
    #[case::thinking(json!({"thinking": {"type": "enabled"}}))]
    #[case::system(json!({"system": "injected"}))]
    #[case::metadata(json!({"metadata": {"user_id": "u1"}}))]
    #[case::output_config(json!({"output_config": {"effort": "high"}}))]
    fn declines_any_param_outside_the_allowlist(#[case] param: Value) {
        assert_eq!(
            reason(json!([{"role": "user", "content": "hi"}]), param.clone()),
            Some(Unsupported("unrecognized request parameter")),
            "expected {param} to decline"
        );
    }

    #[rstest]
    fn declines_tool_calls_tool_results_and_multimodal_content() {
        assert_eq!(
            reason(
                json!([
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": null, "tool_calls": [
                        {"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}
                    ]}
                ]),
                json!({})
            ),
            Some(Unsupported("unrecognized message field"))
        );
        assert_eq!(
            reason(
                json!([
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "tool_call_id": "c1", "content": "ok"}
                ]),
                json!({})
            ),
            Some(Unsupported("unrecognized message field"))
        );
        assert_eq!(
            reason(
                json!([
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
                ]}]),
                json!({})
            ),
            Some(Unsupported("non-text message content"))
        );
        assert_eq!(
            reason(
                json!([
                {"role": "user", "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]}]),
                json!({})
            ),
            Some(Unsupported("non-text message content"))
        );
    }

    #[rstest]
    fn declines_a_message_whose_content_list_is_empty() {
        // An empty list passes every per-part check, so without this it would reach
        // the provider as an empty `content` array and fail after the call rather
        // than declining to Python before it.
        assert_eq!(
            reason(json!([{"role": "user", "content": []}]), json!({})),
            Some(Unsupported("message without content"))
        );
        assert_eq!(
            reason(
                json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
                json!({})
            ),
            None
        );
    }

    #[rstest]
    fn declines_a_conversation_that_does_not_open_on_a_user_turn() {
        assert_eq!(
            reason(
                json!([
                    {"role": "system", "content": "be terse"},
                    {"role": "assistant", "content": "prefill"}
                ]),
                json!({})
            ),
            Some(Unsupported("conversation does not open on a user turn"))
        );
    }

    #[rstest]
    fn accepts_a_plain_text_conversation() {
        assert_eq!(
            reason(
                json!([
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": [{"type": "text", "text": "again"}]}
                ]),
                json!({"max_tokens": 16, "temperature": 0.5})
            ),
            None
        );
    }

    #[rstest]
    fn normalizes_a_text_response_into_openai_shape() {
        let response = transform_response(json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20260101",
            "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " there"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 11, "output_tokens": 4}
        }))
        .expect("response transforms");

        assert_eq!(response.model, "claude-sonnet-4-5-20260101");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[0].message.role, "assistant");
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("hello there")
        );
        assert_eq!(response.choices[0].finish_reason, "stop");
        assert_eq!(response.usage.prompt_tokens, 11);
        assert_eq!(response.usage.completion_tokens, 4);
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[rstest]
    fn folds_cache_tokens_into_prompt_tokens_like_python() {
        let response = transform_response(json!({
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 3
            }
        }))
        .expect("response transforms");
        assert_eq!(response.usage.prompt_tokens, 18);
        assert_eq!(response.usage.total_tokens, 20);
        assert_eq!(response.usage.prompt_tokens_details.cached_tokens, 5);
        assert_eq!(
            response.usage.prompt_tokens_details.cache_creation_tokens,
            3
        );
        assert_eq!(response.usage.prompt_tokens_details.text_tokens, 10);
    }

    #[rstest]
    fn maps_max_tokens_stop_reason_to_length() {
        let response = transform_response(json!({
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .expect("response transforms");
        assert_eq!(response.choices[0].finish_reason, "length");
    }

    #[rstest]
    fn a_refusal_returns_the_completion_python_returns() {
        // `refusal` is a stop_reason, not a content block type, so the content is
        // ordinary text and this normalizes rather than declining. Python maps it
        // to content_filter in _FINISH_REASON_MAP and returns the completion.
        let response = transform_response(json!({
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "I can't help with that."}],
            "stop_reason": "refusal",
            "usage": {"input_tokens": 9, "output_tokens": 6}
        }))
        .expect("a refusal still transforms");
        assert_eq!(response.choices[0].finish_reason, "content_filter");
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("I can't help with that.")
        );
    }

    #[rstest]
    fn reports_no_content_rather_than_an_empty_string() {
        let response = transform_response(json!({
            "model": "claude-sonnet-4-5",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 0}
        }))
        .expect("response transforms");
        assert_eq!(response.choices[0].message.content, None);
    }

    #[rstest]
    fn response_carries_no_id_so_python_keeps_its_chatcmpl_id() {
        let response = transform_response(json!({
            "id": "msg_should_not_leak",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .expect("response transforms");
        let value = serde_json::to_value(response).expect("serializable");
        assert!(
            value.get("id").is_none(),
            "the rust response must not carry an id, got {value}"
        );
    }

    #[rstest]
    fn declines_a_response_carrying_a_non_text_block() {
        let err = transform_response(json!({
            "model": "claude-sonnet-4-5",
            "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .expect_err("non-text block");
        assert_eq!(err, Error::Unsupported("non-text response content block"));
    }

    #[rstest::rstest]
    #[case::not_an_object(json!("nope"))]
    #[case::missing_content(json!({"model": "test-model", "usage": {"input_tokens": 1, "output_tokens": 1}}))]
    #[case::missing_usage(json!({"model": "test-model", "content": []}))]
    #[case::missing_model(json!({"content": [], "usage": {"input_tokens": 1, "output_tokens": 1}}))]
    fn errors_on_a_response_missing_required_fields(#[case] body: Value) {
        let error = transform_response(body).expect_err("invalid response");
        assert!(matches!(error, Error::InvalidResponse(_)));
        let source = std::iter::successors(Some(&error as &dyn std::error::Error), |error| {
            error.source()
        })
        .find_map(|error| error.downcast_ref::<serde_json::Error>())
        .expect("the JSON decoding source is preserved");
        assert_eq!(
            error.to_string(),
            format!("invalid response: invalid messages response: {source}")
        );
    }

    #[rstest]
    fn resolves_the_messages_url_and_x_api_key_auth() {
        let config = &ANTHROPIC_CHAT_COMPLETIONS_CONFIG;
        assert_eq!(
            config
                .get_complete_url(None, "claude-sonnet-4-5", &Map::new(), &|_| None)
                .expect("url builds"),
            "https://api.anthropic.com/v1/messages"
        );
        let validated = config
            .validate_environment(
                Vec::new(),
                Some("sk-x"),
                "claude-sonnet-4-5",
                &Map::new(),
                &|_| None,
            )
            .expect("auth resolves");
        assert!(matches!(
            validated.auth,
            AuthScheme::Credential {
                placement: litellm_auth::CredentialPlacement::Header("x-api-key"),
                ref secret
            } if secret.expose() == "sk-x"
        ));
        assert_eq!(
            config.default_headers(),
            &[
                (
                    "anthropic-version",
                    AnthropicVersion::Version20230601.into_str()
                ),
                ("content-type", "application/json"),
            ]
        );
    }
}
