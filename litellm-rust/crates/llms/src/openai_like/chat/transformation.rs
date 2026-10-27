//! `litellm/llms/openai_like/chat/transformation.py`: the chat config every
//! OpenAI-compatible endpoint shares. The body is already OpenAI-shaped, so
//! parameters pass through verbatim; the port keeps Python's two deviations,
//! the `max_completion_tokens` -> `max_tokens` rename and the usage
//! `*_tokens` null-to-zero sanitize.

use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::core_helpers::unix_now;
use litellm_llms_types::formats::chat_completions::{
    ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse,
    ChatCompletionsUsage, ChatMessage, PromptTokensDetails,
};
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::{
        auth::AuthScheme,
        chat::transformation::{
            BaseConfig, Headers, ProviderChatRequestData, ProviderChatResponseData,
            ValidatedEnvironment,
        },
    },
    openai_like::common_utils::{complete_openai_like_url, openai_compatible_provider_info},
};

/// OpenAI parameter names the Rust path can place verbatim in the request body.
/// Tool parameters are absent on purpose: the message gate already declines
/// tool-call content, and a `tools` request that did get through would produce
/// a tool-call response this port cannot normalize yet, so it declines before
/// the call instead of after it.
const SUPPORTED_PARAMS: &[(&str, &str)] = &[
    ("frequency_penalty", "frequency_penalty"),
    ("logit_bias", "logit_bias"),
    ("logprobs", "logprobs"),
    ("top_logprobs", "top_logprobs"),
    ("max_tokens", "max_tokens"),
    ("max_completion_tokens", "max_completion_tokens"),
    ("modalities", "modalities"),
    ("prediction", "prediction"),
    ("n", "n"),
    ("presence_penalty", "presence_penalty"),
    ("seed", "seed"),
    ("stop", "stop"),
    ("stream_options", "stream_options"),
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("audio", "audio"),
    ("web_search_options", "web_search_options"),
    ("service_tier", "service_tier"),
    ("safety_identifier", "safety_identifier"),
    ("prompt_cache_key", "prompt_cache_key"),
    ("prompt_cache_retention", "prompt_cache_retention"),
    ("store", "store"),
    ("response_format", "response_format"),
];

/// Call configuration the caller may pass that never enters the request body.
const CONFIG_PARAMS: &[&str] = &["custom_endpoint", "extra_headers", "max_retries"];

pub struct OpenAILikeChatConfig;

pub const OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG: OpenAILikeChatConfig = OpenAILikeChatConfig;

impl BaseConfig for OpenAILikeChatConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        vec!["OPENAI_LIKE_API_KEY", "OPENAI_LIKE_API_BASE"]
    }

    fn supported_openai_param_mappings(&self) -> &'static [(&'static str, &'static str)] {
        SUPPORTED_PARAMS
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let custom_endpoint = optional_params
            .get("custom_endpoint")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        complete_openai_like_url(api_base, custom_endpoint, env_lookup)
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> Result<ProviderChatRequestData, Error> {
        let mut params = Map::from_iter(
            optional_params
                .into_iter()
                .filter(|(key, _)| !CONFIG_PARAMS.contains(&key.as_str())),
        );
        // Most OpenAI-compatible endpoints take `max_tokens`, not
        // `max_completion_tokens`, so Python's `map_openai_params` renames it
        // and lets it overwrite a `max_tokens` the caller also sent.
        if let Some(limit) = params.remove("max_completion_tokens") {
            params.insert("max_tokens".to_string(), limit);
        }
        let body = Map::from_iter(
            [
                ("model".to_string(), json!(model)),
                ("messages".to_string(), json!(messages)),
            ]
            .into_iter()
            .chain(params),
        );
        Ok(ProviderChatRequestData {
            body: Value::Object(body),
            stream_shape: Default::default(),
        })
    }

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error> {
        let mut body = response.body;
        sanitize_usage(&mut body);
        let body = body
            .as_object()
            .ok_or_else(|| Error::InvalidResponse("chat response is not an object".into()))?;

        let choices = body
            .get("choices")
            .and_then(Value::as_array)
            .ok_or(Error::MissingField("choices"))?
            .iter()
            .enumerate()
            .map(|(position, choice)| normalize_choice(position, choice))
            .collect::<Result<Vec<_>, _>>()?;

        let usage = body.get("usage").and_then(Value::as_object);
        let field = |name: &str| {
            usage
                .and_then(|usage| usage.get(name))
                .and_then(Value::as_u64)
                .unwrap_or(0)
        };
        let details = usage.and_then(|usage| usage.get("prompt_tokens_details"));

        Ok(ChatCompletionsResponse {
            created: body
                .get("created")
                .and_then(Value::as_u64)
                .unwrap_or_else(unix_now),
            model: body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(model)
                .to_string(),
            choices,
            usage: ChatCompletionsUsage {
                prompt_tokens: field("prompt_tokens"),
                completion_tokens: field("completion_tokens"),
                total_tokens: field("total_tokens"),
                prompt_tokens_details: PromptTokensDetails {
                    cached_tokens: details
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                    cache_creation_tokens: details
                        .and_then(|d| d.get("cache_creation_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                    text_tokens: details
                        .and_then(|d| d.get("text_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                },
            },
        })
    }

    /// `OpenAILikeBase._validate_environment`: a forwarded `authorization` is
    /// the whole credential, and any other call authenticates with the
    /// resolved key as a bearer. The key resolves to `""` when neither the
    /// deployment nor `OPENAI_LIKE_API_KEY` sets one, because vllm-compatible
    /// endpoints take no key; Python still sends `Bearer ` in that case.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let (_, key) = openai_compatible_provider_info(None, api_key, env_lookup);
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(key.unwrap_or_default()),
            },
        })
    }

    fn config_params(&self) -> &'static [&'static str] {
        CONFIG_PARAMS
    }
}

/// `OpenAILikeChatConfig._sanitize_usage_obj`: a provider that reports a null
/// `*_tokens` entry breaks OpenAI clients, so nulls become 0. Python scrubs
/// every top-level usage key ending in `_tokens`.
fn sanitize_usage(body: &mut Value) {
    if let Some(usage) = body.get_mut("usage").and_then(Value::as_object_mut) {
        for (key, value) in usage.iter_mut() {
            if key.ends_with("_tokens") && value.is_null() {
                *value = json!(0);
            }
        }
    }
}

fn normalize_choice(position: usize, choice: &Value) -> Result<ChatCompletionsChoice, Error> {
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or(Error::MissingField("message"))?;
    if message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty())
    {
        // Python rewrites the lone tool call into content only under
        // `json_mode`, a request flag `transform_response` cannot see, and the
        // normalized type cannot carry tool calls at all. Declining is
        // terminal at this point, but passing back an empty assistant turn
        // would fabricate the reply.
        return Err(Error::Unsupported("tool call response"));
    }
    if message.get("refusal").is_some_and(|value| !value.is_null()) {
        return Err(Error::Unsupported("refusal response"));
    }
    let content = message.get("content");
    if content.is_some_and(|value| !value.is_null() && !value.is_string()) {
        return Err(Error::Unsupported("non-text response content"));
    }
    Ok(ChatCompletionsChoice {
        index: choice
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(position as u64),
        message: ChatCompletionsChoiceMessage {
            role: message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("assistant")
                .to_string(),
            content: content.and_then(Value::as_str).map(str::to_string),
        },
        finish_reason: choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        Error,
        base_llm::{
            auth::AuthScheme,
            chat::transformation::{BaseConfig, ProviderChatResponseData, Unsupported},
        },
        openai_like::chat::transformation::OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
    };
    use litellm_llms_types::formats::chat_completions::{ChatCompletionsResponse, ChatMessage};
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

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env_with<'a>(name: &'a str, value: &'a str) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| (key == name).then(|| value.to_string())
    }

    fn transform(model: &str, msgs: Value, opts: Value) -> Value {
        OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
            .transform_request(model, messages(msgs), params(opts))
            .expect("request transforms")
            .body
    }

    fn transform_response(body: Value) -> Result<ChatCompletionsResponse, Error> {
        OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
            .transform_response("some-model", ProviderChatResponseData { body })
    }

    fn reason(msgs: Value, opts: Value) -> Option<Unsupported> {
        OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG.unsupported_reason(&messages(msgs), &params(opts))
    }

    #[rstest]
    fn builds_the_openai_shaped_body() {
        let body = transform(
            "my-model",
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ]),
            json!({"temperature": 0.5, "max_tokens": 8}),
        );
        assert_eq!(body["model"], json!("my-model"));
        assert_eq!(
            body["messages"],
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ])
        );
        assert_eq!(body["temperature"], json!(0.5));
        assert_eq!(body["max_tokens"], json!(8));
    }

    #[rstest]
    fn renames_max_completion_tokens_to_max_tokens() {
        // `OpenAILikeChatConfig.map_openai_params`: most OpenAI-compatible providers
        // support `max_tokens`, not `max_completion_tokens`.
        let body = transform(
            "my-model",
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_completion_tokens": 12}),
        );
        assert_eq!(body["max_tokens"], json!(12));
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[rstest]
    fn max_completion_tokens_wins_when_both_limits_are_sent() {
        // Python assigns `max_tokens = max_completion_tokens` after copying the
        // params, so the renamed value outranks a caller-supplied `max_tokens`.
        let body = transform(
            "my-model",
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 8, "max_completion_tokens": 12}),
        );
        assert_eq!(body["max_tokens"], json!(12));
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[rstest]
    fn call_configuration_never_enters_the_body() {
        let body = transform(
            "my-model",
            json!([{"role": "user", "content": "hi"}]),
            json!({"custom_endpoint": true, "extra_headers": {"x": "y"}, "max_retries": 2}),
        );
        assert_eq!(
            body.as_object().unwrap().keys().collect::<Vec<_>>(),
            vec!["model", "messages"]
        );
    }

    #[rstest]
    #[case::appends_the_chat_completions_suffix("https://vllm.example.com/v1", json!({}), "https://vllm.example.com/v1/chat/completions")]
    #[case::trims_a_trailing_slash("https://vllm.example.com/v1/", json!({}), "https://vllm.example.com/v1/chat/completions")]
    #[case::a_custom_endpoint_is_used_as_is("https://vllm.example.com/v1/chat/completions", json!({"custom_endpoint": true}), "https://vllm.example.com/v1/chat/completions")]
    fn complete_url(#[case] api_base: &str, #[case] opts: Value, #[case] expected: &str) {
        assert_eq!(
            OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
                .get_complete_url(Some(api_base), "my-model", &params(opts), &no_env)
                .expect("url resolves"),
            expected
        );
    }

    #[rstest]
    fn api_base_falls_back_to_the_environment() {
        assert_eq!(
            OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
                .get_complete_url(
                    None,
                    "my-model",
                    &params(json!({})),
                    &env_with("OPENAI_LIKE_API_BASE", "https://env.example.com/v1"),
                )
                .expect("url resolves"),
            "https://env.example.com/v1/chat/completions"
        );
    }

    #[rstest]
    fn a_missing_api_base_is_an_error() {
        assert!(matches!(
            OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
                .get_complete_url(None, "my-model", &params(json!({})), &no_env),
            Err(Error::InvalidRequest(message)) if message.to_string().starts_with("Missing API Base")
        ));
    }

    #[rstest]
    fn the_resolved_key_authenticates_as_a_bearer() {
        let validated = OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
            .validate_environment(
                vec![],
                Some("sk-test"),
                "my-model",
                &params(json!({})),
                &no_env,
            )
            .expect("validates");
        assert!(matches!(
            validated.auth,
            AuthScheme::Credential {
                placement: litellm_auth::CredentialPlacement::Bearer,
                ..
            }
        ));
    }

    #[rstest]
    fn the_key_falls_back_to_the_environment() {
        let validated = OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
            .validate_environment(
                vec![],
                None,
                "my-model",
                &params(json!({})),
                &env_with("OPENAI_LIKE_API_KEY", "sk-env"),
            )
            .expect("validates");
        let AuthScheme::Credential { secret, .. } = validated.auth else {
            panic!("expected a bearer credential");
        };
        assert_eq!(secret.expose(), "sk-env");
    }

    #[rstest]
    fn a_forwarded_authorization_is_the_whole_credential() {
        // Python adds `Bearer <key>` only when the caller did not already send
        // `Authorization`, so the forwarded header wins over the deployment key.
        let headers = vec![("Authorization".to_string(), "Bearer caller".to_string())];
        let validated = OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
            .validate_environment(
                headers,
                Some("sk-test"),
                "my-model",
                &params(json!({})),
                &no_env,
            )
            .expect("validates");
        assert!(matches!(validated.auth, AuthScheme::Forwarded));
    }

    #[rstest]
    fn keyless_calls_still_validate_for_endpoints_that_take_no_key() {
        // vllm-compatible endpoints require no api key; Python resolves `""` and
        // sends `Bearer `, so validation must not fail on the missing key.
        let validated = OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG
            .validate_environment(vec![], None, "my-model", &params(json!({})), &no_env)
            .expect("validates");
        let AuthScheme::Credential { secret, .. } = validated.auth else {
            panic!("expected a bearer credential");
        };
        assert_eq!(secret.expose(), "");
    }

    #[rstest]
    fn normalizes_an_openai_response() {
        let response = transform_response(json!({
            "created": 1_700_000_000,
            "model": "served-model-name",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }))
        .expect("response normalizes");
        assert_eq!(response.created, 1_700_000_000);
        assert_eq!(response.model, "served-model-name");
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("hello")
        );
        assert_eq!(response.choices[0].finish_reason, "stop");
        assert_eq!(response.usage.prompt_tokens, 3);
        assert_eq!(response.usage.completion_tokens, 5);
        assert_eq!(response.usage.total_tokens, 8);
    }

    #[rstest]
    fn null_token_fields_in_usage_become_zero() {
        // `_sanitize_usage_obj`: providers that return null token values break
        // OpenAI clients, so the response is scrubbed at the source.
        let response = transform_response(json!({
            "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": null, "total_tokens": null},
        }))
        .expect("response normalizes");
        assert_eq!(response.usage.completion_tokens, 0);
        assert_eq!(response.usage.total_tokens, 0);
        assert_eq!(response.usage.prompt_tokens, 3);
    }

    #[rstest]
    fn a_tool_call_response_declines_instead_of_dropping_the_calls() {
        // The `json_mode` rewrite needs a request flag the route does not carry, so
        // a tool-call answer falls back to Python rather than losing the calls.
        assert_eq!(
            transform_response(json!({
                "model": "m",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            })),
            Err(Error::Unsupported("tool call response"))
        );
    }

    #[rstest]
    fn a_refusal_declines_instead_of_returning_an_empty_reply() {
        assert_eq!(
            transform_response(json!({
                "model": "m",
                "choices": [{
                    "message": {"role": "assistant", "content": null, "refusal": "cannot help"},
                    "finish_reason": "stop",
                }],
            })),
            Err(Error::Unsupported("refusal response"))
        );
    }

    #[rstest]
    fn a_non_text_response_content_declines() {
        assert_eq!(
            transform_response(json!({
                "model": "m",
                "choices": [{
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                    "finish_reason": "stop",
                }],
            })),
            Err(Error::Unsupported("non-text response content"))
        );
    }

    #[rstest]
    #[case::streaming(json!({"stream": true}), "streaming")]
    #[case::unrecognized_param(json!({"some_provider_knob": 1}), "unrecognized request parameter")]
    fn declines(#[case] opts: Value, #[case] expected: &'static str) {
        assert_eq!(
            reason(json!([{"role": "user", "content": "hi"}]), opts),
            Some(Unsupported(expected))
        );
    }

    #[rstest]
    fn accepts_standard_openai_params() {
        assert_eq!(
            reason(
                json!([{"role": "user", "content": "hi"}]),
                json!({
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 16,
                    "response_format": {"type": "json_object"},
                    "custom_endpoint": true,
                }),
            ),
            None
        );
    }

    #[rstest]
    fn tool_parameters_decline_before_the_call() {
        // A `tools` request would come back with tool calls this port cannot
        // normalize, so it declines at the gate instead of after the call.
        assert_eq!(
            reason(
                json!([{"role": "user", "content": "hi"}]),
                json!({"tools": [{"type": "function", "function": {"name": "f"}}]}),
            ),
            Some(Unsupported("unrecognized request parameter"))
        );
    }
}
