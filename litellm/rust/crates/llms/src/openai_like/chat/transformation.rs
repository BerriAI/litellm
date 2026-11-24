//! `litellm/llms/openai_like/chat/transformation.py`: the chat config every
//! OpenAI-compatible endpoint shares. The body is already OpenAI-shaped, so
//! parameters pass through verbatim; the port keeps Python's two deviations,
//! the `max_completion_tokens` -> `max_tokens` rename and the usage
//! `*_tokens` null-to-zero sanitize. Usage then goes through Python's `Usage` constructor.

use litellm_auth::CredentialPlacement;
use litellm_secret_value::SecretValue;

use litellm_core_utils::{
    core_helpers::unix_now, decline::Decline, prompt_templates::factory::THINKING_BLOCKS_FIELD,
};
use litellm_llms_types::formats::chat_completions::{
    CacheCreationTokenDetails, ChatCompletionsChoice, ChatCompletionsChoiceMessage,
    ChatCompletionsResponse, ChatCompletionsUsage, ChatMessage, CompletionTokensDetails,
    PromptTokensDetails,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::{
        auth::AuthScheme,
        chat::transformation::{
            BaseConfig, Headers, ProviderChatRequestData, ProviderChatResponseData,
            ValidatedEnvironment, unsupported_message_content, unsupported_message_fields,
            unsupported_param,
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

const REASONING_FIELDS: &[&str] = &[THINKING_BLOCKS_FIELD, "reasoning_content"];

const USAGE_COUNTS: [&str; 3] = ["prompt_tokens", "completion_tokens", "total_tokens"];

pub struct OpenAILikeChatConfig;

pub const OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG: OpenAILikeChatConfig = OpenAILikeChatConfig;

impl BaseConfig for OpenAILikeChatConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        vec!["OPENAI_LIKE_API_KEY", "OPENAI_LIKE_API_BASE"]
    }

    fn supported_openai_param_mappings(&self) -> &'static [(&'static str, &'static str)] {
        SUPPORTED_PARAMS
    }

    fn unsupported_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Decline> {
        unsupported_param(SUPPORTED_PARAMS, CONFIG_PARAMS, optional_params).or_else(|| {
            messages.iter().find_map(|message| {
                unsupported_message_fields(message, REASONING_FIELDS)
                    .or_else(|| unsupported_message_content(message.content.as_ref()))
            })
        })
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
                (
                    "messages".to_string(),
                    json!(
                        messages
                            .into_iter()
                            .map(without_null_fields)
                            .collect::<Vec<_>>()
                    ),
                ),
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
        let Value::Object(mut body) = response.body else {
            return Err(Error::InvalidResponse(crate::ErrorDetail::Expected {
                subject: "chat response",
                expected: "an object",
            }));
        };

        let choices = body
            .get("choices")
            .and_then(Value::as_array)
            .ok_or(Error::MissingField("choices"))?
            .iter()
            .enumerate()
            .map(|(position, choice)| normalize_choice(position, choice))
            .collect::<Result<Vec<_>, _>>()?;

        let usage = match body.remove("usage") {
            Some(Value::Object(usage)) => python_usage(host_usage(sanitize_usage(usage))),
            _ => ChatCompletionsUsage::default(),
        };

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
            usage,
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

    fn caller_config_params(&self) -> &'static [&'static str] {
        CONFIG_PARAMS
    }
}

fn without_null_fields(message: ChatMessage) -> ChatMessage {
    ChatMessage {
        extra: message
            .extra
            .into_iter()
            .filter(|(_, value)| !value.is_null())
            .collect(),
        ..message
    }
}

/// `OpenAILikeChatConfig._sanitize_usage_obj`: a provider that reports a null
/// `*_tokens` entry breaks OpenAI clients, so nulls become 0. Python scrubs
/// every top-level usage key ending in `_tokens`, and its `Usage` reads a missing count as 0.
fn sanitize_usage(usage: Map<String, Value>) -> Map<String, Value> {
    let missing_counts: Vec<&str> = USAGE_COUNTS
        .into_iter()
        .filter(|count| !usage.contains_key(*count))
        .collect();
    usage
        .into_iter()
        .map(|(key, value)| match value {
            Value::Null if key.ends_with("_tokens") => (key, json!(0)),
            value => (key, value),
        })
        .chain(
            missing_counts
                .into_iter()
                .map(|count| (count.to_string(), json!(0))),
        )
        .collect()
}

/// Keeps every field that parses on its own, reads a count sent as a numeric string or a whole
/// float as that number, and reads a top-level count that does not parse as 0.
fn host_usage(usage: Map<String, Value>) -> ChatCompletionsUsage {
    let counts: Map<String, Value> = USAGE_COUNTS
        .into_iter()
        .map(|count| (count.to_string(), json!(token_count(usage.get(count)))))
        .collect();
    let parses = |fields: &Map<String, Value>| {
        ChatCompletionsUsage::deserialize(&Value::Object(fields.clone()))
    };
    let normalized: Map<String, Value> = usage
        .into_iter()
        .filter_map(|(key, value)| match value {
            Value::Object(details) if DETAILS.contains(&key.as_str()) => {
                (!details.is_empty()).then(|| (key, Value::Object(readable_details(details))))
            }
            value => Some((key, value)),
        })
        .chain(counts.clone())
        .collect();
    parses(&normalized).unwrap_or_else(|_| {
        let readable: Map<String, Value> = normalized
            .into_iter()
            .filter(|(key, value)| {
                parses(
                    &counts
                        .clone()
                        .into_iter()
                        .chain([(key.clone(), value.clone())])
                        .collect(),
                )
                .is_ok()
            })
            .collect();
        parses(&readable).unwrap_or_default()
    })
}

const DETAILS: [&str; 2] = ["prompt_tokens_details", "completion_tokens_details"];

/// Python drops a null detail and reads a `*_tokens` detail the way it reads a top-level count,
/// where it fails on a detail that does not parse and this drops only that detail.
fn readable_details(details: Map<String, Value>) -> Map<String, Value> {
    details
        .into_iter()
        .filter_map(|(key, value)| match value {
            Value::Null => None,
            Value::Object(nested) if key == "cache_creation_token_details" => {
                Some((key, Value::Object(readable_details(nested))))
            }
            _ if key == "cache_creation_token_details" => None,
            value if key.ends_with("_tokens") => {
                whole_number(&value).map(|count| (key, json!(count)))
            }
            value => Some((key, value)),
        })
        .collect()
}

fn token_count(count: Option<&Value>) -> u64 {
    count.and_then(whole_number).unwrap_or(0)
}

fn whole_number(count: &Value) -> Option<u64> {
    let whole = |number: f64| (number.fract() == 0.0 && number >= 0.0).then_some(number as u64);
    match count {
        Value::Number(number) => number.as_u64().or_else(|| whole(number.as_f64()?)),
        Value::String(text) => text.parse().ok().or_else(|| whole(text.parse().ok()?)),
        _ => None,
    }
}

/// Python's `Usage` constructor: top-level cache and reasoning counts move into the details
/// objects and the details' cache write aliases agree.
fn python_usage(usage: ChatCompletionsUsage) -> ChatCompletionsUsage {
    let top_level_count = |key: &str| usage.extra.get(key).and_then(Value::as_u64);
    let reasoning_tokens = top_level_count("reasoning_tokens").filter(|tokens| *tokens > 0);
    let cached_tokens = usage
        .cache_read_input_tokens
        .or(top_level_count("prompt_cache_hit_tokens"));
    ChatCompletionsUsage {
        completion_tokens_details: completion_details(
            usage.completion_tokens_details,
            reasoning_tokens,
            usage.completion_tokens,
        ),
        prompt_tokens_details: prompt_details(
            usage.prompt_tokens_details,
            cached_tokens,
            usage.cache_creation_input_tokens,
        ),
        extra: usage
            .extra
            .into_iter()
            .filter(|(key, _)| key != "reasoning_tokens")
            .collect(),
        ..usage
    }
}

fn completion_details(
    details: Option<CompletionTokensDetails>,
    reasoning_tokens: Option<u64>,
    completion_tokens: u64,
) -> Option<CompletionTokensDetails> {
    let Some(reasoning_tokens) = reasoning_tokens else {
        return details;
    };
    let details = details.unwrap_or_default();
    let modality_tokens = |key: &str| details.extra.get(key).and_then(Value::as_u64).unwrap_or(0);
    let text_tokens = completion_tokens
        .saturating_sub(reasoning_tokens)
        .saturating_sub(modality_tokens("image_tokens"))
        .saturating_sub(modality_tokens("audio_tokens"));
    Some(CompletionTokensDetails {
        reasoning_tokens: details.reasoning_tokens.or(Some(reasoning_tokens)),
        text_tokens: details.text_tokens.or(Some(text_tokens)),
        ..details
    })
}

fn prompt_details(
    details: Option<PromptTokensDetails>,
    cached_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
) -> Option<PromptTokensDetails> {
    if details.is_none() && cached_tokens.is_none() && cache_creation_input_tokens.is_none() {
        return None;
    }
    let details = details.unwrap_or_default();
    let cache_write_tokens = cache_creation_input_tokens
        .or(details.cache_write_tokens)
        .or(details.cache_creation_tokens)
        .or_else(|| {
            details
                .extra
                .get("cache_creation_input_tokens")
                .and_then(Value::as_u64)
        });
    Some(PromptTokensDetails {
        cached_tokens: cached_tokens.or(details.cached_tokens),
        cache_write_tokens,
        cache_creation_tokens: cache_write_tokens,
        cache_creation_token_details: details.cache_creation_token_details.or_else(|| {
            details
                .extra
                .get("cache_creation")
                .and_then(|nested| CacheCreationTokenDetails::deserialize(nested).ok())
        }),
        ..details
    })
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
        return Err(Error::InvalidResponse(crate::ErrorDetail::Unhandled(
            "tool call response",
        )));
    }
    if message.get("refusal").is_some_and(|value| !value.is_null()) {
        return Err(Error::InvalidResponse(crate::ErrorDetail::Unhandled(
            "refusal response",
        )));
    }
    let content = message.get("content");
    if content.is_some_and(|value| !value.is_null() && !value.is_string()) {
        return Err(Error::InvalidResponse(crate::ErrorDetail::Unhandled(
            "non-text response content",
        )));
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
            reasoning_content: None,
            thinking_blocks: None,
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
            chat::transformation::{BaseConfig, DropParams, ProviderChatResponseData},
        },
        openai_like::chat::transformation::OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
    };
    use litellm_core_utils::decline::Decline;
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

    fn reason(msgs: Value, opts: Value) -> Option<Decline> {
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
    fn keeps_call_configuration_a_caller_sends() {
        let config =
            json!({"custom_endpoint": true, "extra_headers": {"x": "y"}, "max_retries": 2});
        assert_eq!(
            OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG.map_openai_params(
                "my-model",
                params(config.clone()),
                DropParams::Off
            ),
            Ok(params(config))
        );
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
            OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG.get_complete_url(
                None,
                "my-model",
                &params(json!({})),
                &no_env
            ),
            Err(Error::InvalidRequest(crate::ErrorDetail::MissingApiBase))
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

    fn usage_of(usage: Option<Value>) -> Value {
        let choices =
            json!([{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]);
        let body = match usage {
            Some(usage) => json!({"model": "m", "choices": choices, "usage": usage}),
            None => json!({"model": "m", "choices": choices}),
        };
        let response = transform_response(body).expect("response normalizes");
        serde_json::to_value(response.usage).unwrap()
    }

    #[rstest]
    #[case::gpt_5_4_mini_capture(json!({
        "prompt_tokens": 5726, "completion_tokens": 4, "total_tokens": 5730,
        "prompt_tokens_details": {"cached_tokens": 5504, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0
        }
    }))]
    #[case::no_text_tokens_reported(json!({
        "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8,
        "prompt_tokens_details": {"cached_tokens": 1}
    }))]
    #[case::host_extension(json!({
        "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8,
        "prompt_cache_miss_tokens": 2, "cost": 0.25
    }))]
    fn passes_the_host_usage_through_like_python(#[case] usage: Value) {
        assert_eq!(usage_of(Some(usage.clone())), usage);
    }

    #[rstest]
    #[case::deepseek_cache_hit(
        json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "prompt_cache_hit_tokens": 1, "cost": 0.25}),
        json!({
            "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "prompt_cache_hit_tokens": 1, "cost": 0.25,
            "prompt_tokens_details": {"cached_tokens": 1}
        })
    )]
    #[case::anthropic_cache_counts(
        json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "cache_read_input_tokens": 6, "cache_creation_input_tokens": 2}),
        json!({
            "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "cache_read_input_tokens": 6, "cache_creation_input_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 6, "cache_write_tokens": 2, "cache_creation_tokens": 2}
        })
    )]
    #[case::top_level_cache_read_wins(
        json!({
            "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "prompt_cache_hit_tokens": 1, "cache_read_input_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 2, "audio_tokens": 0}
        }),
        json!({
            "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "prompt_cache_hit_tokens": 1, "cache_read_input_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 6, "audio_tokens": 0}
        })
    )]
    #[case::top_level_reasoning(
        json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "reasoning_tokens": 2}),
        json!({
            "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 2, "text_tokens": 3}
        })
    )]
    #[case::reasoning_text_leaves_out_audio(
        json!({
            "prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12, "reasoning_tokens": 2,
            "completion_tokens_details": {"audio_tokens": 3, "reasoning_tokens": 4}
        }),
        json!({
            "prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12,
            "completion_tokens_details": {"audio_tokens": 3, "reasoning_tokens": 4, "text_tokens": 4}
        })
    )]
    #[case::zero_reasoning(
        json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "reasoning_tokens": 0}),
        json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8})
    )]
    #[case::nested_cache_creation(
        json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "prompt_tokens_details": {"cache_creation_tokens": 2}}),
        json!({
            "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8,
            "prompt_tokens_details": {"cache_write_tokens": 2, "cache_creation_tokens": 2}
        })
    )]
    #[case::nested_anthropic_cache_creation(
        json!({
            "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8,
            "prompt_tokens_details": {
                "cache_creation_input_tokens": 2, "cache_creation": {"ephemeral_5m_input_tokens": 1, "ephemeral_30m_input_tokens": 1}
            }
        }),
        json!({
            "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8,
            "prompt_tokens_details": {
                "cache_write_tokens": 2, "cache_creation_tokens": 2,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 1, "ephemeral_30m_input_tokens": 1},
                "cache_creation_input_tokens": 2, "cache_creation": {"ephemeral_5m_input_tokens": 1, "ephemeral_30m_input_tokens": 1}
            }
        })
    )]
    #[case::empty_details(
        json!({
            "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8,
            "prompt_tokens_details": {}, "completion_tokens_details": {}
        }),
        json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8})
    )]
    #[case::kimi_k3_capture(
        json!({
            "prompt_tokens": 105, "completion_tokens": 134, "total_tokens": 239,
            "completion_tokens_details": {"reasoning_tokens": 121}, "prompt_tokens_details": {"cache_write_tokens": 0}
        }),
        json!({
            "prompt_tokens": 105, "completion_tokens": 134, "total_tokens": 239,
            "completion_tokens_details": {"reasoning_tokens": 121},
            "prompt_tokens_details": {"cache_write_tokens": 0, "cache_creation_tokens": 0}
        })
    )]
    #[case::fireworks_deepseek_capture(
        json!({
            "prompt_tokens": 50, "completion_tokens": 38, "total_tokens": 88, "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {
                "accepted_prediction_tokens": null, "reasoning_tokens": 35, "rejected_prediction_tokens": null
            }
        }),
        json!({
            "prompt_tokens": 50, "completion_tokens": 38, "total_tokens": 88, "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 35}
        })
    )]
    fn maps_host_cache_and_reasoning_counts_like_python(
        #[case] usage: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(usage_of(Some(usage)), expected);
    }

    #[rstest]
    #[case::absent(None, json!({"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}))]
    #[case::null(Some(Value::Null), json!({"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}))]
    #[case::null_and_missing_counts(
        Some(json!({"prompt_tokens": 3, "completion_tokens": null})),
        json!({"completion_tokens": 0, "prompt_tokens": 3, "total_tokens": 0})
    )]
    fn reports_a_null_or_missing_count_as_zero_like_python(
        #[case] usage: Option<Value>,
        #[case] expected: Value,
    ) {
        assert_eq!(usage_of(usage), expected);
    }

    #[rstest]
    #[case::count_sent_as_a_numeric_string(
        json!({"prompt_tokens": "3", "completion_tokens": 1, "total_tokens": 4}),
        json!({"completion_tokens": 1, "prompt_tokens": 3, "total_tokens": 4})
    )]
    #[case::count_sent_as_a_whole_float(
        json!({"prompt_tokens": 3.0, "completion_tokens": "1.0", "total_tokens": 4}),
        json!({"completion_tokens": 1, "prompt_tokens": 3, "total_tokens": 4})
    )]
    #[case::detail_sent_as_a_numeric_string(
        json!({
            "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": "2", "cache_creation_token_details": {"ephemeral_5m_input_tokens": "4"}},
            "completion_tokens_details": {"reasoning_tokens": 1.0}
        }),
        json!({
            "completion_tokens": 1, "prompt_tokens": 3, "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 2, "cache_creation_token_details": {"ephemeral_5m_input_tokens": 4}},
            "completion_tokens_details": {"reasoning_tokens": 1}
        })
    )]
    #[case::null_detail(
        json!({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4, "completion_tokens_details": {"reasoning_tokens": null}}),
        json!({"completion_tokens": 1, "prompt_tokens": 3, "total_tokens": 4, "completion_tokens_details": {}})
    )]
    fn reads_a_count_the_way_python_coerces_it(#[case] usage: Value, #[case] expected: Value) {
        assert_eq!(usage_of(Some(usage)), expected);
    }

    #[rstest]
    #[case::count_that_is_not_a_token_count(
        json!({"prompt_tokens": "three", "completion_tokens": 1, "total_tokens": 4}),
        json!({"completion_tokens": 1, "prompt_tokens": 0, "total_tokens": 4})
    )]
    #[case::detail_that_is_not_a_token_count(
        json!({
            "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": "two", "text_tokens": 3}
        }),
        json!({"completion_tokens": 1, "prompt_tokens": 3, "total_tokens": 4, "prompt_tokens_details": {"text_tokens": 3}})
    )]
    #[case::nested_detail_that_is_not_a_token_count(
        json!({
            "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
            "prompt_tokens_details": {
                "cached_tokens": 2, "cache_creation_token_details": {"ephemeral_5m_input_tokens": "x", "ephemeral_1h_input_tokens": 1}
            }
        }),
        json!({
            "completion_tokens": 1, "prompt_tokens": 3, "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 2, "cache_creation_token_details": {"ephemeral_1h_input_tokens": 1}}
        })
    )]
    #[case::details_that_are_not_an_object(
        json!({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4, "completion_tokens_details": "none"}),
        json!({"completion_tokens": 1, "prompt_tokens": 3, "total_tokens": 4})
    )]
    #[case::unreadable_top_level_field_keeps_every_other_field(
        json!({
            "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4, "service_tier": 1, "cost": 0.25,
            "cache_read_input_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 1}
        }),
        json!({
            "completion_tokens": 1, "prompt_tokens": 3, "total_tokens": 4, "cost": 0.25, "cache_read_input_tokens": 2,
            "completion_tokens_details": {"reasoning_tokens": 1}, "prompt_tokens_details": {"cached_tokens": 2}
        })
    )]
    fn a_usage_outside_the_contract_keeps_what_parses_instead_of_failing(
        #[case] usage: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(usage_of(Some(usage)), expected);
    }

    #[rstest]
    fn a_tool_call_response_fails_instead_of_dropping_the_calls() {
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
            Err(Error::InvalidResponse(crate::ErrorDetail::Unhandled(
                "tool call response"
            )))
        );
    }

    #[rstest]
    fn a_refusal_fails_instead_of_returning_an_empty_reply() {
        assert_eq!(
            transform_response(json!({
                "model": "m",
                "choices": [{
                    "message": {"role": "assistant", "content": null, "refusal": "cannot help"},
                    "finish_reason": "stop",
                }],
            })),
            Err(Error::InvalidResponse(crate::ErrorDetail::Unhandled(
                "refusal response"
            )))
        );
    }

    #[rstest]
    fn a_non_text_response_content_is_invalid() {
        assert_eq!(
            transform_response(json!({
                "model": "m",
                "choices": [{
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                    "finish_reason": "stop",
                }],
            })),
            Err(Error::InvalidResponse(crate::ErrorDetail::Unhandled(
                "non-text response content"
            )))
        );
    }

    #[rstest]
    #[case::streaming(json!({"stream": true}), "streaming")]
    #[case::unrecognized_param(json!({"some_provider_knob": 1}), "unrecognized request parameter")]
    fn declines(#[case] opts: Value, #[case] expected: &'static str) {
        assert_eq!(
            reason(json!([{"role": "user", "content": "hi"}]), opts),
            Some(Decline::new(expected))
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
    #[case::thinking(json!({"type": "thinking", "thinking": "t", "signature": "sig"}))]
    #[case::redacted_thinking(json!({"type": "redacted_thinking", "data": "d"}))]
    fn sends_a_reasoning_reply_from_another_provider_on_verbatim(#[case] block: Value) {
        let conversation = json!([
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "t", "thinking_blocks": [block]},
            {"role": "user", "content": "add 1"},
        ]);
        assert_eq!(reason(conversation.clone(), json!({})), None);
        assert_eq!(
            transform("my-model", conversation.clone(), json!({}))["messages"],
            conversation
        );
    }

    #[rstest]
    fn drops_null_reasoning_fields_like_python() {
        let conversation = json!([
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": null, "thinking_blocks": null},
            {"role": "user", "content": "add 1"},
        ]);
        assert_eq!(reason(conversation.clone(), json!({})), None);
        assert_eq!(
            transform("my-model", conversation, json!({}))["messages"][1],
            json!({"role": "assistant", "content": "4"})
        );
    }

    #[rstest]
    fn other_unrecognized_message_fields_still_decline() {
        assert_eq!(
            reason(
                json!([
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "4", "thinking_blocks": [], "audio": {"id": "a"}},
                ]),
                json!({}),
            ),
            Some(Decline::new("unrecognized message field"))
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
            Some(Decline::new("unrecognized request parameter"))
        );
    }
}
