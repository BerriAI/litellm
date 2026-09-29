use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::core_helpers::unix_now;
use litellm_llms_types::formats::chat_completions::{
    ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsRequest,
    ChatCompletionsResponse, ChatCompletionsUsage, ChatMessage, PromptTokensDetails,
};
use litellm_llms_types::recognized::Recognized;
use serde::Deserialize;
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

#[derive(Default, Deserialize)]
struct UsageProjection {
    prompt_tokens: Option<Recognized<u64>>,
    completion_tokens: Option<Recognized<u64>>,
    total_tokens: Option<Recognized<u64>>,
    prompt_tokens_details: Option<Recognized<PromptTokensProjection>>,
}

#[derive(Default, Deserialize)]
struct PromptTokensProjection {
    cached_tokens: Option<Recognized<u64>>,
    cache_creation_tokens: Option<Recognized<u64>>,
    cache_write_tokens: Option<Recognized<u64>>,
    text_tokens: Option<Recognized<u64>>,
}

fn token_count(value: Option<&Recognized<u64>>) -> u64 {
    value.and_then(Recognized::known).copied().unwrap_or(0)
}

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
        Ok(ProviderChatRequestData {
            body: json!(ChatCompletionsRequest {
                model: model.into(),
                messages,
                extra: params,
            }),
            stream_shape: Default::default(),
        })
    }

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error> {
        let body = response
            .body
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

        let usage = body
            .get("usage")
            .map(Recognized::<UsageProjection>::deserialize)
            .transpose()
            .map_err(|error| {
                Error::InvalidResponse(crate::ErrorDetail::invalid("Chat usage", error))
            })?;
        let empty_usage = UsageProjection::default();
        let usage = usage
            .as_ref()
            .and_then(Recognized::known)
            .unwrap_or(&empty_usage);
        let empty_details = PromptTokensProjection::default();
        let details = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(Recognized::known)
            .unwrap_or(&empty_details);

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
                prompt_tokens: token_count(usage.prompt_tokens.as_ref()),
                completion_tokens: token_count(usage.completion_tokens.as_ref()),
                total_tokens: token_count(usage.total_tokens.as_ref()),
                prompt_tokens_details: PromptTokensDetails {
                    cached_tokens: token_count(details.cached_tokens.as_ref()),
                    cache_creation_tokens: token_count(
                        details
                            .cache_creation_tokens
                            .as_ref()
                            .or(details.cache_write_tokens.as_ref()),
                    ),
                    text_tokens: token_count(details.text_tokens.as_ref()),
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
