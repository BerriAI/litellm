//! `litellm/llms/openai_like/chat/transformation.py`: the chat config every
//! OpenAI-compatible endpoint shares. The body is already OpenAI-shaped, so
//! parameters pass through verbatim; the port keeps Python's two deviations,
//! the `max_completion_tokens` -> `max_tokens` rename and the usage
//! `*_tokens` null-to-zero sanitize.

use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::core_helpers::unix_now;
use litellm_types::{
    llms::openai::ChatMessage,
    utils::{ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse},
};
use serde::{Deserialize, Deserializer, de::DeserializeOwned};
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

struct Lenient<T>(Option<T>);

impl<T> Default for Lenient<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<'de, T: DeserializeOwned> Deserialize<'de> for Lenient<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Ok(Self(serde_json::from_value(value).ok()))
    }
}

#[derive(Deserialize)]
struct ResponseBody {
    #[serde(default)]
    created: Lenient<u64>,
    #[serde(default)]
    model: Lenient<String>,
    #[serde(default)]
    choices: Lenient<Vec<Lenient<ResponseChoice>>>,
    #[serde(default)]
    usage: Lenient<ResponseUsage>,
}

#[derive(Deserialize)]
struct ResponseChoice {
    #[serde(default)]
    index: Lenient<u64>,
    #[serde(default)]
    message: Lenient<ResponseMessage>,
    #[serde(default)]
    finish_reason: Lenient<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    role: Lenient<String>,
    content: Option<ResponseContent>,
    #[serde(default)]
    tool_calls: Lenient<Vec<Value>>,
    refusal: Option<Value>,
}

enum ResponseContent {
    Text(String),
    Other,
}

impl<'de> Deserialize<'de> for ResponseContent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match Value::deserialize(deserializer)? {
            Value::String(text) => Self::Text(text),
            _ => Self::Other,
        })
    }
}

#[derive(Deserialize)]
struct ResponseUsage {
    #[serde(default)]
    prompt_tokens: Lenient<u64>,
    #[serde(default)]
    completion_tokens: Lenient<u64>,
    #[serde(default)]
    total_tokens: Lenient<u64>,
    #[serde(default)]
    prompt_tokens_details: Lenient<PromptDetails>,
}

#[derive(Deserialize)]
struct PromptDetails {
    #[serde(default)]
    cached_tokens: Lenient<u64>,
    #[serde(default)]
    cache_creation_tokens: Lenient<u64>,
    #[serde(default)]
    text_tokens: Lenient<u64>,
}

impl BaseConfig for OpenAILikeChatConfig {
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
        if !body.is_object() {
            return Err(Error::InvalidResponse(
                "chat response is not an object".into(),
            ));
        }
        let body: ResponseBody = serde_json::from_value(body)
            .map_err(|error| Error::InvalidResponse(format!("invalid chat response: {error}")))?;
        let choices = body
            .choices
            .0
            .ok_or(Error::MissingField("choices"))?
            .into_iter()
            .enumerate()
            .map(|(position, choice)| normalize_choice(position, choice))
            .collect::<Result<Vec<_>, _>>()?;
        let usage = body.usage.0;
        let details = usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens_details.0.as_ref());

        Ok(ChatCompletionsResponse {
            created: body.created.0.unwrap_or_else(unix_now),
            model: body.model.0.unwrap_or_else(|| model.to_string()),
            choices,
            usage: litellm_types::utils::ChatCompletionsUsage {
                prompt_tokens: usage
                    .as_ref()
                    .and_then(|usage| usage.prompt_tokens.0)
                    .unwrap_or(0),
                completion_tokens: usage
                    .as_ref()
                    .and_then(|usage| usage.completion_tokens.0)
                    .unwrap_or(0),
                total_tokens: usage
                    .as_ref()
                    .and_then(|usage| usage.total_tokens.0)
                    .unwrap_or(0),
                prompt_tokens_details: litellm_types::utils::PromptTokensDetails {
                    cached_tokens: details
                        .and_then(|details| details.cached_tokens.0)
                        .unwrap_or(0),
                    cache_creation_tokens: details
                        .and_then(|details| details.cache_creation_tokens.0)
                        .unwrap_or(0),
                    text_tokens: details
                        .and_then(|details| details.text_tokens.0)
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

fn normalize_choice(
    position: usize,
    choice: Lenient<ResponseChoice>,
) -> Result<ChatCompletionsChoice, Error> {
    let choice = choice.0.ok_or(Error::MissingField("message"))?;
    let message = choice.message.0.ok_or(Error::MissingField("message"))?;
    if message.tool_calls.0.is_some_and(|calls| !calls.is_empty()) {
        // Python rewrites the lone tool call into content only under
        // `json_mode`, a request flag `transform_response` cannot see, and the
        // normalized type cannot carry tool calls at all. Declining is
        // terminal at this point, but passing back an empty assistant turn
        // would fabricate the reply.
        return Err(Error::Unsupported("tool call response"));
    }
    if message.refusal.is_some() {
        return Err(Error::Unsupported("refusal response"));
    }
    let content = match message.content {
        Some(ResponseContent::Text(text)) => Some(text),
        Some(ResponseContent::Other) => {
            return Err(Error::Unsupported("non-text response content"));
        }
        None => None,
    };
    Ok(ChatCompletionsChoice {
        index: choice.index.0.unwrap_or(position as u64),
        message: ChatCompletionsChoiceMessage {
            role: message.role.0.unwrap_or_else(|| "assistant".to_string()),
            content,
        },
        finish_reason: choice.finish_reason.0.unwrap_or_default(),
    })
}
