use litellm_core_utils::decline::Decline;
use litellm_llms_types::formats::chat_completions::{
    ChatCompletionsResponse, ChatMessage, ChatMessageContent,
};
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::chat::streaming::{ChatStream, StreamShape},
};

/// The provider-shaped request body a config produces. Named rather than a bare
/// `Value` so the transform contract stays a typed one, mirroring
/// [`crate::base_llm::audio_transcription::transformation::AudioTranscriptionRequestData`].
pub struct ProviderChatRequestData {
    pub body: Value,
    pub stream_shape: StreamShape,
}

/// The raw provider response body handed back to a config for normalization.
pub struct ProviderChatResponseData {
    pub body: Value,
}

pub const STREAM_PARAM: &str = "stream";

const STOP_PARAM: &str = "stop";

const UNRECOGNIZED_PARAM: &str = "unrecognized request parameter";

const SAMPLING_PARAMS: [&str; 2] = ["temperature", "top_p"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropParams {
    Off,
    On { model_rejects_sampling: bool },
}

/// Message fields that carry no meaning for the upstream body, so their
/// presence does not make a request untranslatable.
const IGNORABLE_MESSAGE_FIELDS: &[&str] = &["name"];

pub use crate::base_llm::auth::{Headers, ValidatedEnvironment};

pub trait BaseConfig: Sync {
    fn secret_names(&self) -> Vec<&'static str>;

    /// Supported OpenAI parameter names paired with their provider names.
    fn supported_openai_param_mappings(&self) -> &'static [(&'static str, &'static str)];

    fn map_openai_params(
        &self,
        _model: &str,
        params: Map<String, Value>,
        _drop_params: DropParams,
    ) -> Result<Map<String, Value>, Decline> {
        map_openai_param_names(
            self.supported_openai_param_mappings(),
            self.caller_config_params(),
            params,
        )
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> Result<ProviderChatRequestData, Error>;

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error>;

    /// `None` means this config has no streaming path yet, so the host keeps the request.
    fn model_response_iterator(&self, _shape: StreamShape) -> Option<ChatStream> {
        None
    }

    /// Shapes the forwarded headers and names the credential, the way Python's
    /// `validate_environment` does, without applying it: `resolve_auth` does that once
    /// for every config.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error>;

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    /// Parameters consumed as call configuration (credentials, endpoints)
    /// rather than placed in the body. Accepted, never serialized.
    fn config_params(&self) -> &'static [&'static str] {
        &[]
    }

    fn caller_config_params(&self) -> &'static [&'static str] {
        &[]
    }

    fn unsupported_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Decline> {
        unsupported_param(
            self.supported_openai_param_mappings(),
            self.config_params(),
            optional_params,
        )
        .or_else(|| messages.iter().find_map(unsupported_message))
    }
}

pub fn map_openai_param_names(
    supported: &'static [(&'static str, &'static str)],
    caller_config: &'static [&'static str],
    params: Map<String, Value>,
) -> Result<Map<String, Value>, Decline> {
    let superseded = superseded_openai_names(supported, &params);
    let (openai_named, provider_named): (Vec<_>, Vec<_>) = params
        .into_iter()
        .filter(|(name, value)| !value.is_null() && !superseded.contains(&name.as_str()))
        .partition(|(name, _)| supported.iter().any(|(openai_name, _)| openai_name == name));
    openai_named
        .into_iter()
        .chain(provider_named)
        .map(|(name, value)| {
            provider_param_name(supported, caller_config, name)
                .map(|provider_name| (provider_name, value))
                .ok_or(Decline::new(UNRECOGNIZED_PARAM))
        })
        .collect()
}

/// When two OpenAI names map to one provider name, the later table entry wins
/// whatever the body order, as Python's `get_optional_params` argument order does.
fn superseded_openai_names(
    supported: &'static [(&'static str, &'static str)],
    params: &Map<String, Value>,
) -> Vec<&'static str> {
    supported
        .iter()
        .enumerate()
        .filter(|(position, (_, provider_name))| {
            supported[position + 1..]
                .iter()
                .any(|(later_name, later_provider_name)| {
                    later_provider_name == provider_name
                        && params
                            .get(*later_name)
                            .is_some_and(|value| !value.is_null())
                })
        })
        .map(|(_, (openai_name, _))| *openai_name)
        .collect()
}

fn provider_param_name(
    supported: &'static [(&'static str, &'static str)],
    caller_config: &'static [&'static str],
    name: String,
) -> Option<String> {
    if name == STREAM_PARAM || caller_config.contains(&name.as_str()) {
        return Some(name);
    }
    supported
        .iter()
        .find(|(openai_name, _)| *openai_name == name)
        .map(|(_, provider_name)| (*provider_name).to_string())
        .or_else(|| {
            supported
                .iter()
                .any(|(_, provider_name)| *provider_name == name)
                .then_some(name)
        })
}

pub fn without_rejected_sampling_params(
    params: Map<String, Value>,
    drop_params: DropParams,
) -> Map<String, Value> {
    match drop_params {
        DropParams::On {
            model_rejects_sampling: true,
        } => params
            .into_iter()
            .filter(|(name, _)| !SAMPLING_PARAMS.contains(&name.as_str()))
            .collect(),
        DropParams::On { .. } | DropParams::Off => params,
    }
}

pub fn map_openai_stop(
    params: Map<String, Value>,
    to_provider_stop: fn(Value) -> Option<Value>,
) -> Map<String, Value> {
    params
        .into_iter()
        .filter_map(|(name, value)| match name.as_str() {
            STOP_PARAM => to_provider_stop(value).map(|value| (name, value)),
            _ => Some((name, value)),
        })
        .collect()
}

pub fn unsupported_param(
    supported: &'static [(&'static str, &'static str)],
    config: &'static [&'static str],
    optional_params: &Map<String, Value>,
) -> Option<Decline> {
    if optional_params
        .get(STREAM_PARAM)
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some(Decline::new("streaming"));
    }
    optional_params
        .keys()
        .any(|key| {
            key != STREAM_PARAM
                && !supported
                    .iter()
                    .any(|(_, provider_name)| *provider_name == key)
                && !config.contains(&key.as_str())
        })
        .then_some(Decline::new(UNRECOGNIZED_PARAM))
}

/// Message shapes the core can translate faithfully: text content, either a
/// plain string or a non-empty list of parts that are all
/// `{"type": "text", "text": ...}`. Tool calls, tool results, and multimodal
/// parts decline so Python's fuller translation handles them.
pub fn unsupported_message(message: &ChatMessage) -> Option<Decline> {
    unsupported_message_fields(message, &[])
        .or_else(|| unsupported_message_content(message.content.as_ref()))
}

pub fn unsupported_message_fields(
    message: &ChatMessage,
    accepted_fields: &[&str],
) -> Option<Decline> {
    if message.extra.keys().any(|key| {
        !IGNORABLE_MESSAGE_FIELDS.contains(&key.as_str())
            && !accepted_fields.contains(&key.as_str())
    }) {
        return Some(Decline::new("unrecognized message field"));
    }
    (!matches!(message.role.as_str(), "system" | "user" | "assistant"))
        .then_some(Decline::new("unrecognized message role"))
}

pub fn unsupported_message_content(content: Option<&ChatMessageContent>) -> Option<Decline> {
    match content {
        None => Some(Decline::new("message without content")),
        Some(ChatMessageContent::Text(_)) => None,
        Some(ChatMessageContent::Parts(parts)) if parts.is_empty() => {
            Some(Decline::new("message without content"))
        }
        Some(ChatMessageContent::Parts(parts)) => parts
            .iter()
            .any(|part| {
                part.get("type").and_then(Value::as_str) != Some("text")
                    || part.get("text").and_then(Value::as_str).is_none()
                    || part.as_object().is_some_and(|object| object.len() != 2)
            })
            .then_some(Decline::new("non-text message content")),
    }
}
