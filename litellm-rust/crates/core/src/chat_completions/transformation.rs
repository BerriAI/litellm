use super::conversation::Conversation;
use crate::Error;
use crate::auth::RequestAuth;
use crate::chat_completions::error::{ChatRequestError, ChatResponseError};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use super::types::{ChatCompletionsResponse, ChatMessage, ChatMessageContent};

/// Why a request cannot be served by the Rust path.
///
/// The core declines rather than guessing: the host turns this into a
/// transparent fallback to the Python implementation, which covers the full
/// surface. Acceptance is an allowlist, so a parameter or message shape the
/// core has never seen declines by construction instead of being translated
/// wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unsupported(pub &'static str);

pub const STREAM_PARAM: &str = "stream";

/// Message fields that carry no meaning for the upstream body, so their
/// presence does not make a request untranslatable.
const IGNORABLE_MESSAGE_FIELDS: &[&str] = &["name"];

pub trait ChatCompletionsProviderConfig: Sync {
    type MappedParams: DeserializeOwned;
    type RequestBody: Serialize;
    type ResponseBody: DeserializeOwned;

    /// Accepted parameter names after Python provider mapping.
    fn supported_provider_params(&self) -> &'static [&'static str];

    /// Parameters consumed as call configuration (credentials, endpoints)
    /// rather than placed in the body. Accepted, never serialized.
    fn config_params(&self) -> &'static [&'static str] {
        &[]
    }

    fn decline_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        unsupported_param(
            self.supported_provider_params(),
            self.config_params(),
            optional_params,
        )
        .or_else(|| messages.iter().find_map(unsupported_message))
    }

    fn auth(
        &self,
        api_key: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<RequestAuth, Error>;

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    /// Whether an auth header the caller already supplied is the credential this
    /// request should authenticate with, so the resolved one is not applied.
    ///
    /// Defaults to false: the deployment's credential outranks anything
    /// forwarded, which is what every provider wants for its own auth header.
    /// A provider overrides this only for a scheme it hands off to entirely.
    fn defers_to_forwarded_auth(&self, _headers: &[(String, String)]) -> bool {
        false
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_request(
        &self,
        model: &str,
        conversation: Conversation,
        params: Self::MappedParams,
    ) -> Result<Self::RequestBody, ChatRequestError>;

    fn transform_response(
        &self,
        model: &str,
        response: Self::ResponseBody,
    ) -> Result<ChatCompletionsResponse, ChatResponseError>;
}

pub fn unsupported_param(
    supported: &'static [&'static str],
    config: &'static [&'static str],
    optional_params: &Map<String, Value>,
) -> Option<Unsupported> {
    if optional_params
        .get(STREAM_PARAM)
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some(Unsupported("streaming"));
    }
    optional_params
        .keys()
        .any(|key| {
            key != STREAM_PARAM
                && !supported.iter().any(|provider_name| *provider_name == key)
                && !config.contains(&key.as_str())
        })
        .then_some(Unsupported("unrecognized request parameter"))
}

/// Message shapes the core can translate faithfully: text content, either a
/// plain string or a non-empty list of parts that are all
/// `{"type": "text", "text": ...}`. Tool calls, tool results, and multimodal
/// parts decline so Python's fuller translation handles them.
pub fn unsupported_message(message: &ChatMessage) -> Option<Unsupported> {
    if message
        .extra
        .keys()
        .any(|key| !IGNORABLE_MESSAGE_FIELDS.contains(&key.as_str()))
    {
        return Some(Unsupported("unrecognized message field"));
    }
    if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
        return Some(Unsupported("unrecognized message role"));
    }
    match &message.content {
        None => Some(Unsupported("message without content")),
        Some(ChatMessageContent::Text(_)) => None,
        Some(ChatMessageContent::Parts(parts)) if parts.is_empty() => {
            Some(Unsupported("message without content"))
        }
        Some(ChatMessageContent::Parts(parts)) => parts
            .iter()
            .any(|part| {
                part.get("type").and_then(Value::as_str) != Some("text")
                    || part.get("text").and_then(Value::as_str).is_none()
                    || part.as_object().is_some_and(|object| object.len() != 2)
            })
            .then_some(Unsupported("non-text message content")),
    }
}
