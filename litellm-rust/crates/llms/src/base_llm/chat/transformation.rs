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

/// Message fields that carry no meaning for the upstream body, so their
/// presence does not make a request untranslatable.
const IGNORABLE_MESSAGE_FIELDS: &[&str] = &["name"];

pub use crate::base_llm::auth::{Headers, ValidatedEnvironment};
pub use litellm_auth::ConnectionArguments;

pub trait BaseConfig: Sync {
    fn secret_names(&self) -> Vec<&'static str>;

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        connection: &ConnectionArguments,
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
        connection: &ConnectionArguments,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error>;

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    fn validate_request(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Result<(), Error> {
        reject_stream(optional_params)?;
        messages.iter().try_for_each(validate_message)
    }
}

pub fn reject_stream(optional_params: &Map<String, Value>) -> Result<(), Error> {
    let streaming = optional_params
        .get(STREAM_PARAM)
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if streaming {
        return Err(Error::Unsupported("streaming"));
    }
    Ok(())
}

pub fn validate_message(message: &ChatMessage) -> Result<(), Error> {
    if message
        .extra
        .keys()
        .any(|key| !IGNORABLE_MESSAGE_FIELDS.contains(&key.as_str()))
    {
        return Err(Error::Unsupported("unrecognized message field"));
    }
    if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
        return Err(Error::Unsupported("unrecognized message role"));
    }
    match &message.content {
        None => Err(Error::Unsupported("message without content")),
        Some(ChatMessageContent::Text(_)) => Ok(()),
        Some(ChatMessageContent::Parts(parts)) if parts.is_empty() => {
            Err(Error::Unsupported("message without content"))
        }
        Some(ChatMessageContent::Parts(parts)) => {
            let non_text = parts.iter().any(|part| {
                part.get("type").and_then(Value::as_str) != Some("text")
                    || part.get("text").and_then(Value::as_str).is_none()
                    || part.as_object().is_some_and(|object| object.len() != 2)
            });
            if non_text {
                return Err(Error::Unsupported("non-text message content"));
            }
            Ok(())
        }
    }
}
