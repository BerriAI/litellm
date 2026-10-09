use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_http::request::string_headers as shared_string_headers;
use litellm_llms::{
    anthropic::chat::transformation::ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
    base_llm::chat::transformation::BaseConfig,
    bedrock::chat::converse_transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
    openai_like::chat::transformation::OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
};
use litellm_llms_types::formats::chat_completions::ChatMessage;
use serde_json::{Map, Value};

use super::Error;
use crate::types::{ChatCompletionsCall, ResolvedChatCompletionsRequest};

const HEADER_CONTEXT: &str = "chat completions";

pub(super) enum ChatProvider {
    Anthropic,
    Bedrock,
    OpenaiLike,
}

impl ChatProvider {
    pub(super) fn config(self) -> &'static dyn BaseConfig {
        match self {
            Self::Anthropic => &ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
            Self::Bedrock => &BEDROCK_CHAT_COMPLETIONS_CONFIG,
            Self::OpenaiLike => &OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
        }
    }
}

pub(super) fn chat_completions_provider(provider: LlmProviders) -> Option<ChatProvider> {
    match provider {
        LlmProviders::Anthropic => Some(ChatProvider::Anthropic),
        LlmProviders::Bedrock => Some(ChatProvider::Bedrock),
        LlmProviders::OpenaiLike => Some(ChatProvider::OpenaiLike),
        _ => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers).map_err(Error::from)
}

pub(super) fn resolve_request(
    call: ChatCompletionsCall,
    config: &'static dyn BaseConfig,
) -> Result<ResolvedChatCompletionsRequest, Error> {
    let messages = serde_json::from_value::<Vec<ChatMessage>>(call.messages).map_err(|err| {
        Error::InvalidRequest(litellm_llms::ErrorDetail::invalid(
            "chat completions messages",
            err,
        ))
    })?;
    if messages.is_empty() {
        return Err(Error::InvalidRequest(
            "chat completions requires at least one message".into(),
        ));
    }
    if let Some(reason) = config.unsupported_reason(&messages, &call.optional_params) {
        return Err(Error::Unsupported(reason.0));
    }
    Ok(ResolvedChatCompletionsRequest {
        messages,
        optional_params: call.optional_params,
        api_key: call.api_key,
        api_base: call.api_base,
        extra_headers: call.extra_headers,
        timeout: call.timeout,
    })
}
