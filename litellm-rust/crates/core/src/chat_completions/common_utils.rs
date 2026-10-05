use litellm_http::request::string_headers as shared_string_headers;
use litellm_llms::{
    anthropic::chat::transformation::ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
    base_llm::chat::transformation::BaseConfig,
    baseten::chat::transformation::BASETEN_CHAT_COMPLETIONS_CONFIG,
    bedrock::chat::converse_transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
    openai_like::chat::transformation::OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
};
use serde_json::{Map, Value};

use super::Error;
use crate::provider::LlmProviders;

const HEADER_CONTEXT: &str = "chat completions";

pub(super) enum ChatProvider {
    Anthropic,
    Bedrock,
    Baseten,
    OpenaiLike,
}

impl ChatProvider {
    pub(super) fn config(self) -> &'static dyn BaseConfig {
        match self {
            Self::Anthropic => &ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
            Self::Bedrock => &BEDROCK_CHAT_COMPLETIONS_CONFIG,
            Self::Baseten => &BASETEN_CHAT_COMPLETIONS_CONFIG,
            Self::OpenaiLike => &OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
        }
    }
}

pub(super) fn chat_completions_provider(provider: LlmProviders) -> Option<ChatProvider> {
    match provider {
        LlmProviders::Anthropic => Some(ChatProvider::Anthropic),
        LlmProviders::Bedrock => Some(ChatProvider::Bedrock),
        LlmProviders::Baseten => Some(ChatProvider::Baseten),
        LlmProviders::OpenaiLike => Some(ChatProvider::OpenaiLike),
        LlmProviders::AwsTextract
        | LlmProviders::AzureAi
        | LlmProviders::Deepseek
        | LlmProviders::Cohere
        | LlmProviders::Mistral
        | LlmProviders::Openai
        | LlmProviders::Reducto
        | LlmProviders::VertexAi => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers).map_err(Error::from)
}
