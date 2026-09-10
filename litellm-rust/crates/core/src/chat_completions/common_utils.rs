use crate::Error;
use crate::http_utils::string_headers as shared_string_headers;
use crate::providers::anthropic::chat_completions::transformation::ANTHROPIC_CHAT_COMPLETIONS_CONFIG;
use serde_json::{Map, Value};

use super::transformation::ChatCompletionsProviderConfig;

const HEADER_CONTEXT: &str = "chat completions";

#[derive(Clone, Copy)]
pub(super) enum ChatProviderConfig {
    Anthropic,
    #[cfg(feature = "bedrock-auth")]
    Bedrock,
}

pub(super) fn chat_completions_provider_config(provider: &str) -> Option<ChatProviderConfig> {
    match provider {
        "anthropic" => Some(ChatProviderConfig::Anthropic),
        #[cfg(feature = "bedrock-auth")]
        "bedrock" => Some(ChatProviderConfig::Bedrock),
        _ => None,
    }
}

impl ChatProviderConfig {
    pub(super) fn decline_reason(
        self,
        messages: &[super::types::ChatMessage],
        params: &Map<String, Value>,
    ) -> Option<super::transformation::Unsupported> {
        match self {
            Self::Anthropic => decline_reason(&ANTHROPIC_CHAT_COMPLETIONS_CONFIG, messages, params),
            #[cfg(feature = "bedrock-auth")]
            Self::Bedrock => decline_reason(
                &crate::providers::bedrock::chat_completions::transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
                messages,
                params,
            ),
        }
    }
}

fn decline_reason<C: ChatCompletionsProviderConfig>(
    config: &C,
    messages: &[super::types::ChatMessage],
    params: &Map<String, Value>,
) -> Option<super::transformation::Unsupported> {
    config.decline_reason(messages, params).or_else(|| {
        serde_json::from_value::<C::MappedParams>(Value::Object(params.clone()))
            .err()
            .map(|_| super::transformation::Unsupported("invalid provider-mapped parameters"))
    })
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers)
}
