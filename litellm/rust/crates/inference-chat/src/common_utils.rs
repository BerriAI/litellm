use litellm_core_utils::{decline::Decline, get_llm_provider_logic::LlmProviders};
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
use crate::types::{ChatCompletionsCall, OptionalParams, ResolvedChatCompletionsRequest};

const HEADER_CONTEXT: &str = "chat completions";

const CLIENT_SIDE_CREDENTIAL: &str = "client-side credential needs general_settings.allow_client_side_credentials or the deployment's configurable_clientside_auth_params";

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
    model: &str,
    config: &'static dyn BaseConfig,
) -> Result<ResolvedChatCompletionsRequest, Error> {
    let messages = serde_json::from_value::<Vec<ChatMessage>>(call.messages).map_err(|err| {
        Error::invalid_request(litellm_llms::ErrorDetail::invalid(
            "chat completions messages",
            err,
        ))
    })?;
    if messages.is_empty() {
        return Err(Error::invalid_request(
            litellm_llms::ErrorDetail::NoMessages {
                route: "chat completions",
            },
        ));
    }
    let optional_params = match call.optional_params {
        OptionalParams::OpenAi {
            params,
            client_side,
            drop_params,
        } => {
            let (call_config, openai): (Map<String, Value>, Map<String, Value>) =
                params.into_iter().partition(|(name, _)| {
                    config.config_params().contains(&name.as_str()) && client_side.admits(name)
                });
            if openai.iter().any(|(name, value)| {
                !value.is_null()
                    && config.config_params().contains(&name.as_str())
                    && !config.caller_config_params().contains(&name.as_str())
            }) {
                return Err(Error::Declined(Decline::new(CLIENT_SIDE_CREDENTIAL)));
            }
            config
                .map_openai_params(model, openai, drop_params)
                .map_err(Error::Declined)?
                .into_iter()
                .chain(call_config)
                .collect()
        }
        OptionalParams::ProviderMapped(params) => params,
    };
    if let Some(reason) = config.unsupported_reason(&messages, &optional_params) {
        return Err(Error::Declined(reason));
    }
    Ok(ResolvedChatCompletionsRequest {
        messages,
        optional_params,
        api_key: call.api_key,
        api_base: call.api_base,
        extra_headers: call.extra_headers,
        timeout: call.timeout,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::chat_completions_provider;
    use litellm_core_utils::get_llm_provider_logic::LlmProviders;

    #[rstest]
    #[case::anthropic(LlmProviders::Anthropic, ("anthropic-version", "2023-06-01"))]
    #[case::bedrock(LlmProviders::Bedrock, ("Content-Type", "application/json"))]
    fn resolves_supported_provider_configs(
        #[case] provider: LlmProviders,
        #[case] expected_header: (&str, &str),
    ) {
        let config = chat_completions_provider(provider)
            .expect("provider has a chat completions config")
            .config();
        assert!(config.default_headers().contains(&expected_header));
    }

    #[rstest]
    #[case::openai(LlmProviders::Openai)]
    fn rejects_providers_without_a_chat_config(#[case] provider: LlmProviders) {
        assert!(chat_completions_provider(provider).is_none());
    }
}
