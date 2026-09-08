use super::anthropic::chat_completions::transformation::ANTHROPIC_CHAT_COMPLETIONS_CONFIG;
use super::anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG;
use super::azure_ai::messages::transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG;
use super::azure_ai::ocr::transformation as azure_ai;
#[cfg(feature = "bedrock-auth")]
use super::bedrock::audio_transcription::BEDROCK_AUDIO_TRANSCRIPTION_CONFIG;
use super::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use super::vertex_ai::ocr::transformation as vertex_ai;
use crate::Error;
use crate::audio_transcription::transformation::AudioTranscriptionProviderConfig;
use crate::chat_completions::transformation::ChatCompletionsProviderConfig;
use crate::messages::transformation::AnthropicMessagesProviderConfig;
use crate::ocr::transformation::OcrProviderConfig;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn chat_completions_provider_config(
    provider: &str,
) -> Option<&'static dyn ChatCompletionsProviderConfig> {
    match provider {
        "anthropic" => Some(&ANTHROPIC_CHAT_COMPLETIONS_CONFIG),
        #[cfg(feature = "bedrock-auth")]
        "bedrock" => Some(
            &crate::providers::bedrock::chat_completions::transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
        ),
        _ => None,
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn messages_provider_config(
    provider: &str,
) -> Option<&'static dyn AnthropicMessagesProviderConfig> {
    match provider {
        "anthropic" => Some(&ANTHROPIC_MESSAGES_CONFIG),
        "azure_ai" => Some(&AZURE_ANTHROPIC_MESSAGES_CONFIG),
        _ => None,
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn audio_transcription_provider_config(
    provider: &str,
) -> Option<&'static dyn AudioTranscriptionProviderConfig> {
    #[cfg(feature = "bedrock-auth")]
    if provider == "bedrock" {
        return Some(&BEDROCK_AUDIO_TRANSCRIPTION_CONFIG);
    }
    let _ = provider;
    None
}

pub fn ocr_provider_config(
    provider: &str,
    model: &str,
) -> Result<&'static dyn OcrProviderConfig, Error> {
    match provider {
        "mistral" => Ok(&MISTRAL_OCR_CONFIG),
        "azure_ai" => azure_ai::config_for_model(model),
        "vertex_ai" => vertex_ai::config_for_model(model),
        _ => Err(Error::Unsupported("OCR provider")),
    }
}

pub fn realtime_provider_config(
    model: &str,
) -> Result<
    (
        &str,
        &'static (dyn crate::realtime::transformation::RealtimeProviderConfig + Sync),
    ),
    Error,
> {
    let (provider, model) = model.split_once('/').unwrap_or(("openai", model));
    match provider {
        "openai" => Ok((
            model,
            &super::openai::realtime::transformation::OPENAI_REALTIME_CONFIG,
        )),
        _ => Err(Error::InvalidProvider(format!(
            "realtime route does not support provider '{provider}'"
        ))),
    }
}

pub fn responses_websocket_provider_config()
-> &'static dyn crate::responses::websocket::ResponsesWebSocketProviderConfig {
    &super::openai::responses::transformation::OPENAI_RESPONSES_WS_CONFIG
}

pub(crate) fn resolve_audio_route_provider<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> crate::routing_utils::provider::CustomLlmProvider<'a> {
    use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};
    get_custom_llm_provider(model, custom_llm_provider).unwrap_or(CustomLlmProvider {
        model,
        custom_llm_provider: "bedrock",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::RequestBodyPolicy;
    use crate::ocr::transformation::OcrResponseHandling;

    #[test]
    fn unsupported_pairs_do_not_select_another_provider() {
        let default_audio = resolve_audio_route_provider("model", None);
        assert_eq!(default_audio.custom_llm_provider, "bedrock");
        assert_eq!(default_audio.model, "model");
        let explicit_audio = resolve_audio_route_provider("unknown/model", None);
        assert_eq!(explicit_audio.custom_llm_provider, "unknown");
        assert!(audio_transcription_provider_config(explicit_audio.custom_llm_provider).is_none());
        for provider in ["unknown", "openai", "mistral", "reducto"] {
            assert!(chat_completions_provider_config(provider).is_none());
            assert!(messages_provider_config(provider).is_none());
            assert!(audio_transcription_provider_config(provider).is_none());
        }
        assert!(matches!(
            ocr_provider_config("reducto", "model"),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            realtime_provider_config("anthropic/model"),
            Err(Error::InvalidProvider(_))
        ));
        assert_eq!(
            chat_completions_provider_config("bedrock").is_some(),
            cfg!(feature = "bedrock-auth")
        );
        assert_eq!(
            audio_transcription_provider_config("bedrock").is_some(),
            cfg!(feature = "bedrock-auth")
        );
    }

    #[test]
    fn every_buffered_provider_declares_its_body_policy() {
        assert_eq!(
            chat_completions_provider_config("anthropic")
                .unwrap()
                .request_body_policy(),
            RequestBodyPolicy::StructuredAtSend
        );
        #[cfg(feature = "bedrock-auth")]
        assert_eq!(
            chat_completions_provider_config("bedrock")
                .unwrap()
                .request_body_policy(),
            RequestBodyPolicy::SerializedAtBuild
        );
        for provider in ["anthropic", "azure_ai"] {
            assert_eq!(
                messages_provider_config(provider)
                    .unwrap()
                    .request_body_policy(),
                RequestBodyPolicy::StructuredAtBuild
            );
        }
        for (provider, model) in [
            ("mistral", "mistral-ocr-latest"),
            ("azure_ai", "mistral"),
            ("azure_ai", "doc-intelligence/prebuilt-layout"),
            ("vertex_ai", "mistral"),
            ("vertex_ai", "deepseek"),
        ] {
            assert_eq!(
                ocr_provider_config(provider, model)
                    .unwrap()
                    .request_body_policy(),
                RequestBodyPolicy::StructuredAtSend
            );
        }
        #[cfg(feature = "bedrock-auth")]
        assert_eq!(
            audio_transcription_provider_config("bedrock")
                .unwrap()
                .request_body_policy(),
            RequestBodyPolicy::StructuredAtBuild
        );
    }

    #[test]
    fn ocr_selection_preserves_model_specific_protocols() {
        assert_eq!(
            ocr_provider_config("azure_ai", "doc-intelligence/prebuilt-layout")
                .unwrap()
                .response_handling(),
            OcrResponseHandling::AzureDocumentIntelligencePoll
        );
        assert_eq!(
            ocr_provider_config("azure_ai", "mistral")
                .unwrap()
                .response_handling(),
            OcrResponseHandling::Json
        );
        assert!(
            ocr_provider_config("vertex_ai", "mistral")
                .unwrap()
                .requires_data_uri_document()
        );
        assert!(
            !ocr_provider_config("vertex_ai", "deepseek")
                .unwrap()
                .requires_data_uri_document()
        );
        for provider in ["azure_ai", "vertex_ai"] {
            assert!(matches!(
                ocr_provider_config(provider, "cohere"),
                Err(Error::Unsupported(_))
            ));
        }
    }

    #[test]
    fn websocket_adapters_preserve_model_and_credential_policy() {
        let (model, config) = realtime_provider_config("openai/model/variant").unwrap();
        assert_eq!(model, "model/variant");
        assert_eq!(
            config.complete_url(Some("http://localhost"), model),
            "ws://localhost/v1/realtime?model=model%2Fvariant"
        );
        assert_eq!(
            config
                .resolve_api_key(Some(" explicit "), &|_| panic!("explicit key must win"))
                .unwrap(),
            "explicit"
        );
        assert_eq!(
            config
                .resolve_api_key(Some(" "), &|key| (key == "OPENAI_API_KEY")
                    .then(|| " env ".into()))
                .unwrap(),
            " env "
        );
        assert!(matches!(
            config.resolve_api_key(None, &|_| None),
            Err(Error::Auth(_))
        ));

        let config = responses_websocket_provider_config();
        let event = serde_json::from_value(
            serde_json::json!({"type": "response.create", "model": "caller"}),
        )
        .unwrap();
        let events = config
            .transform_ws_request(&event, "deployment/model")
            .unwrap()
            .events;
        assert_eq!(events[0].model(), Some("deployment/model"));
        assert_eq!(
            config
                .resolve_api_key(Some(" explicit "), &|_| panic!("explicit key must win"))
                .unwrap(),
            "explicit"
        );
        assert!(matches!(
            config.resolve_api_key(None, &|_| Some(" ".into())),
            Err(Error::Auth(_))
        ));
    }
}
