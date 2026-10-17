//! Python's `GoogleAIStudioGeminiConfig` for `/chat/completions`, with the key and URL
//! `_get_token_and_url` and `_check_custom_proxy` build for `custom_llm_provider="gemini"`.

use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_llms_types::{
    formats::chat_completions::{ChatCompletionsResponse, ChatMessage},
    providers::vertex_ai::{GEMINI_API_BASE, GEMINI_API_KEY_HEADER, GENERATE_CONTENT},
};
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        chat::{
            params::ChatRequestContext,
            streaming::{ChatStream, StreamShape},
            transformation::{
                BaseConfig, ProviderChatRequestData, ProviderChatResponseData, Unsupported,
            },
        },
    },
    vertex_ai::gemini::{
        handler::gemini_chat_stream,
        transformation::remote_media,
        vertex_and_google_ai_studio_gemini::{
            CONFIG_PARAMS, GeminiSurface, build_request, is_gemini_3_or_newer, streaming_url,
            transform_generate_content_response,
        },
    },
};

pub const GEMINI_API_BASE_ENV: &str = "GEMINI_API_BASE";
pub const GEMINI_API_KEY_ENV: &str = "GEMINI_API_KEY";

/// `get_api_key_from_env`, then the older PaLM key.
const API_KEY_ENVS: &[&str] = &["GOOGLE_API_KEY", GEMINI_API_KEY_ENV, "PALM_API_KEY"];

pub struct GoogleAiStudioGeminiConfig;

pub const GOOGLE_AI_STUDIO_GEMINI_CHAT_COMPLETIONS_CONFIG: GoogleAiStudioGeminiConfig =
    GoogleAiStudioGeminiConfig;

/// The id AI Studio is called with: Python strips the provider prefix.
fn studio_model(model: &str) -> &str {
    model.strip_prefix("gemini/").unwrap_or(model)
}

fn api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    api_key
        .filter(|key| !key.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            API_KEY_ENVS
                .iter()
                .find_map(|name| env_lookup(name).filter(|key| !key.is_empty()))
        })
        .ok_or(Error::Auth(litellm_auth::Error::MissingApiKey {
            provider: "Gemini",
            environment_variable: GEMINI_API_KEY_ENV,
        }))
}

impl BaseConfig for GoogleAiStudioGeminiConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        API_KEY_ENVS
            .iter()
            .copied()
            .chain([GEMINI_API_BASE_ENV])
            .collect()
    }

    /// `get_gemini_url`, or `{api_base}/models/{model}:generateContent` for a custom base.
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let model = studio_model(model);
        let custom = api_base
            .map(str::to_owned)
            .or_else(|| env_lookup(GEMINI_API_BASE_ENV))
            .filter(|base| !base.is_empty());
        Ok(match custom {
            Some(base) => format!("{base}/models/{model}:{GENERATE_CONTENT}"),
            None => {
                let version = match is_gemini_3_or_newer(model) {
                    true => "v1alpha",
                    false => "v1beta",
                };
                format!("{GEMINI_API_BASE}/{version}/models/{model}:{GENERATE_CONTENT}")
            }
        })
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
        context: &ChatRequestContext<'_>,
    ) -> Result<ProviderChatRequestData, Error> {
        build_request(
            GeminiSurface::AiStudio,
            model,
            messages,
            optional_params,
            context,
        )
    }

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
        _shape: &StreamShape,
    ) -> Result<ChatCompletionsResponse, Error> {
        transform_generate_content_response(model, response.body)
    }

    fn remote_media(&self, messages: &[ChatMessage]) -> Vec<String> {
        remote_media(messages, GeminiSurface::AiStudio)
    }

    fn model_response_iterator(&self, _shape: StreamShape) -> Option<ChatStream> {
        Some(gemini_chat_stream())
    }

    fn streaming_request(&self, url: String, body: Value) -> Result<(String, Value), Error> {
        Ok((streaming_url(&url), body))
    }

    /// The key always wins over a forwarded header, as Python writes `x-goog-api-key` last.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key_param: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Header(GEMINI_API_KEY_HEADER),
                secret: SecretValue::new(api_key(api_key_param, env_lookup)?),
            },
        })
    }

    fn config_params(&self) -> &'static [&'static str] {
        CONFIG_PARAMS
    }

    /// Every request is translated; a param or message Python refuses is refused here too.
    fn unsupported_reason(
        &self,
        _messages: &[ChatMessage],
        _optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        None
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn base_env(name: &str) -> Option<String> {
        (name == GEMINI_API_BASE_ENV).then(|| "https://proxy.example/gemini".to_owned())
    }

    fn key_env(name: &str) -> Option<String> {
        match name {
            "GEMINI_API_KEY" => Some("gemini-key".into()),
            "PALM_API_KEY" => Some("palm-key".into()),
            _ => None,
        }
    }

    #[rstest]
    #[case::v1beta(None, "gemini/gemini-2.5-flash", no_env as fn(&str) -> Option<String>, "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")]
    #[case::v1alpha(None, "gemini-3-pro-preview", no_env as fn(&str) -> Option<String>, "https://generativelanguage.googleapis.com/v1alpha/models/gemini-3-pro-preview:generateContent")]
    #[case::custom(Some("https://gw.example/v1beta"), "gemini-2.5-flash", no_env as fn(&str) -> Option<String>, "https://gw.example/v1beta/models/gemini-2.5-flash:generateContent")]
    #[case::env_base(None, "gemini-2.5-flash", base_env as fn(&str) -> Option<String>, "https://proxy.example/gemini/models/gemini-2.5-flash:generateContent")]
    fn urls_follow_pythons_gemini_url(
        #[case] api_base: Option<&str>,
        #[case] model: &str,
        #[case] env: fn(&str) -> Option<String>,
        #[case] expected: &str,
    ) {
        let url = GOOGLE_AI_STUDIO_GEMINI_CHAT_COMPLETIONS_CONFIG
            .get_complete_url(api_base, model, &Map::new(), &env)
            .unwrap();
        assert_eq!(url, expected);
    }

    #[rstest]
    #[case::explicit(Some("explicit"), "explicit")]
    #[case::from_env(None, "gemini-key")]
    fn the_key_goes_in_the_goog_header(#[case] key: Option<&str>, #[case] expected: &str) {
        let validated = GOOGLE_AI_STUDIO_GEMINI_CHAT_COMPLETIONS_CONFIG
            .validate_environment(Vec::new(), key, "gemini-2.5-flash", &Map::new(), &key_env)
            .unwrap();
        assert!(matches!(
            validated.auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-goog-api-key"),
                ref secret
            } if secret.expose() == expected
        ));
    }

    #[test]
    fn a_missing_key_is_an_auth_error() {
        let error = GOOGLE_AI_STUDIO_GEMINI_CHAT_COMPLETIONS_CONFIG
            .validate_environment(Vec::new(), None, "gemini-2.5-flash", &Map::new(), &no_env)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Gemini",
                ..
            })
        ));
    }

    #[test]
    fn streams_call_stream_generate_content_with_sse() {
        let (url, _) = GOOGLE_AI_STUDIO_GEMINI_CHAT_COMPLETIONS_CONFIG
            .streaming_request(
                "https://generativelanguage.googleapis.com/v1beta/models/m:generateContent".into(),
                Value::Null,
            )
            .unwrap();
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/m:streamGenerateContent?alt=sse"
        );
    }
}
