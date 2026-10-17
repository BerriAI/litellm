use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_http::request::string_headers as shared_string_headers;
use litellm_llms::{
    anthropic::chat::transformation::ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
    base_llm::chat::transformation::BaseConfig,
    bedrock::chat::converse_transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
    gemini::chat::transformation::GOOGLE_AI_STUDIO_GEMINI_CHAT_COMPLETIONS_CONFIG,
    openai_like::chat::transformation::OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
    vertex_ai::gemini::vertex_and_google_ai_studio_gemini::{
        VERTEX_GEMINI_CHAT_COMPLETIONS_CONFIG, is_vertex_gemini_model,
    },
};
use serde_json::{Map, Value};

use super::Error;

const HEADER_CONTEXT: &str = "chat completions";

pub(super) enum ChatProvider {
    Anthropic,
    Bedrock,
    Gemini,
    OpenaiLike,
    VertexGemini,
}

impl ChatProvider {
    pub(super) fn config(self) -> &'static dyn BaseConfig {
        match self {
            Self::Anthropic => &ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
            Self::Bedrock => &BEDROCK_CHAT_COMPLETIONS_CONFIG,
            Self::Gemini => &GOOGLE_AI_STUDIO_GEMINI_CHAT_COMPLETIONS_CONFIG,
            Self::OpenaiLike => &OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
            Self::VertexGemini => &VERTEX_GEMINI_CHAT_COMPLETIONS_CONFIG,
        }
    }
}

/// Vertex AI serves only its Gemini models here; Claude and partner models stay refused.
pub(super) fn chat_completions_provider(
    provider: LlmProviders,
    model: &str,
) -> Option<ChatProvider> {
    match provider {
        LlmProviders::Anthropic => Some(ChatProvider::Anthropic),
        LlmProviders::Bedrock => Some(ChatProvider::Bedrock),
        LlmProviders::Gemini => Some(ChatProvider::Gemini),
        LlmProviders::OpenaiLike => Some(ChatProvider::OpenaiLike),
        LlmProviders::VertexAi if is_vertex_gemini_model(model) => Some(ChatProvider::VertexGemini),
        _ => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::studio(LlmProviders::Gemini, "gemini-2.5-flash", Some("gemini"))]
    #[case::vertex_gemini(LlmProviders::VertexAi, "gemini-2.5-pro", Some("vertex_gemini"))]
    #[case::vertex_claude(LlmProviders::VertexAi, "claude-sonnet-4@20250514", None)]
    #[case::vertex_partner(LlmProviders::VertexAi, "meta/llama-3.1-405b-instruct-maas", None)]
    #[case::vertex_gemma(LlmProviders::VertexAi, "gemma/gemma-3-27b-it", None)]
    fn vertex_routes_only_its_gemini_models(
        #[case] provider: LlmProviders,
        #[case] model: &str,
        #[case] expected: Option<&str>,
    ) {
        let routed = chat_completions_provider(provider, model).map(|provider| match provider {
            ChatProvider::Gemini => "gemini",
            ChatProvider::VertexGemini => "vertex_gemini",
            ChatProvider::Anthropic | ChatProvider::Bedrock | ChatProvider::OpenaiLike => "other",
        });
        assert_eq!(routed, expected);
    }
}
