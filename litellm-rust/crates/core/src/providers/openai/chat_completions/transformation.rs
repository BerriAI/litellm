use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::Error;
use crate::chat_completions::response_utils::usage_from_parts;
use crate::chat_completions::transformation::{ChatCompletionsAuth, ChatCompletionsProviderConfig};
use crate::chat_completions::types::{
    ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse, ChatMessage,
    ProviderChatRequestData, ProviderChatResponseData,
};

const SUPPORTED_PARAMS: &[(&str, &str)] = &[
    ("max_completion_tokens", "max_completion_tokens"),
    ("temperature", "temperature"),
    ("top_p", "top_p"),
];

pub struct OpenAIChatCompletionsConfig;

pub const OPENAI_CHAT_COMPLETIONS_CONFIG: OpenAIChatCompletionsConfig = OpenAIChatCompletionsConfig;

#[derive(Deserialize)]
struct OpenAIResponse {
    created: u64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    index: u64,
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

impl ChatCompletionsProviderConfig for OpenAIChatCompletionsConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let base = api_base
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .or_else(|| env_lookup("OPENAI_BASE_URL"))
            .or_else(|| env_lookup("OPENAI_API_BASE"))
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let base = base.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            return Ok(base.to_string());
        }
        Ok(format!("{base}/chat/completions"))
    }

    fn auth(
        &self,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ChatCompletionsAuth, Error> {
        let token = api_key
            .map(str::to_string)
            .or_else(|| env_lookup("OPENAI_API_KEY"))
            .filter(|value| !value.is_empty())
            .ok_or_else(|| Error::Auth("OPENAI_API_KEY is required".to_string()))?;
        Ok(ChatCompletionsAuth::Bearer { token })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn supported_openai_params(&self) -> &'static [(&'static str, &'static str)] {
        SUPPORTED_PARAMS
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> Result<ProviderChatRequestData, Error> {
        Ok(ProviderChatRequestData {
            body: Value::Object(Map::from_iter(
                [
                    ("model".to_string(), json!(model)),
                    ("messages".to_string(), json!(messages)),
                ]
                .into_iter()
                .chain(optional_params),
            )),
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_response(
        &self,
        _model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error> {
        let parsed: OpenAIResponse = serde_json::from_value(response.body).map_err(|error| {
            Error::InvalidResponse(format!("invalid OpenAI chat completions response: {error}"))
        })?;
        Ok(ChatCompletionsResponse {
            created: parsed.created,
            model: parsed.model,
            choices: parsed
                .choices
                .into_iter()
                .map(|choice| ChatCompletionsChoice {
                    index: choice.index,
                    message: ChatCompletionsChoiceMessage {
                        role: choice.message.role,
                        content: choice.message.content,
                    },
                    finish_reason: choice.finish_reason.unwrap_or_else(|| "stop".to_string()),
                })
                .collect(),
            usage: usage_from_parts(
                parsed.usage.prompt_tokens,
                parsed.usage.completion_tokens,
                0,
                0,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_base_selects_chat_completions_endpoint() {
        let url = OPENAI_CHAT_COMPLETIONS_CONFIG
            .complete_url(Some("http://localhost:8000"), "gpt-5", &Map::new(), &|_| {
                None
            })
            .expect("valid URL");
        assert_eq!(url, "http://localhost:8000/chat/completions");
    }
}
