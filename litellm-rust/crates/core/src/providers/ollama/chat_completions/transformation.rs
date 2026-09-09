use serde_json::{Map, Value, json};

use crate::chat_completions::response_utils::{finish_reason_for, unix_now};
use crate::chat_completions::transformation::{ChatCompletionsAuth, ChatCompletionsProviderConfig};
use crate::chat_completions::types::{
    ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse,
    ChatCompletionsUsage, ChatMessage, ChatMessageContent, ProviderChatRequestData,
    ProviderChatResponseData,
};
use crate::constants::{
    DEFAULT_OLLAMA_API_BASE, OLLAMA_API_BASE_ENV, OLLAMA_API_KEY_ENV, OLLAMA_CHAT_PATH,
};
use crate::error::Error;

/// OpenAI parameter names paired with the Ollama `options` names the Rust path
/// places verbatim into the body. The host runs `map_openai_params` before the
/// gate, so these arrive already translated, and the gate only needs the
/// translated names to recognize them.
const SUPPORTED_PARAMS: &[(&str, &str)] = &[
    ("max_tokens", "num_predict"),
    ("max_completion_tokens", "num_predict"),
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("seed", "seed"),
    ("frequency_penalty", "repeat_penalty"),
    ("stop", "stop"),
];

pub struct OllamaChatConfig;

pub const OLLAMA_CHAT_COMPLETIONS_CONFIG: OllamaChatConfig = OllamaChatConfig;

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn complete_ollama_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let api_base = non_empty(api_base)
        .map(str::to_string)
        .or_else(|| env_lookup(OLLAMA_API_BASE_ENV).filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_OLLAMA_API_BASE.to_string());

    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with(OLLAMA_CHAT_PATH) {
        return api_base.to_string();
    }
    format!("{api_base}{OLLAMA_CHAT_PATH}")
}

fn resolve_ollama_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_lookup(OLLAMA_API_KEY_ENV).filter(|value| !value.trim().is_empty()))
}

fn message_text(message: &ChatMessage) -> String {
    match &message.content {
        Some(ChatMessageContent::Text(text)) => text.clone(),
        // The capability gate already proved every part is a `{"type":"text",
        // "text":..}` block, so concatenating the text fields mirrors Python's
        // `convert_content_list_to_str`, which joins with the empty string.
        Some(ChatMessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        None => String::new(),
    }
}

impl ChatCompletionsProviderConfig for OllamaChatConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_ollama_url(api_base, env_lookup))
    }

    fn auth(
        &self,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ChatCompletionsAuth, Error> {
        Ok(match resolve_ollama_api_key(api_key, env_lookup) {
            Some(token) => ChatCompletionsAuth::Bearer { token },
            None => ChatCompletionsAuth::None,
        })
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
        // Python always sets `images` to `[]` for a text-only message, so emit it
        // to keep the upstream body byte-equal. The gate declines image content,
        // so this is always empty here.
        let ollama_messages: Vec<Value> = messages
            .iter()
            .map(|message| {
                json!({
                    "role": message.role,
                    "content": message_text(message),
                    "images": [],
                })
            })
            .collect();

        // Python pulls `stream` out of `optional_params` and places it at the top
        // level. The gate declines `stream: true`, so the value reaching here is
        // either absent or `false`; drop it from `options` and hardcode `false`.
        let options: Map<String, Value> = optional_params
            .into_iter()
            .filter(|(key, _)| key.as_str() != "stream")
            .collect();

        Ok(ProviderChatRequestData {
            body: json!({
                "model": model,
                "messages": ollama_messages,
                "options": options,
                "stream": false,
            }),
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error> {
        let body = response.body.as_object().ok_or_else(|| {
            Error::InvalidResponse("ollama chat response is not an object".into())
        })?;

        let message = body
            .get("message")
            .and_then(Value::as_object)
            .ok_or(Error::MissingField("message"))?;
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string);
        let reasoning_content = message
            .get("thinking")
            .and_then(Value::as_str)
            .map(str::to_string);
        // Ollama reports usage as `prompt_eval_count` and `eval_count` on the
        // `done: true` response. Non-streaming `/api/chat` always returns both.
        // 
        // When both are present report them verbatim; when they are absent,
        // leave `usage` `None` so the Python bridge estimates it.
        let prompt_tokens = body.get("prompt_eval_count").and_then(Value::as_u64);
        let completion_tokens = body.get("eval_count").and_then(Value::as_u64);
        let usage = match (prompt_tokens, completion_tokens) {
            (Some(prompt_tokens), Some(completion_tokens)) => Some(ChatCompletionsUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
            }),
            _ => None,
        };

        Ok(ChatCompletionsResponse {
            created: unix_now(),
            // Python prefixes the request model with `ollama_chat/`, not the
            // server-reported `body["model"]`, so the cost and logging keys agree
            // with the deployment's model name on both paths.
            model: format!("ollama_chat/{model}"),
            choices: vec![ChatCompletionsChoice {
                index: 0,
                message: ChatCompletionsChoiceMessage {
                    role: "assistant".to_string(),
                    content,
                    reasoning_content,
                },
                finish_reason: finish_reason_for(
                    body.get("done_reason")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )
                .to_string(),
            }],
            usage,
        })
    }

    // No `unsupported_reason` override: Ollama has no Anthropic-style "must open
    // on a user turn" rule, so the default trait impl, which composes the shared
    // `unsupported_param` + `unsupported_message` helpers, is exactly the gate
    // this provider wants.
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
