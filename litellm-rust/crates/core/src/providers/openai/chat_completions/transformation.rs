use serde_json::{Map, Value, json};

use crate::chat_completions::conversation::{Conversation, build_conversation};
use crate::chat_completions::response_utils::{finish_reason_for, unix_now, usage_from_parts};
use crate::chat_completions::transformation::{
    ChatCompletionsAuth, ChatCompletionsProviderConfig, Unsupported, unsupported_message,
    unsupported_param,
};
use crate::chat_completions::types::{
    ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse, ChatMessage,
    ProviderChatRequestData, ProviderChatResponseData,
};
use crate::error::{CoreError, CoreResult};

const SUPPORTED_PARAMS: &[&str] = &[
    "max_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
];

pub struct OpenAIChatCompletionsConfig;

pub const OPENAI_CHAT_COMPLETIONS_CONFIG: OpenAIChatCompletionsConfig = OpenAIChatCompletionsConfig;

fn openai_body(model: &str, conversation: &Conversation, params: Map<String, Value>) -> Value {
    let messages: Vec<Value> = conversation
        .system
        .iter()
        .map(|s| json!({"role": "system", "content": s}))
        .chain(conversation.turns.iter().map(|turn| {
            json!({
                "role": turn.role.as_str(),
                "content": turn.texts.join("\n"),
            })
        }))
        .collect();

    let body = Map::from_iter(
        [
            ("model".to_string(), json!(model)),
            ("messages".to_string(), json!(messages)),
        ]
        .into_iter()
        .chain(params),
    );
    Value::Object(body)
}

impl ChatCompletionsProviderConfig for OpenAIChatCompletionsConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let base = api_base.unwrap_or("https://api.openai.com/v1");
        let base = base.trim_end_matches('/');
        Ok(format!("{base}/chat/completions"))
    }

    fn auth(
        &self,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<ChatCompletionsAuth> {
        let key = api_key
            .map(|s| s.to_string())
            .or_else(|| env_lookup("OPENAI_API_KEY"))
            .ok_or_else(|| {
                CoreError::Auth("OpenAI API key not found. Set api_key or OPENAI_API_KEY.".into())
            })?;
        Ok(ChatCompletionsAuth::Bearer { token: key })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    fn defers_to_forwarded_auth(&self, _headers: &[(String, String)]) -> bool {
        false
    }

    fn supported_params(&self) -> &'static [&'static str] {
        SUPPORTED_PARAMS
    }

    fn unsupported_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        unsupported_param(SUPPORTED_PARAMS, &[], optional_params)
            .or_else(|| messages.iter().find_map(unsupported_message))
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> CoreResult<ProviderChatRequestData> {
        Ok(ProviderChatRequestData {
            body: openai_body(model, &build_conversation(&messages), optional_params),
        })
    }

    fn transform_response(
        &self,
        _model: &str,
        response: ProviderChatResponseData,
    ) -> CoreResult<ChatCompletionsResponse> {
        let body = response
            .body
            .as_object()
            .ok_or_else(|| CoreError::InvalidResponse("openai response is not an object".into()))?;

        let choices = body
            .get("choices")
            .and_then(Value::as_array)
            .ok_or(CoreError::MissingField("choices"))?;

        let first_choice = choices
            .first()
            .ok_or(CoreError::MissingField("choices[0]"))?;

        let message = first_choice
            .get("message")
            .ok_or(CoreError::MissingField("choices[0].message"))?;

        let content = message
            .get("content")
            .and_then(Value::as_str)
            .map(|s| s.to_string());

        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("assistant");

        let finish_reason = first_choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("");

        let usage = body.get("usage").and_then(Value::as_object);
        let field = |name: &str| -> u64 {
            usage
                .and_then(|u| u.get(name))
                .and_then(Value::as_u64)
                .unwrap_or(0)
        };

        let prompt_details = usage
            .and_then(|u| u.get("prompt_tokens_details"))
            .and_then(Value::as_object);
        let cache_read = prompt_details
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);

        Ok(ChatCompletionsResponse {
            created: body
                .get("created")
                .and_then(Value::as_u64)
                .unwrap_or_else(unix_now),
            model: body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            choices: vec![ChatCompletionsChoice {
                index: first_choice
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                message: ChatCompletionsChoiceMessage {
                    role: role.to_string(),
                    content,
                },
                finish_reason: finish_reason_for(finish_reason).to_string(),
            }],
            usage: usage_from_parts(
                field("prompt_tokens"),
                field("completion_tokens"),
                cache_read,
                0,
            ),
        })
    }
}
