use std::sync::OnceLock;

use serde_json::{Map, Value, json};

use crate::Error;
use crate::chat_completions::types::{
    ChatCompletionsRequest, ChatCompletionsResponse, ChatMessage, ChatMessageContent,
};
use crate::chat_completions::{
    chat_completions, chat_completions_decline_reason, chat_completions_with_config,
};
use crate::http_utils::{has_header, http_request, string_headers, truncate_error_body};
use crate::providers::openai::chat_completions::transformation::OPENAI_CHAT_COMPLETIONS_CONFIG;
use crate::routing_utils::provider::get_custom_llm_provider;

use super::http_types::{
    ResponsesApiResponse, ResponsesInput, ResponsesInputContent, ResponsesOutputContent,
    ResponsesOutputItem, ResponsesRequest,
};

const SUPPORTED_PARAMS: &[&str] = &["instructions", "max_output_tokens", "temperature", "top_p"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponsesTransport {
    Native,
    ChatCompletionsAdapter,
}

fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

fn resolve_provider<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<(&'a str, &'a str), Error> {
    get_custom_llm_provider(model, custom_llm_provider)
        .map(|provider| (provider.model, provider.custom_llm_provider))
        .ok_or_else(|| Error::InvalidProvider("unable to resolve Responses provider".to_string()))
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn responses_transport(
    provider: &str,
    use_chat_completions_api: bool,
) -> Result<ResponsesTransport, Error> {
    if use_chat_completions_api {
        return match provider {
            "openai" | "anthropic" | "bedrock" => Ok(ResponsesTransport::ChatCompletionsAdapter),
            _ => Err(Error::Unsupported("provider has no Rust Responses adapter")),
        };
    }
    match provider {
        "openai" => Ok(ResponsesTransport::Native),
        "anthropic" | "bedrock" => Ok(ResponsesTransport::ChatCompletionsAdapter),
        _ => Err(Error::Unsupported(
            "provider has no Rust Responses transport",
        )),
    }
}

fn parse_input(input: Value) -> Result<ResponsesInput, Error> {
    serde_json::from_value(input)
        .map_err(|error| Error::InvalidRequest(format!("invalid Responses input: {error}")))
}

fn unsupported_reason(
    input: &ResponsesInput,
    optional_params: &Map<String, Value>,
) -> Option<&'static str> {
    if optional_params
        .keys()
        .any(|key| !SUPPORTED_PARAMS.contains(&key.as_str()))
    {
        return Some("unrecognized Responses parameter");
    }
    let ResponsesInput::Items(items) = input else {
        return None;
    };
    if items.is_empty() {
        return Some("empty Responses input");
    }
    items.iter().find_map(|item| {
        if !item.extra.is_empty() || !matches!(item.role.as_str(), "system" | "user" | "assistant")
        {
            return Some("unsupported Responses input item");
        }
        match &item.content {
            ResponsesInputContent::Text(_) => None,
            ResponsesInputContent::Parts(parts)
                if !parts.is_empty()
                    && parts.iter().all(|part| {
                        matches!(
                            part.part_type.as_str(),
                            "input_text" | "output_text" | "text"
                        ) && part.extra.is_empty()
                    }) =>
            {
                None
            }
            ResponsesInputContent::Parts(_) => Some("unsupported Responses input content"),
        }
    })
}

pub fn responses_decline_reason(
    model: &str,
    input: Value,
    optional_params: &Map<String, Value>,
    custom_llm_provider: Option<&str>,
    use_chat_completions_api: bool,
) -> Option<&'static str> {
    let Ok((resolved_model, provider)) = resolve_provider(model, custom_llm_provider) else {
        return Some("unable to resolve Responses provider");
    };
    let Ok(transport) = responses_transport(provider, use_chat_completions_api) else {
        return Some("provider has no Rust Responses transport");
    };
    let Ok(input) = parse_input(input) else {
        return Some("invalid Responses input");
    };
    if let Some(reason) = unsupported_reason(&input, optional_params) {
        return Some(reason);
    }
    if transport == ResponsesTransport::ChatCompletionsAdapter {
        if provider == "openai" {
            return None;
        }
        let messages = adapter_messages(
            &input,
            optional_params.get("instructions").and_then(Value::as_str),
        );
        return chat_completions_decline_reason(
            resolved_model,
            Some(provider),
            json!(messages),
            &adapter_params(provider, optional_params),
        );
    }
    None
}

fn message_text(content: &ResponsesInputContent) -> String {
    match content {
        ResponsesInputContent::Text(text) => text.clone(),
        ResponsesInputContent::Parts(parts) => {
            parts.iter().map(|part| part.text.as_str()).collect()
        }
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
fn transform_responses_request_to_chat_completions(
    input: &ResponsesInput,
    instructions: Option<&str>,
) -> Vec<ChatMessage> {
    adapter_messages(input, instructions)
}

fn adapter_messages(input: &ResponsesInput, instructions: Option<&str>) -> Vec<ChatMessage> {
    let instruction = instructions.map(|text| ChatMessage {
        role: "system".to_string(),
        content: Some(ChatMessageContent::Text(text.to_string())),
        name: None,
        extra: Map::new(),
    });
    let messages = match input {
        ResponsesInput::Text(text) => vec![ChatMessage {
            role: "user".to_string(),
            content: Some(ChatMessageContent::Text(text.clone())),
            name: None,
            extra: Map::new(),
        }],
        ResponsesInput::Items(items) => items
            .iter()
            .map(|item| ChatMessage {
                role: item.role.clone(),
                content: Some(ChatMessageContent::Text(message_text(&item.content))),
                name: None,
                extra: Map::new(),
            })
            .collect(),
    };
    instruction.into_iter().chain(messages).collect()
}

fn adapter_params(provider: &str, optional_params: &Map<String, Value>) -> Map<String, Value> {
    Map::from_iter(
        optional_params
            .iter()
            .filter(|(key, _)| key.as_str() != "instructions")
            .map(|(key, value)| {
                let mapped = match (provider, key.as_str()) {
                    ("anthropic", "max_output_tokens") => "max_tokens",
                    ("bedrock", "max_output_tokens") => "maxTokens",
                    ("openai", "max_output_tokens") => "max_completion_tokens",
                    (_, name) => name,
                };
                (mapped.to_string(), value.clone())
            }),
    )
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
fn transform_chat_completions_response_to_responses(
    response: ChatCompletionsResponse,
    provider: &str,
) -> ResponsesApiResponse {
    let status = response
        .choices
        .first()
        .map_or("completed", |choice| match choice.finish_reason.as_str() {
            "length" => "incomplete",
            "content_filter" => "failed",
            _ => "completed",
        })
        .to_string();
    let output = response
        .choices
        .into_iter()
        .filter_map(|choice| {
            choice
                .message
                .content
                .map(|text| ResponsesOutputItem::Message {
                    id: format!("msg_{:016x}", rand::random::<u64>()),
                    status: status.clone(),
                    role: choice.message.role,
                    content: vec![ResponsesOutputContent::OutputText {
                        text,
                        annotations: vec![],
                        extra: Map::new(),
                    }],
                    extra: Map::new(),
                })
        })
        .collect();
    ResponsesApiResponse {
        id: format!("resp_{:016x}", rand::random::<u64>()),
        created_at: response.created,
        output,
        extra: Map::from_iter(
            [
                ("model".to_string(), json!(response.model)),
                ("object".to_string(), json!("response")),
                ("status".to_string(), json!(status)),
                ("metadata".to_string(), json!({})),
                ("parallel_tool_calls".to_string(), json!(false)),
                ("temperature".to_string(), json!(0.0)),
                ("tool_choice".to_string(), json!("auto")),
                ("tools".to_string(), json!([])),
                ("text".to_string(), json!({})),
                (
                    "usage".to_string(),
                    if provider == "openai" {
                        json!({
                            "input_tokens": response.usage.prompt_tokens,
                            "output_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                            "cost": null,
                            "input_tokens_details": null,
                            "output_tokens_details": null,
                        })
                    } else {
                        json!({
                            "input_tokens": response.usage.prompt_tokens,
                            "output_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                            "cost": null,
                            "input_tokens_details": {
                                "audio_tokens": null,
                                "cached_tokens": response.usage.prompt_tokens_details.cached_tokens,
                                "text_tokens": response.usage.prompt_tokens_details.text_tokens,
                                "cache_write_tokens": response.usage.prompt_tokens_details.cache_creation_tokens,
                            },
                            "output_tokens_details": {
                                "audio_tokens": null,
                                "reasoning_tokens": 0,
                                "text_tokens": response.usage.completion_tokens,
                            },
                        })
                    },
                ),
            ]
            .into_iter()
            .chain((provider == "anthropic").then(|| {
                (
                    "provider_specific_fields".to_string(),
                    json!({"citations": null, "thinking_blocks": null}),
                )
            })),
        ),
    }
}

async fn adapter_response(
    request: ResponsesRequest<'_>,
    model: &str,
    provider: &str,
    input: ResponsesInput,
) -> Result<ResponsesApiResponse, Error> {
    let messages = transform_responses_request_to_chat_completions(
        &input,
        request
            .optional_params
            .get("instructions")
            .and_then(Value::as_str),
    );
    let optional_params = adapter_params(provider, &request.optional_params);
    let chat_request = ChatCompletionsRequest {
        model,
        messages: json!(messages),
        optional_params,
        api_key: request.api_key,
        api_base: request.api_base,
        custom_llm_provider: Some(provider),
        extra_headers: request.extra_headers,
        timeout: request.timeout,
    };
    let response = if provider == "openai" {
        chat_completions_with_config(chat_request, model, &OPENAI_CHAT_COMPLETIONS_CONFIG).await?
    } else {
        chat_completions(chat_request).await?
    };
    Ok(transform_chat_completions_response_to_responses(
        response, provider,
    ))
}

fn native_url(api_base: Option<&str>) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("https://api.openai.com/v1")
        .trim_end_matches('/');
    if base.ends_with("/responses") {
        return base.to_string();
    }
    format!("{base}/responses")
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
async fn execute_responses_provider_call(
    request: ResponsesRequest<'_>,
    model: &str,
    input: ResponsesInput,
) -> Result<ResponsesApiResponse, Error> {
    let api_key = request
        .api_key
        .map(str::to_string)
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| Error::Auth("OPENAI_API_KEY is required".to_string()))?;
    let body = Value::Object(Map::from_iter(
        [
            ("model".to_string(), json!(model)),
            ("input".to_string(), json!(input)),
        ]
        .into_iter()
        .chain(request.optional_params),
    ));
    let bytes = serde_json::to_vec(&body).map_err(|error| {
        Error::InvalidRequest(format!("failed to serialize Responses request: {error}"))
    })?;
    let mut headers = string_headers("Responses", request.extra_headers)?;
    headers.retain(|(name, _)| !name.eq_ignore_ascii_case("authorization"));
    headers.push(("authorization".to_string(), format!("Bearer {api_key}")));
    if !has_header(&headers, "content-type") {
        headers.push(("content-type".to_string(), "application/json".to_string()));
    }
    let mut builder = http_client().post(native_url(request.api_base)).body(bytes);
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    if let Some(timeout) = request.timeout {
        builder = builder.timeout(timeout);
    }
    let response = http_request(builder).await.map_err(|error| {
        if error.is_connect() || error.is_builder() {
            Error::Connect(error.to_string())
        } else {
            Error::Network(error.to_string())
        }
    })?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|error| Error::Network(error.to_string()))?;
    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }
    serde_json::from_str(&text).map_err(|error| {
        Error::InvalidResponse(format!("invalid Responses response JSON: {error}"))
    })
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn responses(request: ResponsesRequest<'_>) -> Result<ResponsesApiResponse, Error> {
    let (model, provider) = resolve_provider(request.model, request.custom_llm_provider)?;
    let transport = responses_transport(provider, request.use_chat_completions_api)?;
    let input = parse_input(request.input.clone())?;
    if let Some(reason) = unsupported_reason(&input, &request.optional_params) {
        return Err(Error::Unsupported(reason));
    }
    match transport {
        ResponsesTransport::Native => execute_responses_provider_call(request, model, input).await,
        ResponsesTransport::ChatCompletionsAdapter => {
            adapter_response(request, model, provider, input).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_selects_native_and_adapter_transports() {
        assert_eq!(
            responses_transport("openai", false),
            Ok(ResponsesTransport::Native)
        );
        assert_eq!(
            responses_transport("openai", true),
            Ok(ResponsesTransport::ChatCompletionsAdapter)
        );
        assert_eq!(
            responses_transport("anthropic", false),
            Ok(ResponsesTransport::ChatCompletionsAdapter)
        );
        assert_eq!(
            responses_transport("bedrock", false),
            Ok(ResponsesTransport::ChatCompletionsAdapter)
        );
    }

    #[test]
    fn unsupported_state_declines_before_provider_call() {
        let reason = responses_decline_reason(
            "gpt-5",
            json!("hello"),
            &Map::from_iter([("previous_response_id".to_string(), json!("resp_1"))]),
            Some("openai"),
            false,
        );
        assert_eq!(reason, Some("unrecognized Responses parameter"));
    }
}
