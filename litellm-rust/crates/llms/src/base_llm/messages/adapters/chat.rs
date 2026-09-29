use litellm_llms_types::formats::{
    chat_completions::{ChatCompletionsResponse, ChatMessage, ChatMessageContent},
    messages::{
        ContentBlockType, Message, MessageContent, MessagesRequest, MessagesResponse, SystemPrompt,
    },
};
use serde_json::{Map, Value, json};

use crate::{Error, ErrorDetail};

pub struct PreparedMessagesToChat {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub optional_params: Map<String, Value>,
    response_id: String,
}

impl PreparedMessagesToChat {
    pub fn complete(self, response: ChatCompletionsResponse) -> Result<MessagesResponse, Error> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or(Error::MissingField("choices"))?;
        let stop_reason = match choice.finish_reason.as_str() {
            "stop" => "end_turn",
            "length" => "max_tokens",
            "tool_calls" => "tool_use",
            "content_filter" => "refusal",
            _ => return Err(Error::Unsupported("chat finish reason")),
        };
        let text = choice
            .message
            .content
            .into_iter()
            .map(|text| json!({"type": "text", "text": text}));
        let calls = choice.message.tool_calls.into_iter().flatten().map(|call| {
            let input: Value = serde_json::from_str(&call.function.arguments)
                .map_err(|error| Error::InvalidResponse(ErrorDetail::invalid("tool arguments", error)))?;
            Ok(json!({"type": "tool_use", "id": call.id, "name": call.function.name, "input": input}))
        });
        let content = text
            .map(Ok)
            .chain(calls)
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(MessagesResponse {
            id: self.response_id,
            message_type: "message".into(),
            role: "assistant".into(),
            model: response.model,
            content,
            stop_reason: Some(stop_reason.into()),
            stop_sequence: None,
            usage: Some(json!({
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            })),
            container: None,
            extra: Map::new(),
        })
    }
}

pub fn prepare(
    request: MessagesRequest,
    generate_id: impl FnOnce() -> String,
) -> Result<PreparedMessagesToChat, Error> {
    let MessagesRequest {
        model,
        messages,
        params,
    } = request;
    if messages.is_empty() {
        return Err(Error::MissingField("messages"));
    }
    let max_tokens = params.max_tokens.ok_or(Error::MissingField("max_tokens"))?;
    let unsupported = params.metadata.is_some()
        || params.top_k.is_some()
        || params.thinking.is_some()
        || params.service_tier.is_some()
        || params.container.is_some()
        || params.mcp_servers.is_some()
        || params.context_management.is_some()
        || params.output_format.is_some()
        || params.output_config.is_some()
        || params.speed.is_some()
        || params.inference_geo.is_some()
        || params.reasoning_effort.is_some()
        || params.compaction.is_some()
        || !params.extra.is_empty();
    if unsupported {
        return Err(Error::Unsupported("Messages option through Chat"));
    }
    let system = match params.system {
        None => None,
        Some(SystemPrompt::Text(text)) => Some(chat_message("system", Some(text), Map::new())),
        Some(SystemPrompt::Blocks(blocks)) => Some(chat_message(
            "system",
            Some(
                blocks
                    .into_iter()
                    .map(|block| {
                        if block.block_type != Some(ContentBlockType::Text)
                            || block.cache_control.is_some()
                            || !block.extra.is_empty()
                        {
                            return Err(Error::Unsupported("system blocks through Chat"));
                        }
                        block.text.ok_or(Error::MissingField("system text"))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n"),
            ),
            Map::new(),
        )),
    };
    let translated = messages
        .into_iter()
        .map(translate_message)
        .collect::<Result<Vec<_>, _>>()?;
    let messages = system
        .into_iter()
        .chain(translated.into_iter().flatten())
        .collect();
    let tools = params
        .tools
        .map(|tools| {
            tools
                .into_iter()
                .map(|tool| {
                    let value = serde_json::to_value(tool).map_err(|error| {
                        Error::InvalidRequest(ErrorDetail::invalid("tool", error))
                    })?;
                    let object = value
                        .as_object()
                        .ok_or(Error::Unsupported("non-function Messages tool"))?;
                    if object.contains_key("type")
                        || object.get("name").and_then(Value::as_str).is_none()
                        || object.get("input_schema").is_none()
                        || object
                            .get("name")
                            .and_then(Value::as_str)
                            .is_some_and(|name| name.len() > 64)
                    {
                        return Err(Error::Unsupported("non-function Messages tool"));
                    }
                    let function = Map::from_iter(
                        [
                            ("name".into(), Some(object["name"].clone())),
                            ("description".into(), object.get("description").cloned()),
                            ("parameters".into(), Some(object["input_schema"].clone())),
                        ]
                        .into_iter()
                        .filter_map(|(name, value)| value.map(|value| (name, value))),
                    );
                    Ok(json!({
                        "type": "function",
                        "function": function
                    }))
                })
                .collect::<Result<Vec<_>, Error>>()
        })
        .transpose()?;
    let optional_params = Map::from_iter(
        [
            ("max_tokens", Some(Value::from(max_tokens))),
            ("stream", params.stream.map(Value::from)),
            ("temperature", params.temperature.map(Value::from)),
            ("top_p", params.top_p.map(Value::from)),
            ("stop", params.stop_sequences.map(|stops| json!(stops))),
            ("tools", tools.map(Value::from)),
            (
                "tool_choice",
                params.tool_choice.map(translate_tool_choice).transpose()?,
            ),
        ]
        .into_iter()
        .filter_map(|(name, value)| value.map(|value| (name.into(), value))),
    );
    Ok(PreparedMessagesToChat {
        model,
        messages,
        optional_params,
        response_id: generate_id(),
    })
}

fn translate_message(message: Message) -> Result<Vec<ChatMessage>, Error> {
    if !message.extra.is_empty() {
        return Err(Error::Unsupported("Messages message fields through Chat"));
    }
    match (message.role.as_str(), message.content) {
        ("user" | "assistant", MessageContent::Text(text)) => {
            Ok(vec![chat_message(&message.role, Some(text), Map::new())])
        }
        ("user", MessageContent::Blocks(blocks)) => translate_user_blocks(blocks),
        ("assistant", MessageContent::Blocks(blocks)) => translate_assistant_blocks(blocks),
        _ => Err(Error::Unsupported("Messages role through Chat")),
    }
}

enum UserBlock {
    Text(String),
    Tool(ChatMessage),
}

fn translate_user_blocks(
    blocks: Vec<litellm_llms_types::formats::messages::ContentBlock>,
) -> Result<Vec<ChatMessage>, Error> {
    let converted = blocks
        .into_iter()
        .map(translate_user_block)
        .collect::<Result<Vec<_>, _>>()?;
    let (tools, text): (Vec<_>, Vec<_>) = converted
        .into_iter()
        .partition(|block| matches!(block, UserBlock::Tool(_)));
    let tools = tools.into_iter().filter_map(|block| match block {
        UserBlock::Tool(message) => Some(message),
        UserBlock::Text(_) => None,
    });
    let parts = text
        .into_iter()
        .filter_map(|block| match block {
            UserBlock::Text(text) => Some(json!({"type": "text", "text": text})),
            UserBlock::Tool(_) => None,
        })
        .collect::<Vec<_>>();
    Ok(tools
        .chain((!parts.is_empty()).then(|| ChatMessage {
            role: "user".into(),
            content: Some(ChatMessageContent::Parts(parts)),
            name: None,
            extra: Map::new(),
        }))
        .collect())
}

fn translate_user_block(
    block: litellm_llms_types::formats::messages::ContentBlock,
) -> Result<UserBlock, Error> {
    if !block.extra.is_empty()
        || block.cache_control.is_some()
        || block.provider_specific_fields.is_some()
    {
        return Err(Error::Unsupported("Messages user block through Chat"));
    }
    match block.block_type {
        Some(ContentBlockType::Text) => Ok(UserBlock::Text(
            block.text.ok_or(Error::MissingField("text"))?,
        )),
        Some(ContentBlockType::ToolResult) => {
            let id = block
                .tool_use_id
                .ok_or(Error::MissingField("tool_use_id"))?;
            let content = match block.content {
                Some(Value::String(text)) => text,
                Some(Value::Array(parts)) => match parts.as_slice() {
                    [] => String::new(),
                    [part] if part.get("type") == Some(&Value::String("text".into())) => part
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .ok_or(Error::Unsupported("non-text tool result"))?,
                    _ => return Err(Error::Unsupported("multi-part tool result through Chat")),
                },
                None | Some(Value::Null) => String::new(),
                _ => return Err(Error::Unsupported("tool result content")),
            };
            Ok(UserBlock::Tool(chat_message(
                "tool",
                Some(content),
                Map::from_iter([("tool_call_id".into(), Value::String(id))]),
            )))
        }
        _ => Err(Error::Unsupported("Messages user block through Chat")),
    }
}

fn translate_assistant_blocks(
    blocks: Vec<litellm_llms_types::formats::messages::ContentBlock>,
) -> Result<Vec<ChatMessage>, Error> {
    let converted = blocks.into_iter().map(|block| {
        if !block.extra.is_empty() || block.cache_control.is_some() || block.provider_specific_fields.is_some() {
            return Err(Error::Unsupported("Messages assistant block through Chat"));
        }
        match block.block_type {
            Some(ContentBlockType::Text) => Ok((Some(block.text.ok_or(Error::MissingField("text"))?), None)),
            Some(ContentBlockType::ToolUse) => {
                let id = block.id.ok_or(Error::MissingField("id"))?;
                let name = block.name.ok_or(Error::MissingField("name"))?;
                if name.len() > 64 {
                    return Err(Error::Unsupported("tool name through Chat"));
                }
                Ok((None, Some(json!({
                    "id": id, "type": "function", "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&block.input.unwrap_or_else(|| json!({})))
                            .map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("tool input", error)))?,
                    }
                }))))
            }
            _ => Err(Error::Unsupported("Messages assistant block through Chat")),
        }
    }).collect::<Result<Vec<_>, Error>>()?;
    let text = converted
        .iter()
        .filter_map(|(text, _)| text.as_deref())
        .collect::<String>();
    let calls = converted
        .into_iter()
        .filter_map(|(_, call)| call)
        .collect::<Vec<_>>();
    let extra = if calls.is_empty() {
        Map::new()
    } else {
        Map::from_iter([("tool_calls".into(), Value::Array(calls))])
    };
    Ok(vec![chat_message(
        "assistant",
        (!text.is_empty()).then_some(text),
        extra,
    )])
}

fn translate_tool_choice(value: Value) -> Result<Value, Error> {
    let object = value
        .as_object()
        .ok_or(Error::Unsupported("Messages tool choice through Chat"))?;
    match object.get("type").and_then(Value::as_str) {
        Some("auto") if object.len() == 1 => Ok(json!("auto")),
        Some("any") if object.len() == 1 => Ok(json!("required")),
        Some("none") if object.len() == 1 => Ok(json!("none")),
        Some("tool") if object.len() == 2 => object
            .get("name")
            .and_then(Value::as_str)
            .map(|name| json!({"type": "function", "function": {"name": name}}))
            .ok_or(Error::Unsupported("Messages tool choice through Chat")),
        _ => Err(Error::Unsupported("Messages tool choice through Chat")),
    }
}

fn chat_message(role: &str, content: Option<String>, extra: Map<String, Value>) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.map(ChatMessageContent::Text),
        name: None,
        extra,
    }
}
