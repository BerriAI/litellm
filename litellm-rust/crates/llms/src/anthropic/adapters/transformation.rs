//! `litellm/llms/anthropic/pass_through/adapters/transformation.py`: a Messages request for a
//! provider that only speaks Chat Completions, and that provider's answer read back as Messages.

use std::collections::HashMap;

use litellm_llms_types::{
    formats::messages::{
        ContentBlock, ContentBlockType, Message, MessageContent, MessagesRequest, MessagesResponse,
        MessagesTool, SystemPrompt, ThinkingConfig, streaming::MessagesStreamUsage,
    },
    recognized::Recognized,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::{
    Error, ErrorDetail, anthropic::common_utils::normalize_anthropic_tool_use_id,
    openai::chat::compatible::reasoning_effort_from_thinking_budget,
};

/// `OPENAI_MAX_TOOL_NAME_LENGTH`, and the prefix and hash a longer name is cut to.
const MAX_TOOL_NAME_LENGTH: usize = 64;
const TOOL_NAME_PREFIX_LENGTH: usize = 55;
const TOOL_NAME_HASH_LENGTH: usize = 8;

/// The effort an adaptive thinking request gets when it names none.
const ADAPTIVE_EFFORT: &str = "medium";

pub struct ChatRequest {
    /// The Chat Completions body, `model` included.
    pub body: Map<String, Value>,
    pub tool_names: ToolNames,
}

/// The original name of each tool whose name was cut to fit the Chat Completions limit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolNames(HashMap<String, String>);

impl ToolNames {
    pub fn original<'a>(&'a self, name: &'a str) -> &'a str {
        self.0.get(name).map_or(name, String::as_str)
    }
}

pub fn translate_anthropic_to_openai(request: MessagesRequest) -> Result<ChatRequest, Error> {
    let MessagesRequest {
        model,
        messages,
        params,
    } = request;
    let messages: Vec<Value> = params
        .system
        .and_then(system_message)
        .into_iter()
        .chain(messages.into_iter().flat_map(translate_message))
        .collect();
    let mut body = Map::from_iter([
        ("model".to_owned(), Value::String(model)),
        ("messages".to_owned(), Value::Array(messages)),
    ]);
    let (tools, web_search, tool_names) = translate_tools(params.tools.unwrap_or_default());
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools));
    }
    if web_search {
        body.insert("web_search_options".into(), json!({}));
    }
    if let Some(choice) = params.tool_choice.filter(|choice| !choice.is_null()) {
        if choice.get("disable_parallel_tool_use") == Some(&Value::Bool(true)) {
            body.insert("parallel_tool_calls".into(), Value::Bool(false));
        }
        body.insert("tool_choice".into(), translate_tool_choice(choice)?);
    }
    let effort = match params.reasoning_effort {
        Some(effort) => Some(serde_json::to_value(effort).map_err(invalid_request)?),
        None => thinking_effort(params.thinking, params.output_config).map(Value::from),
    };
    let user = params
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("user_id"))
        .filter(|user| user.is_string())
        .cloned();
    let stop = params.stop_sequences.filter(|stop| !stop.is_empty());
    let stream = params.stream;
    let copied = [
        ("max_completion_tokens", params.max_tokens.map(Value::from)),
        ("temperature", params.temperature.map(Value::from)),
        ("top_p", params.top_p.map(Value::from)),
        ("top_k", params.top_k.map(Value::from)),
        ("stop", stop.map(Value::from)),
        ("reasoning_effort", effort),
        ("user", user),
        ("stream", stream.map(Value::from)),
        (
            "stream_options",
            (stream == Some(true)).then(|| json!({"include_usage": true})),
        ),
    ];
    body.extend(
        copied
            .into_iter()
            .filter_map(|(name, value)| Some((name.to_owned(), value?))),
    );
    for (name, value) in params.extra {
        body.entry(name).or_insert(value);
    }
    Ok(ChatRequest { body, tool_names })
}

/// Cuts a name past the Chat Completions limit to a prefix and a hash of the whole name, so two
/// long names that share a prefix stay apart.
pub fn truncate_tool_name(name: &str) -> String {
    if name.chars().count() <= MAX_TOOL_NAME_LENGTH {
        return name.to_owned();
    }
    let hash = Sha256::digest(name.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let prefix: String = name.chars().take(TOOL_NAME_PREFIX_LENGTH).collect();
    format!("{prefix}_{}", &hash[..TOOL_NAME_HASH_LENGTH])
}

fn system_message(system: SystemPrompt) -> Option<Value> {
    match system {
        SystemPrompt::Text(text) if text.is_empty() => None,
        SystemPrompt::Text(text) => Some(json!({"role": "system", "content": text})),
        SystemPrompt::Blocks(blocks) => {
            let parts = text_parts(&blocks);
            (!parts.is_empty()).then(|| json!({"role": "system", "content": parts}))
        }
    }
}

fn text_parts(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks
        .iter()
        .filter(|block| block.is_type(ContentBlockType::Text))
        .map(|block| text_part(block.text.as_deref().unwrap_or_default()))
        .collect()
}

fn text_part(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

fn translate_message(message: Message) -> Vec<Value> {
    let blocks = match message.content {
        MessageContent::Text(text) => {
            return vec![json!({"role": message.role, "content": text})];
        }
        MessageContent::Blocks(blocks) => blocks,
    };
    match message.role.as_str() {
        "assistant" => assistant_message(&blocks).into_iter().collect(),
        "system" => {
            let parts = text_parts(&blocks);
            (!parts.is_empty())
                .then(|| json!({"role": "system", "content": parts}))
                .into_iter()
                .collect()
        }
        _ => user_messages(&blocks),
    }
}

/// A user turn's tool results become `tool` messages ahead of whatever else the turn says, as
/// Chat Completions wants each result right after the assistant turn that called it.
fn user_messages(blocks: &[ContentBlock]) -> Vec<Value> {
    let parts: Vec<Value> = blocks.iter().filter_map(user_part).collect();
    blocks
        .iter()
        .filter(|block| block.is_type(ContentBlockType::ToolResult))
        .map(tool_message)
        .chain((!parts.is_empty()).then(|| json!({"role": "user", "content": parts})))
        .collect()
}

fn user_part(block: &ContentBlock) -> Option<Value> {
    match block.block_type.as_ref()? {
        ContentBlockType::Text => Some(text_part(block.text.as_deref().unwrap_or_default())),
        ContentBlockType::Other(kind) => media_part(kind, block.extra.get("source")?),
        _ => None,
    }
}

/// An image as an `image_url` part, and a base64 document as a `file` part, the shape OpenAI
/// takes PDFs in.
fn media_part(kind: &str, source: &Value) -> Option<Value> {
    let url = source_url(source)?;
    match (kind, source.get("type").and_then(Value::as_str)) {
        ("image", _) => Some(json!({"type": "image_url", "image_url": {"url": url}})),
        ("document", Some("base64")) => Some(json!({"type": "file", "file": {"file_data": url}})),
        ("document", _) => Some(json!({"type": "image_url", "image_url": {"url": url}})),
        _ => None,
    }
}

/// `anthropic_image_source_to_openai_url`.
fn source_url(source: &Value) -> Option<String> {
    let field = |name: &str| source.get(name).and_then(Value::as_str);
    match field("type")? {
        "base64" => {
            let data = field("data").filter(|data| !data.is_empty())?;
            let media_type = field("media_type").unwrap_or("image/jpeg");
            Some(format!("data:{media_type};base64,{data}"))
        }
        "url" => Some(field("url").unwrap_or_default().to_owned()),
        _ => None,
    }
}

fn tool_message(block: &ContentBlock) -> Value {
    json!({
        "role": "tool",
        "tool_call_id": block.tool_use_id.as_deref().unwrap_or_default(),
        "content": tool_result_content(block.content.as_ref()),
    })
}

/// A lone text result stays a string; anything richer becomes content parts.
fn tool_result_content(content: Option<&Value>) -> Value {
    let items = match content {
        Some(Value::String(text)) => return Value::String(text.clone()),
        Some(Value::Array(items)) => items,
        _ => return Value::String(String::new()),
    };
    let parts: Vec<Value> = items.iter().filter_map(tool_result_part).collect();
    match parts.as_slice() {
        [] => Value::String(String::new()),
        [part] if part["type"] == "text" => part["text"].clone(),
        _ => Value::Array(parts),
    }
}

fn tool_result_part(item: &Value) -> Option<Value> {
    if let Value::String(text) = item {
        return Some(text_part(text));
    }
    match item.get("type")?.as_str()? {
        "text" => Some(text_part(
            item.get("text").and_then(Value::as_str).unwrap_or_default(),
        )),
        "image" | "document" => {
            let url = source_url(item.get("source")?)?;
            Some(json!({"type": "image_url", "image_url": {"url": url}}))
        }
        _ => None,
    }
}

/// Thinking blocks are dropped: a Chat Completions host has no field to take them back in.
fn assistant_message(blocks: &[ContentBlock]) -> Option<Value> {
    let texts: Vec<&str> = blocks
        .iter()
        .filter(|block| block.is_type(ContentBlockType::Text))
        .map(|block| block.text.as_deref().unwrap_or_default())
        .collect();
    let tool_calls: Vec<Value> = blocks
        .iter()
        .filter(|block| block.is_type(ContentBlockType::ToolUse))
        .map(|block| {
            json!({
                "id": block.id.as_deref().unwrap_or_default(),
                "type": "function",
                "function": {
                    "name": truncate_tool_name(block.name.as_deref().unwrap_or_default()),
                    "arguments": block.input.clone().unwrap_or_else(|| json!({})).to_string(),
                },
            })
        })
        .collect();
    if texts.is_empty() && tool_calls.is_empty() {
        return None;
    }
    let content = (!texts.is_empty()).then(|| texts.concat());
    let mut message = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    Some(message)
}

/// The function tools, whether a web search tool asked for `web_search_options`, and the names
/// cut to fit. Anthropic's own server tools have no Chat Completions counterpart and are dropped.
fn translate_tools(tools: Vec<Recognized<MessagesTool>>) -> (Vec<Value>, bool, ToolNames) {
    let custom: Vec<(usize, Map<String, Value>)> = tools
        .into_iter()
        .enumerate()
        .filter_map(|(index, tool)| match tool {
            Recognized::Unrecognized(Value::Object(tool)) => Some((index, tool)),
            _ => None,
        })
        .collect();
    let web_search = custom.iter().any(|(_, tool)| is_web_search(tool));
    let (functions, renamed): (Vec<Value>, Vec<Option<(String, String)>>) = custom
        .into_iter()
        .filter(|(_, tool)| !is_web_search(tool))
        .map(|(index, tool)| function_tool(index, tool))
        .unzip();
    (
        functions,
        web_search,
        ToolNames(renamed.into_iter().flatten().collect()),
    )
}

fn is_web_search(tool: &Map<String, Value>) -> bool {
    let text = |name: &str| tool.get(name).and_then(Value::as_str);
    text("type").is_some_and(|kind| kind.starts_with("web_search"))
        || (text("name") == Some("web_search") && !tool.contains_key("input_schema"))
}

/// A tool already in Chat Completions shape passes as is; the truncated name, when there is
/// one, comes back paired with the original.
fn function_tool(index: usize, tool: Map<String, Value>) -> (Value, Option<(String, String)>) {
    if tool.get("type").and_then(Value::as_str) == Some("function") && tool.contains_key("function")
    {
        return (Value::Object(tool), None);
    }
    let original = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .map_or_else(|| format!("litellm_unnamed_tool_{index}"), str::to_owned);
    let name = truncate_tool_name(&original);
    let function: Map<String, Value> = [
        ("name", Some(Value::String(name.clone()))),
        ("description", tool.get("description").cloned()),
        ("parameters", tool.get("input_schema").cloned()),
        ("strict", tool.get("strict").cloned()),
    ]
    .into_iter()
    .filter_map(|(key, value)| Some((key.to_owned(), value?)))
    .collect();
    let renamed = (name != original).then_some((name, original));
    (json!({"type": "function", "function": function}), renamed)
}

fn translate_tool_choice(choice: Value) -> Result<Value, Error> {
    let name = |choice: &Value| {
        choice
            .get("name")
            .and_then(Value::as_str)
            .map(truncate_tool_name)
            .unwrap_or_default()
    };
    match choice.get("type").and_then(Value::as_str) {
        Some("any") => Ok(json!("required")),
        Some("auto") => Ok(json!("auto")),
        Some("none") => Ok(json!("none")),
        Some("tool") => Ok(json!({"type": "function", "function": {"name": name(&choice)}})),
        _ => Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
            field: "tool_choice",
            expected: "a tool choice of type auto, any, tool or none",
            actual: choice,
        })),
    }
}

/// `reasoning_effort` for a thinking request: the budget's tier, or an adaptive request's own
/// effort. Disabled thinking sends nothing, which a model without reasoning also accepts.
fn thinking_effort(
    thinking: Option<Recognized<ThinkingConfig>>,
    output_config: Option<Recognized<litellm_llms_types::formats::messages::OutputConfig>>,
) -> Option<String> {
    match thinking? {
        Recognized::Known(ThinkingConfig::Enabled(enabled)) => {
            let budget = match enabled.budget_tokens? {
                Recognized::Known(budget) => i64::try_from(budget).unwrap_or(i64::MAX),
                Recognized::Unrecognized(_) => return None,
            };
            Some(reasoning_effort_from_thinking_budget(budget).to_owned())
        }
        Recognized::Known(ThinkingConfig::Adaptive(_)) => Some(
            output_config
                .as_ref()
                .and_then(Recognized::known)
                .and_then(|config| config.effort.as_ref())
                .and_then(|effort| serde_json::to_value(effort).ok())
                .and_then(|effort| effort.as_str().map(str::to_owned))
                .unwrap_or_else(|| ADAPTIVE_EFFORT.to_owned()),
        ),
        _ => None,
    }
}

fn invalid_request(error: serde_json::Error) -> Error {
    Error::InvalidRequest(ErrorDetail::invalid("Anthropic messages request", error))
}

#[derive(Deserialize)]
struct ChatResponse {
    id: Option<String>,
    model: Option<String>,
    #[serde(default)]
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    #[serde(default)]
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct ChatMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    refusal: Option<String>,
    tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Default, Deserialize)]
pub(super) struct ChatToolCall {
    pub(super) index: Option<u64>,
    pub(super) id: Option<String>,
    #[serde(default)]
    pub(super) function: ChatFunction,
}

#[derive(Default, Deserialize)]
pub(super) struct ChatFunction {
    pub(super) name: Option<String>,
    pub(super) arguments: Option<String>,
}

/// The usage counts hosts send, cache reads under whichever name the host uses.
#[derive(Clone, Default, Deserialize)]
pub(super) struct ChatUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Clone, Default, Deserialize)]
struct PromptTokensDetails {
    cached_tokens: Option<u64>,
    cache_creation_tokens: Option<u64>,
    cache_write_tokens: Option<u64>,
}

impl ChatUsage {
    /// `translate_openai_usage_to_anthropic_usage_delta`: Chat Completions counts cache reads
    /// and writes inside `prompt_tokens`, Messages counts them apart from `input_tokens`.
    pub(super) fn anthropic(&self) -> MessagesStreamUsage {
        let positive = |count: Option<u64>| count.filter(|count| *count > 0);
        let details = self.prompt_tokens_details.as_ref();
        let read =
            positive(self.cache_read_input_tokens).or_else(|| positive(details?.cached_tokens));
        let written = positive(self.cache_creation_input_tokens)
            .or_else(|| positive(details?.cache_creation_tokens))
            .or_else(|| positive(details?.cache_write_tokens));
        let cached = read.unwrap_or(0).saturating_add(written.unwrap_or(0));
        MessagesStreamUsage {
            input_tokens: Some(self.prompt_tokens.unwrap_or(0).saturating_sub(cached)),
            output_tokens: Some(self.completion_tokens.unwrap_or(0)),
            cache_creation_input_tokens: written,
            cache_read_input_tokens: read,
            ..MessagesStreamUsage::default()
        }
    }
}

pub fn translate_openai_response_to_anthropic(
    body: &[u8],
    model: &str,
    tool_names: &ToolNames,
) -> Result<MessagesResponse, Error> {
    let response: ChatResponse = serde_json::from_slice(body)
        .map_err(|error| Error::InvalidResponse(ErrorDetail::invalid("chat completion", error)))?;
    let (message, finish_reason) = response
        .choices
        .into_iter()
        .next()
        .map(|choice| (choice.message, choice.finish_reason))
        .unwrap_or_default();
    let refused = message
        .refusal
        .as_deref()
        .is_some_and(|text| !text.is_empty());
    let thinking = message
        .reasoning_content
        .filter(|text| !text.is_empty())
        .map(|text| json!({"type": "thinking", "thinking": text, "signature": ""}));
    let texts = [message.content, message.refusal]
        .into_iter()
        .flatten()
        .filter(|text| !text.is_empty())
        .map(|text| json!({"type": "text", "text": text}));
    let tool_uses = message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|call| tool_use(call, tool_names))
        .collect::<Result<Vec<_>, _>>()?;
    let content = thinking.into_iter().chain(texts).chain(tool_uses).collect();
    let usage = response.usage.unwrap_or_default().anthropic();
    Ok(MessagesResponse {
        id: response.id.unwrap_or_else(message_id),
        message_type: "message".into(),
        role: "assistant".into(),
        model: response
            .model
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| model.to_owned()),
        content,
        stop_reason: Some(stop_reason(finish_reason.as_deref(), refused).into()),
        stop_sequence: None,
        usage: Some(serde_json::to_value(usage).map_err(invalid_response)?),
        container: None,
        extra: Map::new(),
    })
}

fn tool_use(call: ChatToolCall, tool_names: &ToolNames) -> Result<Value, Error> {
    let name = call.function.name.unwrap_or_default();
    Ok(json!({
        "type": "tool_use",
        "id": normalize_anthropic_tool_use_id(call.id.as_deref().unwrap_or_default()),
        "name": tool_names.original(&name),
        "input": tool_input(call.function.arguments.as_deref())?,
    }))
}

/// `parse_tool_call_arguments`, without its repair of truncated JSON.
fn tool_input(arguments: Option<&str>) -> Result<Value, Error> {
    match arguments
        .map(str::trim)
        .filter(|arguments| !arguments.is_empty())
    {
        None => Ok(json!({})),
        Some(arguments) => serde_json::from_str(arguments).map_err(|error| {
            Error::InvalidResponse(ErrorDetail::invalid("tool call arguments", error))
        }),
    }
}

/// `_translate_openai_finish_reason_to_anthropic`, with a refusal winning over anything but a
/// cut-off answer.
pub(super) fn stop_reason(finish_reason: Option<&str>, refused: bool) -> &'static str {
    let mapped = match finish_reason {
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        Some("content_filter" | "refusal") => "refusal",
        _ => "end_turn",
    };
    match (mapped, refused) {
        ("max_tokens", _) | (_, false) => mapped,
        (_, true) => "refusal",
    }
}

pub(super) fn message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

pub(super) fn invalid_response(error: serde_json::Error) -> Error {
    Error::InvalidResponse(ErrorDetail::invalid("Anthropic messages response", error))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    const LONG_NAME: &str =
        "mcp__workspace_server__search_every_document_in_the_shared_drive_by_title";

    fn translate(request: Value) -> ChatRequest {
        translate_anthropic_to_openai(serde_json::from_value(request).unwrap()).unwrap()
    }

    #[rstest]
    fn a_conversation_becomes_the_chat_body() {
        let image = "iVBORw0KGgo=";
        let translated = translate(json!({
            "model": "gpt-6-astra",
            "max_tokens": 256,
            "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "what is in this picture?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "look closer", "signature": "sig"},
                    {"type": "text", "text": "Let me "},
                    {"type": "text", "text": "check."},
                    {"type": "tool_use", "id": "toolu_1", "name": LONG_NAME, "input": {"title": "cat"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a cat"}]},
                    {"type": "text", "text": "thanks"},
                ]},
                {"role": "assistant", "content": "It is a cat."},
                {"role": "user", "content": "and the color?"},
            ],
            "tools": [
                {"name": LONG_NAME, "description": "find", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
            ],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
            "metadata": {"user_id": "user-7"},
            "stop_sequences": ["END"],
            "temperature": 0.5,
            "stream": true,
        }));
        let short_name = truncate_tool_name(LONG_NAME);
        assert_eq!(
            Value::Object(translated.body),
            json!({
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "be brief"}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "what is in this picture?"},
                        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{image}")}},
                    ]},
                    {"role": "assistant", "content": "Let me check.", "tool_calls": [
                        {"id": "toolu_1", "type": "function",
                         "function": {"name": short_name, "arguments": "{\"title\":\"cat\"}"}},
                    ]},
                    {"role": "tool", "tool_call_id": "toolu_1", "content": "a cat"},
                    {"role": "user", "content": [{"type": "text", "text": "thanks"}]},
                    {"role": "assistant", "content": "It is a cat."},
                    {"role": "user", "content": "and the color?"},
                ],
                "tools": [{"type": "function", "function": {
                    "name": short_name, "description": "find", "parameters": {"type": "object"},
                }}],
                "web_search_options": {},
                "parallel_tool_calls": false,
                "tool_choice": "required",
                "max_completion_tokens": 256,
                "temperature": 0.5,
                "stop": ["END"],
                "user": "user-7",
                "stream": true,
                "stream_options": {"include_usage": true},
            })
        );
        assert_eq!(translated.tool_names.original(&short_name), LONG_NAME);
    }

    #[rstest]
    #[case::short("get_weather", "get_weather")]
    #[case::at_the_limit(&"a".repeat(64), &"a".repeat(64))]
    fn names_within_the_limit_are_kept(#[case] name: &str, #[case] expected: &str) {
        assert_eq!(truncate_tool_name(name), expected);
    }

    #[rstest]
    fn long_names_sharing_a_prefix_stay_apart_within_the_limit() {
        let first = truncate_tool_name(&format!("{}_one", "a".repeat(70)));
        let second = truncate_tool_name(&format!("{}_two", "a".repeat(70)));
        assert_ne!(first, second);
        assert_eq!(first.chars().count(), 64);
        assert!(first.starts_with(&"a".repeat(55)));
    }

    #[rstest]
    #[case::auto(json!({"type": "auto"}), json!("auto"))]
    #[case::any(json!({"type": "any"}), json!("required"))]
    #[case::none(json!({"type": "none"}), json!("none"))]
    #[case::tool(
        json!({"type": "tool", "name": "get_weather"}),
        json!({"type": "function", "function": {"name": "get_weather"}})
    )]
    fn tool_choice_maps_to_chat(#[case] choice: Value, #[case] expected: Value) {
        let translated = translate(json!({"model": "m", "messages": [], "tool_choice": choice}));
        assert_eq!(translated.body["tool_choice"], expected);
        assert!(!translated.body.contains_key("parallel_tool_calls"));
    }

    #[rstest]
    fn an_unknown_tool_choice_is_refused() {
        let request = json!({"model": "m", "messages": [], "tool_choice": {"type": "sometimes"}});
        assert!(matches!(
            translate_anthropic_to_openai(serde_json::from_value(request).unwrap()),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[case::enabled_low(json!({"thinking": {"type": "enabled", "budget_tokens": 1024}}), Some("low"))]
    #[case::enabled_high(json!({"thinking": {"type": "enabled", "budget_tokens": 32000}}), Some("high"))]
    #[case::adaptive_default(json!({"thinking": {"type": "adaptive"}}), Some("medium"))]
    #[case::adaptive_effort(
        json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}}),
        Some("xhigh")
    )]
    #[case::explicit_effort(
        json!({"thinking": {"type": "enabled", "budget_tokens": 1024}, "reasoning_effort": "high"}),
        Some("high")
    )]
    #[case::disabled(json!({"thinking": {"type": "disabled"}}), None)]
    #[case::absent(json!({}), None)]
    fn thinking_becomes_reasoning_effort(#[case] params: Value, #[case] expected: Option<&str>) {
        let mut request = json!({"model": "m", "messages": []});
        request
            .as_object_mut()
            .unwrap()
            .extend(params.as_object().unwrap().clone());
        let translated = translate(request);
        assert_eq!(
            translated
                .body
                .get("reasoning_effort")
                .and_then(Value::as_str),
            expected
        );
        assert!(!translated.body.contains_key("thinking"));
    }

    #[rstest]
    #[case::text_result(json!("done"), json!("done"))]
    #[case::no_content(Value::Null, json!(""))]
    #[case::rich_result(
        json!([{"type": "text", "text": "chart:"}, {"type": "image", "source": {"type": "url", "url": "https://x/c.png"}}]),
        json!([{"type": "text", "text": "chart:"}, {"type": "image_url", "image_url": {"url": "https://x/c.png"}}])
    )]
    fn tool_results_become_tool_messages(#[case] content: Value, #[case] expected: Value) {
        let translated = translate(
            json!({"model": "m", "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": content},
            ]}]}),
        );
        assert_eq!(
            translated.body["messages"],
            json!([{"role": "tool", "tool_call_id": "t1", "content": expected}])
        );
    }

    #[rstest]
    fn a_pdf_document_becomes_a_file_part() {
        let translated = translate(
            json!({"model": "m", "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBER"}},
            ]}]}),
        );
        assert_eq!(
            translated.body["messages"][0]["content"],
            json!([{"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBER"}}])
        );
    }

    fn respond(response: Value, tool_names: &ToolNames) -> Result<Value, Error> {
        let body = serde_json::to_vec(&response).unwrap();
        translate_openai_response_to_anthropic(&body, "fallback-model", tool_names)
            .map(|message| serde_json::to_value(message).unwrap())
    }

    #[rstest]
    fn a_completion_becomes_a_message() {
        let tool_names = translate(json!({"model": "m", "messages": [], "tools": [
            {"name": LONG_NAME, "input_schema": {"type": "object"}},
        ]}))
        .tool_names;
        let message = respond(
            json!({
                "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-6-astra",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant",
                    "reasoning_content": "the user wants a search",
                    "content": "Searching.",
                    "tool_calls": [{"id": "call:1", "type": "function", "function": {
                        "name": truncate_tool_name(LONG_NAME), "arguments": "{\"title\": \"cat\"}",
                    }}],
                }}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 60, "audio_tokens": 0}},
            }),
            &tool_names,
        )
        .unwrap();
        assert_eq!(
            message,
            json!({
                "id": "chatcmpl-1", "type": "message", "role": "assistant", "model": "gpt-6-astra",
                "content": [
                    {"type": "thinking", "thinking": "the user wants a search", "signature": ""},
                    {"type": "text", "text": "Searching."},
                    {"type": "tool_use", "id": "call_1", "name": LONG_NAME, "input": {"title": "cat"}},
                ],
                "stop_reason": "tool_use",
                "stop_sequence": null,
                "usage": {"input_tokens": 40, "output_tokens": 20, "cache_read_input_tokens": 60},
            })
        );
    }

    #[rstest]
    #[case::stop(json!({"content": "hi"}), "stop", "end_turn")]
    #[case::length(json!({"content": "hi"}), "length", "max_tokens")]
    #[case::content_filter(json!({"content": ""}), "content_filter", "refusal")]
    #[case::refusal(json!({"content": null, "refusal": "I can't help"}), "stop", "refusal")]
    #[case::cut_off_refusal(json!({"content": null, "refusal": "I can't"}), "length", "max_tokens")]
    fn finish_reasons_map_to_stop_reasons(
        #[case] message: Value,
        #[case] finish_reason: &str,
        #[case] expected: &str,
    ) {
        let message = respond(
            json!({"choices": [{"message": message, "finish_reason": finish_reason}]}),
            &ToolNames::default(),
        )
        .unwrap();
        assert_eq!(message["stop_reason"], expected);
        assert_eq!(message["model"], "fallback-model");
        assert_eq!(
            message["usage"],
            json!({"input_tokens": 0, "output_tokens": 0})
        );
    }

    #[rstest]
    #[case::invalid_arguments(json!({"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
        {"id": "c", "function": {"name": "f", "arguments": "{\"x\":"}},
    ]}}]}))]
    #[case::not_a_completion(json!([]))]
    fn malformed_completions_are_invalid_responses(#[case] response: Value) {
        assert!(matches!(
            respond(response, &ToolNames::default()),
            Err(Error::InvalidResponse(_))
        ));
    }
}
