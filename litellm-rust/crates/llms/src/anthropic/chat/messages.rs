//! Chat messages as the Messages API takes them: Python's `translate_system_message`,
//! `place_mid_conversation_system` and `anthropic_messages_pt`.

use std::collections::HashSet;

use base64::Engine;
use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    anthropic::common_utils::ENCRYPTED_REASONING_SIGNATURE_PREFIX,
    base_llm::chat::{
        media::{data_url, image_url},
        params::{parse_tool_arguments, truthy},
    },
};

pub const EMPTY_TEXT_PLACEHOLDER: &str =
    "[System: Empty message content sanitised to satisfy protocol]";
pub const CONVERTED_SYSTEM_NOTE: &str = "Operator note (not from the user): the following was originally a mid-conversation system-role reminder.";
const TOOL_USE_ID_FALLBACK: &str = "tool_use_id";

fn bad_request(message: impl Into<String>) -> Error {
    Error::InvalidRequest(ErrorDetail::Message(message.into()))
}

pub fn role(message: &Value) -> &str {
    match message.get("role").and_then(Value::as_str) {
        Some("developer") => "system",
        Some(role) => role,
        None => "",
    }
}

/// Python's truthiness of a message's `content`.
fn content_is_empty(message: &Value) -> bool {
    match message.get("content") {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.is_empty(),
        Some(Value::Array(items)) => items.is_empty(),
        Some(_) => false,
    }
}

/// `[^a-zA-Z0-9_-]` -> `_`, an empty id becoming `tool_use_id`.
pub fn sanitize_tool_use_id(id: &str) -> String {
    let sanitized: String = id
        .chars()
        .map(
            |c| match c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                true => c,
                false => '_',
            },
        )
        .collect();
    match sanitized.is_empty() {
        true => TOOL_USE_ID_FALLBACK.to_owned(),
        false => sanitized,
    }
}

/// `add_cache_control_to_content`: a dict `cache_control` on the source is copied over.
fn with_cache_control(mut block: Value, source: &Value) -> Value {
    if let (Some(cache_control @ Value::Object(_)), Value::Object(fields)) =
        (source.get("cache_control"), &mut block)
    {
        fields.insert("cache_control".into(), cache_control.clone());
    }
    block
}

/// `_cache_control` of the mid-conversation module: only an ephemeral one, rebuilt.
fn normalized_cache_control(source: &Value) -> Option<Value> {
    let cache_control = source.get("cache_control")?.as_object()?;
    if cache_control.get("type").and_then(Value::as_str) != Some("ephemeral") {
        return None;
    }
    Some(match cache_control.get("ttl").and_then(Value::as_str) {
        Some(ttl @ ("1h" | "5m")) => json!({"type": "ephemeral", "ttl": ttl}),
        _ => json!({"type": "ephemeral"}),
    })
}

/// The leading run of system messages, and the rest.
pub fn split_leading_system_run(messages: Vec<Value>) -> (Vec<Value>, Vec<Value>) {
    let leading = messages
        .iter()
        .position(|message| role(message) != "system")
        .unwrap_or(messages.len());
    let mut messages = messages;
    let rest = messages.split_off(leading);
    (messages, rest)
}

/// `translate_system_message` for the leading run.
pub fn system_blocks(run: &[Value]) -> Vec<Value> {
    let mut blocks = Vec::new();
    for message in run {
        match message.get("content") {
            Some(Value::String(text)) => {
                if text.is_empty() {
                    continue;
                }
                let mut block = Map::from_iter([
                    ("type".to_owned(), json!("text")),
                    ("text".to_owned(), json!(text)),
                ]);
                if let Some(cache_control) = message.get("cache_control") {
                    block.insert("cache_control".into(), cache_control.clone());
                }
                blocks.push(Value::Object(block));
            }
            Some(Value::Array(parts)) => {
                for part in parts {
                    let text = part.get("text").cloned().unwrap_or(Value::Null);
                    let kind = part.get("type").cloned().unwrap_or(Value::Null);
                    let empty = match &text {
                        Value::String(text) => text.is_empty(),
                        Value::Null => true,
                        _ => false,
                    };
                    if kind.as_str() == Some("text") && empty {
                        continue;
                    }
                    let mut block =
                        Map::from_iter([("type".to_owned(), kind), ("text".to_owned(), text)]);
                    if let Some(cache_control) = part.get("cache_control") {
                        block.insert("cache_control".into(), cache_control.clone());
                    }
                    blocks.push(Value::Object(block));
                }
            }
            _ => {}
        }
    }
    blocks
}

/// The non-empty text parts of a system message, each with its normalized `cache_control`.
fn system_text_parts(message: &Value) -> Vec<Value> {
    let block = |text: &str, cache_control: Option<Value>| {
        let mut block = Map::from_iter([
            ("type".to_owned(), json!("text")),
            ("text".to_owned(), json!(text)),
        ]);
        if let Some(cache_control) = cache_control {
            block.insert("cache_control".into(), cache_control);
        }
        Value::Object(block)
    };
    match message.get("content") {
        Some(Value::String(text)) if !text.is_empty() => {
            vec![block(text, normalized_cache_control(message))]
        }
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| {
                let text = part.get("text")?.as_str().filter(|text| !text.is_empty())?;
                Some(block(text, normalized_cache_control(part)))
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// `system_message_as_user` for each message of a run that carries text.
pub fn converted_user_turns(run: &[Value]) -> Vec<Value> {
    run.iter()
        .filter_map(|message| {
            let parts = system_text_parts(message);
            if parts.is_empty() {
                return None;
            }
            let mut content = vec![json!({"type": "text", "text": CONVERTED_SYSTEM_NOTE})];
            content.extend(parts);
            Some(json!({"role": "user", "content": content}))
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    System,
    Tool,
    User,
    Other,
}

fn kind(message: &Value) -> Kind {
    match role(message) {
        "system" => Kind::System,
        "tool" | "function" => Kind::Tool,
        "user" => Kind::User,
        _ => Kind::Other,
    }
}

/// `_converted_for_unflagged_model`: every later system message becomes a user turn in
/// place, a run caught before a tool result moving behind it.
pub fn place_mid_conversation_system(messages: Vec<Value>) -> Vec<Value> {
    if !messages.iter().any(|message| kind(message) == Kind::System) {
        return messages;
    }
    let mut runs: Vec<(Kind, Vec<Value>)> = Vec::new();
    for message in messages {
        let message_kind = kind(&message);
        match runs.last_mut() {
            Some((last, run)) if *last == message_kind => run.push(message),
            _ => runs.push((message_kind, vec![message])),
        }
    }
    let mut placed = Vec::new();
    for index in 0..runs.len() {
        let (run_kind, run) = &runs[index];
        let follower = runs.get(index + 1).map(|(kind, _)| *kind);
        match run_kind {
            Kind::System => {
                if follower != Some(Kind::Tool) {
                    placed.extend(converted_user_turns(run));
                }
            }
            Kind::Tool if index > 0 && runs[index - 1].0 == Kind::System => {
                placed.extend(run.iter().cloned());
                placed.extend(converted_user_turns(&runs[index - 1].1));
            }
            _ => placed.extend(run.iter().cloned()),
        }
    }
    placed
}

/// `_sanitize_empty_text_content`.
fn sanitize_empty_text(mut message: Value) -> Value {
    if !matches!(role(&message), "user" | "assistant") {
        return message;
    }
    match message.get_mut("content") {
        Some(Value::String(text)) if text.trim().is_empty() => {
            *text = EMPTY_TEXT_PLACEHOLDER.to_owned();
        }
        Some(Value::Array(parts)) => {
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("text") {
                    continue;
                }
                let blank = part
                    .get("text")
                    .and_then(Value::as_str)
                    .is_none_or(|text| text.trim().is_empty());
                if blank && let Value::Object(fields) = part {
                    fields.insert("text".into(), json!(EMPTY_TEXT_PLACEHOLDER));
                }
            }
        }
        _ => {}
    }
    message
}

/// `convert_to_anthropic_image_obj`: a data URL as its media type and payload.
fn image_source(url: &str, format: Option<&str>) -> Result<(String, String), Error> {
    let (media_type, data) = data_url(url).ok_or_else(|| {
        bad_request(format!(
            "Image url not in expected format. Example Expected input - \"image_url\": \"data:image/jpeg;base64,{{base64_image}}\". Supported formats - ['image/jpeg', 'image/png', 'image/gif', 'image/webp']. Error: {url:.40}"
        ))
    })?;
    let media_type = match format {
        Some(format) => format.to_owned(),
        None => media_type.replace("\\/", "/"),
    };
    Ok((media_type, data.to_owned()))
}

/// `create_anthropic_image_param`.
fn image_block(part: &Value) -> Result<Value, Error> {
    let url = image_url(part).unwrap_or_default();
    let format = part
        .get("image_url")
        .and_then(|image| image.get("format"))
        .and_then(Value::as_str);
    if url.starts_with("https://") {
        return Ok(json!({"type": "image", "source": {"type": "url", "url": url}}));
    }
    if url.starts_with("http") {
        return Err(Error::Unsupported(
            "an http:// image the host did not download",
        ));
    }
    let (media_type, data) = image_source(url, format)?;
    Ok(
        json!({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}),
    )
}

/// `anthropic_process_openai_file_message`.
fn file_block(part: &Value) -> Result<Value, Error> {
    let file = part.get("file").ok_or_else(|| {
        bad_request("Content block has type='file' but is missing the required 'file' field")
    })?;
    let format = file.get("format").and_then(Value::as_str);
    if let Some(data) = file
        .get("file_data")
        .and_then(Value::as_str)
        .filter(|data| !data.is_empty())
    {
        if data.starts_with("http") {
            return Err(Error::Unsupported(
                "a file_data URL the host did not download",
            ));
        }
        let (media_type, data) = image_source(data, format)?;
        return Ok(
            json!({"type": "document", "source": {"type": "base64", "media_type": media_type, "data": data}}),
        );
    }
    match file
        .get("file_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    {
        Some(id) if format.is_none() && id.starts_with("http") => {
            Ok(json!({"type": "document", "source": {"type": "url", "url": id}}))
        }
        // Uploaded files need the Files API betas and, for a container upload, the code
        // execution tool.
        Some(_) => Err(Error::Unsupported("an Anthropic Files API file_id")),
        None => Err(bad_request(format!(
            "Either file_data or file_id must be present in the file message: {part}"
        ))),
    }
}

/// `_is_anthropic_document_data_uri`.
fn is_document_data_uri(url: &str) -> bool {
    url.strip_prefix("data:")
        .map(|rest| rest.split([';', ',']).next().unwrap_or_default())
        .is_some_and(|media_type| matches!(media_type, "application/pdf" | "text/plain"))
}

/// `convert_to_anthropic_tool_result`.
fn tool_result(message: &Value) -> Result<Value, Error> {
    let content = match message.get("content") {
        Some(Value::String(text)) => json!(text),
        Some(Value::Array(parts)) => {
            let mut blocks = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let mut block = Map::from_iter([
                            ("type".to_owned(), json!("text")),
                            (
                                "text".to_owned(),
                                part.get("text").cloned().unwrap_or(Value::Null),
                            ),
                        ]);
                        if let Some(cache_control) =
                            part.get("cache_control").filter(|value| !value.is_null())
                        {
                            block.insert("cache_control".into(), cache_control.clone());
                        }
                        blocks.push(Value::Object(block));
                    }
                    Some("image_url") => {
                        let url = image_url(part).unwrap_or_default();
                        let block = match is_document_data_uri(url) {
                            true => {
                                file_block(&json!({"type": "file", "file": {"file_data": url}}))?
                            }
                            false => image_block(part)?,
                        };
                        blocks.push(with_cache_control(block, part));
                    }
                    Some("tool_reference") => blocks.push(json!({
                        "type": "tool_reference",
                        "tool_name": part.get("tool_name").cloned().unwrap_or(Value::Null),
                    })),
                    Some("file") => blocks.push(with_cache_control(file_block(part)?, part)),
                    _ => {}
                }
            }
            Value::Array(blocks)
        }
        _ => json!(""),
    };
    let id = match role(message) {
        "tool" => message
            .get("tool_call_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| bad_request("A tool message needs a tool_call_id."))?,
        _ => message
            .get("tool_call_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
    };
    let mut result = Map::from_iter([
        ("type".to_owned(), json!("tool_result")),
        ("tool_use_id".to_owned(), json!(sanitize_tool_use_id(&id))),
        ("content".to_owned(), content),
    ]);
    if let Some(cache_control) = message
        .get("cache_control")
        .filter(|value| !value.is_null())
    {
        result.insert("cache_control".into(), cache_control.clone());
    }
    Ok(Value::Object(result))
}

pub fn is_encrypted_reasoning(block: &Value) -> bool {
    let field = match block.get("type").and_then(Value::as_str) {
        Some("thinking") => block.get("signature"),
        Some("redacted_thinking") => block.get("data"),
        _ => None,
    };
    field
        .and_then(Value::as_str)
        .is_some_and(|value| value.starts_with(ENCRYPTED_REASONING_SIGNATURE_PREFIX))
}

/// `is_unsignable_thinking_block`.
pub fn is_unsignable_thinking_block(block: &Value) -> bool {
    if is_encrypted_reasoning(block) {
        return true;
    }
    block.get("type").and_then(Value::as_str) == Some("thinking")
        && block
            .get("signature")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
}

/// `native_redacted_thinking_data`: the base64 wrapper Converse adds, removed.
pub fn native_redacted_thinking_data(data: &str) -> String {
    let engine = base64::engine::general_purpose::STANDARD;
    engine
        .decode(data)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .filter(|payload| payload.is_ascii() && engine.decode(payload).is_ok())
        .unwrap_or_else(|| data.to_owned())
}

/// `_replayable_thinking_blocks`.
fn replayable_thinking_blocks(blocks: &[Value]) -> Vec<Value> {
    blocks
        .iter()
        .filter(|block| !is_unsignable_thinking_block(block))
        .map(|block| {
            let mut block = block.clone();
            if let Value::Object(fields) = &mut block {
                if fields.get("type").and_then(Value::as_str) == Some("redacted_thinking")
                    && let Some(Value::String(data)) = fields.get("data")
                {
                    let native = native_redacted_thinking_data(data);
                    fields.insert("data".into(), json!(native));
                }
                fields.remove("index");
            }
            block
        })
        .collect()
}

/// The tool calls of an assistant message, as `tool_use` blocks.
fn tool_uses(tool_calls: &[Value], seen: &mut HashSet<String>) -> Result<Vec<Value>, Error> {
    let mut blocks = Vec::new();
    for call in tool_calls {
        if call.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
        if id.starts_with("srvtoolu_") {
            return Err(Error::Unsupported(
                "replaying an Anthropic server tool call",
            ));
        }
        let function = call.get("function");
        let name = function
            .and_then(|function| function.get("name"))
            .cloned()
            .unwrap_or(Value::Null);
        let input = parse_tool_arguments(function.and_then(|function| function.get("arguments")))?;
        let id = sanitize_tool_use_id(id);
        let block = with_cache_control(
            json!({"type": "tool_use", "id": id, "name": name, "input": input}),
            call,
        );
        if !seen.insert(id) {
            continue;
        }
        blocks.push(block);
    }
    Ok(blocks)
}

/// `anthropic_messages_pt` over messages whose later system messages are already placed.
pub fn anthropic_messages_pt(messages: Vec<Value>) -> Result<Vec<Value>, Error> {
    let messages: Vec<Value> = messages.into_iter().map(sanitize_empty_text).collect();
    if messages.is_empty() {
        return Err(bad_request(
            "Anthropic requires at least one non-system message. Either provide one, or set `litellm.modify_params = True` // `litellm_settings::modify_params: True` to add the dummy user message - {'role': 'user', 'content': 'Please continue.'}.",
        ));
    }
    let user_types = |message: &Value| matches!(role(message), "user" | "tool" | "function");
    // `_tool_result_later_in_turn`: whether a client tool result follows in the same turn.
    let mut tool_result_later = vec![false; messages.len()];
    let mut later = false;
    for index in (0..messages.len()).rev() {
        tool_result_later[index] = later;
        let message = &messages[index];
        later = user_types(message) && (later || matches!(role(message), "tool" | "function"));
    }

    let mut out: Vec<Value> = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        let start = index;
        let mut user_content: Vec<Value> = Vec::new();
        while index < messages.len() && user_types(&messages[index]) {
            let message = &messages[index];
            if role(message) == "user" {
                match message.get("content") {
                    _ if content_is_empty(message) => {
                        if !tool_result_later[index]
                            && message.get("cache_control").is_some_and(truthy)
                        {
                            user_content.push(with_cache_control(
                                json!({"type": "text", "text": EMPTY_TEXT_PLACEHOLDER}),
                                message,
                            ));
                        }
                    }
                    Some(Value::Array(parts)) => {
                        for part in parts {
                            match part.get("type").and_then(Value::as_str) {
                                Some("image_url") => {
                                    user_content.push(with_cache_control(image_block(part)?, part))
                                }
                                Some("text") => user_content.push(with_cache_control(
                                    json!({"type": "text", "text": part.get("text").cloned().unwrap_or(Value::Null)}),
                                    part,
                                )),
                                Some("document") => {
                                    user_content.push(with_cache_control(part.clone(), part))
                                }
                                Some("file") => {
                                    user_content.push(with_cache_control(file_block(part)?, part))
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Value::String(text)) => user_content.push(with_cache_control(
                        json!({"type": "text", "text": text}),
                        message,
                    )),
                    _ => {}
                }
            } else {
                user_content.push(tool_result(message)?);
            }
            index += 1;
        }
        if !user_content.is_empty() {
            out.push(json!({"role": "user", "content": user_content}));
        } else if let Some(empty) = messages[start..index]
            .iter()
            .rev()
            .find(|message| role(message) == "user" && content_is_empty(message))
        {
            out.push(json!({"role": "user", "content": [with_cache_control(
                json!({"type": "text", "text": EMPTY_TEXT_PLACEHOLDER}),
                empty,
            )]}));
        }

        let mut seen_tool_ids: HashSet<String> = HashSet::new();
        let mut assistant_content: Vec<Value> = Vec::new();
        while index < messages.len() && role(&messages[index]) == "assistant" {
            let message = &messages[index];
            let provider_fields = message.get("provider_specific_fields");
            if let Some(Value::Array(blocks)) =
                provider_fields.and_then(|fields| fields.get("compaction_blocks"))
            {
                assistant_content.extend(blocks.iter().cloned());
            }
            if provider_fields.is_some_and(|fields| {
                fields
                    .get("web_search_results")
                    .is_some_and(|v| !v.is_null())
                    || fields.get("tool_results").is_some_and(|v| !v.is_null())
            }) && message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| !calls.is_empty())
            {
                return Err(Error::Unsupported(
                    "replaying Anthropic server tool results",
                ));
            }
            let thinking_blocks = match message.get("thinking_blocks") {
                Some(Value::Array(blocks)) => Some(replayable_thinking_blocks(blocks)),
                _ => None,
            };
            let parts = message.get("content").and_then(Value::as_array);
            let inline_thinking = parts.is_some_and(|parts| {
                parts.iter().any(|part| {
                    matches!(
                        part.get("type").and_then(Value::as_str),
                        Some("thinking" | "redacted_thinking")
                    )
                })
            });
            if let Some(blocks) = thinking_blocks
                && !inline_thinking
            {
                assistant_content.extend(blocks);
            }
            match message.get("content") {
                Some(Value::Array(parts)) => {
                    for part in parts.iter().filter(|part| part.is_object()) {
                        let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
                        let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                        if kind == "thinking" && !is_unsignable_thinking_block(part) {
                            assistant_content.push(part.clone());
                        } else if kind == "text" && !text.is_empty() {
                            assistant_content.push(with_cache_control(
                                json!({"type": "text", "text": text}),
                                part,
                            ));
                        } else if kind == "server_tool_use" || kind.ends_with("_tool_result") {
                            assistant_content.push(part.clone());
                        }
                    }
                }
                Some(Value::String(text)) if !text.is_empty() => {
                    assistant_content.push(with_cache_control(
                        json!({"type": "text", "text": text}),
                        message,
                    ));
                }
                _ => {}
            }
            if let Some(Value::Array(calls)) = message.get("tool_calls") {
                assistant_content.extend(tool_uses(calls, &mut seen_tool_ids)?);
            }
            if let Some(call) = message.get("function_call").filter(|call| !call.is_null()) {
                let input = parse_tool_arguments(call.get("arguments"))?;
                assistant_content.push(json!({
                    "type": "tool_use",
                    "id": uuid::Uuid::new_v4().to_string(),
                    "name": call.get("name").filter(|name| !name.is_null()).cloned().unwrap_or_else(|| json!("")),
                    "input": input,
                }));
            }
            index += 1;
        }
        if !assistant_content.is_empty() {
            out.push(json!({"role": "assistant", "content": assistant_content}));
        }
        if index == start {
            return Err(bad_request(format!(
                "Invalid Message passed in {}",
                messages[index]
            )));
        }
    }
    if let Some(last) = out.last_mut()
        && last["role"] == "assistant"
        && let Some(Value::Array(blocks)) = last.get_mut("content")
    {
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("text")
                && let Some(Value::String(text)) = block.get_mut("text")
            {
                *text = text.trim_end().to_owned();
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn list(value: Value) -> Vec<Value> {
        value.as_array().unwrap().clone()
    }

    #[rstest]
    fn tool_round_trip_merges_turns() {
        let messages = anthropic_messages_pt(list(json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "call.1", "type": "function", "function": {"name": "get", "arguments": "{\"c\": \"P\"}"}}]},
            {"role": "tool", "tool_call_id": "call.1", "content": "sunny"},
            {"role": "user", "content": [{"type": "text", "text": "thanks", "cache_control": {"type": "ephemeral"}}]}
        ])))
        .unwrap();
        assert_eq!(
            messages,
            list(json!([
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "get", "input": {"c": "P"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"},
                    {"type": "text", "text": "thanks", "cache_control": {"type": "ephemeral"}}
                ]}
            ]))
        );
    }

    #[rstest]
    fn images_empty_text_and_final_assistant_whitespace() {
        let messages = anthropic_messages_pt(list(json!([
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}},
                {"type": "image_url", "image_url": "https://x/y.png"},
                {"type": "text", "text": " "}
            ]},
            {"role": "assistant", "content": "Sure  "}
        ])))
        .unwrap();
        assert_eq!(
            messages[0]["content"],
            json!([
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBO"}},
                {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
                {"type": "text", "text": EMPTY_TEXT_PLACEHOLDER}
            ])
        );
        assert_eq!(
            messages[1]["content"],
            json!([{"type": "text", "text": "Sure"}])
        );
    }

    #[rstest]
    fn later_system_messages_become_operator_notes() {
        let placed = place_mid_conversation_system(list(json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "system", "content": "remember", "cache_control": {"type": "ephemeral", "ttl": "1h", "x": 1}},
            {"role": "tool", "tool_call_id": "t", "content": "r"}
        ])));
        assert_eq!(placed[2]["role"], "tool");
        assert_eq!(
            placed[3],
            json!({"role": "user", "content": [
                {"type": "text", "text": CONVERTED_SYSTEM_NOTE},
                {"type": "text", "text": "remember", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ]})
        );
    }

    #[rstest]
    fn unsigned_thinking_is_dropped_and_signed_thinking_leads() {
        let messages = anthropic_messages_pt(list(json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "thinking_blocks": [
                {"type": "thinking", "thinking": "t", "signature": "s", "index": 0},
                {"type": "thinking", "thinking": "u"}
            ]}
        ])))
        .unwrap();
        assert_eq!(
            messages[1]["content"],
            json!([{"type": "thinking", "thinking": "t", "signature": "s"}, {"type": "text", "text": "a"}])
        );
    }
}
