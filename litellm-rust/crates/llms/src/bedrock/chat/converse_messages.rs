//! Chat messages and tools as Converse takes them: Python's `_transform_system_message`,
//! `_neutralize_orphaned_tool_blocks`, `_bedrock_converse_messages_pt` and `_bedrock_tools_pt`.

use std::collections::HashSet;

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::{
    Error, ErrorDetail,
    anthropic::chat::messages::{CONVERTED_SYSTEM_NOTE, native_redacted_thinking_data},
    base_llm::chat::{params::truthy, response_format::unpack_defs},
    bedrock::chat::converse_params::{BedrockModel, make_valid_bedrock_tool_name},
};

const SUPPORTED_IMAGE_FORMATS: [&str; 4] = ["png", "jpeg", "gif", "webp"];
const SUPPORTED_DOCUMENT_FORMATS: [&str; 9] = [
    "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md",
];
const SUPPORTED_VIDEO_FORMATS: [&str; 9] = [
    "mp4", "mov", "mkv", "webm", "flv", "mpeg", "mpg", "wmv", "3gp",
];
const DOCUMENT_PLACEHOLDER_TEXT: &str = ".";
const TOOL_RESULT_IMAGE_PLACEHOLDER: &str =
    "[Tool returned an image - see the following user message]";
const TOOL_USE_ID_MAX_LEN: usize = 64;
const TOOL_USE_ID_HASH_LEN: usize = 8;
const HASH_SAMPLE_BYTES: usize = 64 * 1024;
const BAD_MESSAGE: &str = "Invalid Message ";

fn bad_request(message: impl Into<String>) -> Error {
    Error::InvalidRequest(ErrorDetail::Message(message.into()))
}

fn role(message: &Value) -> &str {
    message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

fn is_system(message: &Value) -> bool {
    role(message) == "system"
}

/// `CachePointBlock` for a `cache_control`, or nothing for a model that refuses them.
pub fn cache_point(control: Option<&Value>, model: &BedrockModel<'_>) -> Option<Value> {
    control.filter(|control| !control.is_null())?;
    if !model.accepts_cache_points() {
        return None;
    }
    Some(json!({"cachePoint": cache_point_block(control, model)}))
}

fn cache_point_block(control: Option<&Value>, model: &BedrockModel<'_>) -> Value {
    let mut block = Map::from_iter([("type".to_owned(), json!("default"))]);
    if let Some(ttl) = control
        .and_then(|control| control.get("ttl"))
        .and_then(Value::as_str)
        .filter(|ttl| matches!(*ttl, "5m" | "1h"))
        && model.cache_ttl_supported()
    {
        block.insert("ttl".into(), json!(ttl));
    }
    Value::Object(block)
}

/// The leading run of system messages as Converse `system` blocks.
fn system_blocks(leading: &[Value], model: &BedrockModel<'_>) -> Vec<Value> {
    let mut blocks = Vec::new();
    for message in leading {
        match message.get("content") {
            Some(Value::String(text)) if !text.is_empty() => {
                blocks.push(json!({"text": text}));
                blocks.extend(cache_point(message.get("cache_control"), model));
            }
            Some(Value::Array(parts)) => {
                for part in parts {
                    let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                    if part.get("type").and_then(Value::as_str) == Some("text") && !text.is_empty()
                    {
                        blocks.push(json!({"text": text}));
                        blocks.extend(cache_point(part.get("cache_control"), model));
                    }
                }
            }
            _ => {}
        }
    }
    blocks
}

fn assistant_has_tool_calls(message: &Value) -> bool {
    role(message) == "assistant" && message.get("tool_calls").is_some_and(truthy)
}

/// `_opens_with_tool_result`.
fn opens_with_tool_result(message: &Value) -> bool {
    match role(message) {
        "tool" => true,
        "user" => {
            let first = match message.get("content") {
                Some(Value::Array(parts)) => parts.first(),
                _ => None,
            };
            first
                .and_then(|part| part.get("type"))
                .and_then(Value::as_str)
                == Some("tool_result")
        }
        _ => false,
    }
}

/// `_reordered_around_tool_results`.
fn reordered_around_tool_results(messages: &[Value], index: usize) -> Vec<Value> {
    let message = &messages[index];
    let system_run_start = |end: usize| {
        (0..end)
            .rev()
            .find(|j| !is_system(&messages[*j]))
            .map_or(0, |j| j + 1)
    };
    if opens_with_tool_result(message) {
        if index + 1 < messages.len() && opens_with_tool_result(&messages[index + 1]) {
            return vec![message.clone()];
        }
        let tool_run_start = (0..=index)
            .rev()
            .find(|j| !opens_with_tool_result(&messages[*j]))
            .map_or(0, |j| j + 1);
        let run_start = system_run_start(tool_run_start);
        let run = &messages[run_start..tool_run_start];
        if !run.is_empty() && run_start >= 1 && assistant_has_tool_calls(&messages[run_start - 1]) {
            let mut kept = vec![message.clone()];
            kept.extend(run.iter().cloned());
            return kept;
        }
        return vec![message.clone()];
    }
    if !is_system(message) {
        return vec![message.clone()];
    }
    let run_start = system_run_start(index);
    let run_end = (index..messages.len())
        .find(|j| !is_system(&messages[*j]))
        .unwrap_or(messages.len());
    if run_end < messages.len()
        && opens_with_tool_result(&messages[run_end])
        && run_start > 0
        && assistant_has_tool_calls(&messages[run_start - 1])
    {
        return Vec::new();
    }
    vec![message.clone()]
}

/// `_system_role_message_as_user`: a mid-conversation system message as a user turn.
fn system_as_user(message: &Value) -> Option<Value> {
    let block = |text: &str, cache_control: Option<&Value>| {
        let mut block = Map::from_iter([
            ("type".to_owned(), json!("text")),
            ("text".to_owned(), json!(text)),
        ]);
        if let Some(cache_control) = cache_control.filter(|c| !c.is_null()) {
            block.insert("cache_control".into(), cache_control.clone());
        }
        Value::Object(block)
    };
    let texts: Vec<Value> = match message.get("content") {
        Some(Value::String(text)) if !text.is_empty() => {
            vec![block(text, message.get("cache_control"))]
        }
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| {
                let text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())?;
                Some(block(text, part.get("cache_control")))
            })
            .collect(),
        _ => Vec::new(),
    };
    if texts.is_empty() {
        return None;
    }
    let mut content = vec![json!({"type": "text", "text": CONVERTED_SYSTEM_NOTE})];
    content.extend(texts);
    Some(json!({"role": "user", "content": content}))
}

/// `_transform_system_message`: the conversation after the leading system run, and that run
/// as `system` blocks.
pub fn transform_system_message(
    messages: Vec<Value>,
    model: &BedrockModel<'_>,
) -> (Vec<Value>, Vec<Value>) {
    let leading = messages
        .iter()
        .position(|message| !is_system(message))
        .unwrap_or(messages.len());
    let system = system_blocks(&messages[..leading], model);
    let remaining = &messages[leading..];
    let kept = (0..remaining.len())
        .flat_map(|index| reordered_around_tool_results(remaining, index))
        .filter_map(|message| match is_system(&message) {
            true => system_as_user(&message),
            false => Some(message),
        })
        .collect();
    (kept, system)
}

/// `convert_content_list_to_str`.
fn content_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

/// `_neutralize_orphaned_tool_blocks`: tool calls and results without `tools=` as text.
pub fn neutralize_orphaned_tool_blocks(
    messages: Vec<Value>,
    has_tools: bool,
    guardrail_active: bool,
) -> Vec<Value> {
    let orphaned = messages.iter().any(|message| {
        assistant_has_tool_calls(message) || matches!(role(message), "tool" | "function")
    });
    if has_tools || !orphaned {
        return messages;
    }
    messages
        .into_iter()
        .map(|message| {
            if assistant_has_tool_calls(&message) {
                let base = content_text(&message);
                let calls = message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|call| {
                        let function = call.get("function");
                        let name = function
                            .and_then(|f| f.get("name"))
                            .and_then(Value::as_str)
                            .filter(|name| !name.is_empty())
                            .unwrap_or("unknown_tool");
                        let arguments = function
                            .and_then(|f| f.get("arguments"))
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let label = match call
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty())
                        {
                            Some(id) => format!("tool call {id}"),
                            None => "tool call".to_owned(),
                        };
                        format!("[{label}: {name}({arguments})]")
                    });
                let text: Vec<String> = std::iter::once(base)
                    .chain(calls)
                    .filter(|part| !part.is_empty())
                    .collect();
                return json!({"role": "assistant", "content": text.join("\n")});
            }
            if matches!(role(&message), "tool" | "function") {
                let target = ["tool_call_id", "name"]
                    .iter()
                    .find_map(|key| {
                        message
                            .get(*key)
                            .and_then(Value::as_str)
                            .filter(|v| !v.is_empty())
                    })
                    .unwrap_or("unknown");
                let rendered = content_text(&message).trim().to_owned();
                let rendered = match rendered.is_empty() {
                    true => "<non-text tool result omitted>".to_owned(),
                    false => rendered,
                };
                let text = format!("[tool result for {target}: {rendered}]");
                return match guardrail_active {
                    true => {
                        json!({"role": "user", "content": [{"type": "guarded_text", "text": text}]})
                    }
                    false => json!({"role": "user", "content": text}),
                };
            }
            message
        })
        .collect()
}

/// `_convert_consecutive_user_messages_to_guarded_text`.
pub fn guard_trailing_user_messages(mut messages: Vec<Value>) -> Vec<Value> {
    for message in messages.iter_mut().rev() {
        if role(message) != "user" {
            break;
        }
        match message.get_mut("content") {
            Some(Value::Array(parts)) => {
                if parts
                    .iter()
                    .any(|part| part.get("type").and_then(Value::as_str) == Some("guarded_text"))
                {
                    continue;
                }
                for part in parts.iter_mut() {
                    if part.get("type").and_then(Value::as_str) == Some("text") {
                        let text = part.get("text").cloned().unwrap_or(Value::Null);
                        *part = json!({"type": "guarded_text", "text": text});
                    }
                }
            }
            Some(Value::String(text)) => {
                let text = text.clone();
                message["content"] = json!([{"type": "guarded_text", "text": text}]);
            }
            _ => {}
        }
    }
    messages
}

/// `_sanitize_bedrock_tool_use_id`.
pub fn sanitize_tool_use_id(id: &str) -> String {
    let sanitized: String = id
        .chars()
        .map(
            |c| match c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-') {
                true => c,
                false => '_',
            },
        )
        .collect();
    if sanitized == id && sanitized.len() <= TOOL_USE_ID_MAX_LEN {
        return sanitized;
    }
    let digest = hex(&Sha256::digest(id.as_bytes()));
    let head: String = sanitized
        .chars()
        .take(TOOL_USE_ID_MAX_LEN - TOOL_USE_ID_HASH_LEN - 1)
        .collect();
    format!("{head}_{}", &digest[..TOOL_USE_ID_HASH_LEN])
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// `mimetypes.guess_all_extensions` for the document types Converse takes.
fn document_format(mime_type: &str) -> Result<String, Error> {
    let format = match mime_type {
        "application/pdf" => "pdf",
        "text/csv" => "csv",
        "application/msword" => "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => "docx",
        "application/vnd.ms-excel" => "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => "xlsx",
        "text/html" => "html",
        "text/plain" => "txt",
        "text/markdown" | "text/x-markdown" => "md",
        _ => {
            return Err(bad_request(format!(
                "No supported extensions for MIME type: {mime_type}. Supported formats: {}",
                python_list(&SUPPORTED_DOCUMENT_FORMATS)
            )));
        }
    };
    Ok(format.to_owned())
}

fn python_list(items: &[&str]) -> String {
    format!(
        "[{}]",
        items
            .iter()
            .map(|item| format!("'{item}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn is_document_mime(mime_type: &str) -> bool {
    mime_type.starts_with("application") || mime_type.starts_with("text")
}

/// `BedrockImageProcessor.validate_format`.
fn validate_format(mime_type: &str, image_format: &str) -> Result<String, Error> {
    if is_document_mime(mime_type) {
        return document_format(mime_type);
    }
    let supported: Vec<&str> = SUPPORTED_VIDEO_FORMATS
        .iter()
        .chain(SUPPORTED_IMAGE_FORMATS.iter())
        .copied()
        .collect();
    if !supported.contains(&image_format) {
        return Err(bad_request(format!(
            "Unsupported image format: {image_format}. Supported formats: {}",
            python_list(&supported)
        )));
    }
    Ok(image_format.to_owned())
}

/// The name Python derives from a document's content: a hash of its first 64 KB and length.
fn document_hash(data: &str) -> String {
    let normalized: String = data.split_whitespace().collect();
    let bytes = normalized.as_bytes();
    let mut hasher = Sha256::new();
    hasher.update(&bytes[..bytes.len().min(HASH_SAMPLE_BYTES)]);
    hasher.update(bytes.len().to_string().as_bytes());
    hex(&hasher.finalize())[..16].to_owned()
}

/// `BedrockImageProcessor.process_image_sync` over a URL the host already inlined.
fn media_block(url: &str, format: Option<&str>) -> Result<Value, Error> {
    if !url.contains("base64") {
        return Err(match url.contains("http://") || url.contains("https://") {
            true => Error::Unsupported("a media URL the host did not download"),
            false => bad_request(
                "Unsupported image type. Expected either image url or base64 encoded string",
            ),
        });
    }
    let (metadata, data) = url.split_once(',').unwrap_or((url, ""));
    let (mut mime_type, mut image_format) = match metadata
        .strip_prefix("data:")
        .and_then(|rest| rest.split(";base64").next())
    {
        Some(mime) => {
            let mime = mime.split(';').next().unwrap_or(mime).to_owned();
            let format = mime.split('/').nth(1).unwrap_or_default().to_owned();
            (mime, format)
        }
        None => ("image/jpeg".to_owned(), "jpeg".to_owned()),
    };
    if let Some(format) = format.filter(|format| !format.is_empty()) {
        mime_type = format.to_owned();
        image_format = mime_type.split('/').nth(1).unwrap_or_default().to_owned();
    }
    let image_format = validate_format(&mime_type, &image_format)?;
    let source = json!({"bytes": data});
    if is_document_mime(&mime_type) {
        let name = format!("DocumentPDFmessages_{}_{image_format}", document_hash(data));
        return Ok(json!({"document": {"source": source, "format": image_format, "name": name}}));
    }
    if SUPPORTED_VIDEO_FORMATS
        .iter()
        .any(|video| image_format.starts_with(video))
    {
        return Ok(json!({"video": {"source": source, "format": image_format}}));
    }
    Ok(json!({"image": {"source": source, "format": image_format}}))
}

fn image_url_block(part: &Value) -> Result<Value, Error> {
    let image_url = part.get("image_url");
    let (url, format) = match image_url {
        Some(Value::Object(image)) => (
            image.get("url").and_then(Value::as_str).unwrap_or_default(),
            image.get("format").and_then(Value::as_str),
        ),
        Some(Value::String(url)) => (url.as_str(), None),
        _ => ("", None),
    };
    media_block(url, format)
}

/// `process_file_message`.
fn file_block(part: &Value) -> Result<Value, Error> {
    let Some(file) = part.get("file").filter(|file| !file.is_null()) else {
        return Err(bad_request(
            "Content block has type='file' but is missing the required 'file' field",
        ));
    };
    let source = ["file_id", "file_data"].iter().find_map(|key| {
        file.get(*key)
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
    });
    let Some(source) = source else {
        return Err(bad_request(format!(
            "file_data and file_id cannot both be None. Got={part}"
        )));
    };
    media_block(source, file.get("format").and_then(Value::as_str))
}

/// `_process_document_message`.
fn document_block(part: &Value) -> Result<Value, Error> {
    let source = part.get("source").cloned().unwrap_or(Value::Null);
    let source_type = source
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if source_type != "base64" {
        return Err(bad_request(format!(
            "Bedrock Converse only supports base64-encoded document sources, got '{source_type}'. Please convert the document to base64 before sending to Bedrock."
        )));
    }
    let media_type = source
        .get("media_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let data = source
        .get("data")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let format = validate_format(media_type, media_type.split('/').nth(1).unwrap_or_default())?;
    let name = format!("Document_{}_{format}", document_hash(data));
    Ok(json!({"document": {"source": {"bytes": data}, "format": format, "name": name}}))
}

/// `skip_empty_text_blocks` for a list: whitespace-only text parts go.
fn non_empty_parts(parts: &[Value]) -> Vec<Value> {
    parts
        .iter()
        .filter(|part| {
            !(part.get("type").and_then(Value::as_str) == Some("text")
                && part
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .is_empty())
        })
        .cloned()
        .collect()
}

fn user_blocks(message: &Value, model: &BedrockModel<'_>) -> Result<Vec<Value>, Error> {
    let mut blocks = Vec::new();
    match message.get("content") {
        Some(Value::Array(parts)) => {
            for part in non_empty_parts(parts) {
                if !part.is_object() {
                    continue;
                }
                let text = || part.get("text").cloned().unwrap_or(Value::Null);
                match part.get("type").and_then(Value::as_str) {
                    Some("text" | "grounding_source" | "query") => {
                        blocks.push(json!({"text": text()}))
                    }
                    Some("guarded_text") => {
                        blocks.push(json!({"guardContent": {"text": {"text": text()}}}))
                    }
                    Some("image_url") => blocks.push(image_url_block(&part)?),
                    Some("file") => blocks.push(file_block(&part)?),
                    Some("document") => blocks.push(document_block(&part)?),
                    _ => {}
                }
                blocks.extend(cache_point(part.get("cache_control"), model));
            }
        }
        Some(Value::String(text)) if !text.is_empty() => {
            blocks.push(json!({"text": text}));
            blocks.extend(cache_point(message.get("cache_control"), model));
        }
        _ => {}
    }
    Ok(blocks)
}

/// `_parse_bedrock_tool_result_content_list` item by item.
fn tool_result_parts(content: &[Value]) -> Result<Vec<Value>, Error> {
    let mut parts = Vec::new();
    for item in content {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => {
                parts.push(json!({"text": item.get("text").cloned().unwrap_or(Value::Null)}))
            }
            Some(kind @ ("image_url" | "file")) => {
                let block = match kind {
                    "image_url" => image_url_block(item)?,
                    _ => file_block(item)?,
                };
                for key in ["image", "document"] {
                    if let Some(media) = block.get(key) {
                        parts.push(json!({key: media}));
                    }
                }
            }
            Some("tool_reference") => {
                if let Some(name) = item.get("tool_name").and_then(Value::as_str) {
                    parts.push(json!({"text": name}));
                }
            }
            _ => {}
        }
    }
    Ok(parts)
}

/// `_convert_to_bedrock_tool_call_result`.
fn tool_result(message: &Value) -> Result<Value, Error> {
    let search_results: Vec<Value> = message
        .get("search_results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|result| result.is_object())
        .map(|result| json!({"searchResult": result}))
        .collect();
    let used_search_results = !search_results.is_empty();
    let content = match used_search_results {
        true => search_results,
        false => match message.get("content") {
            Some(Value::String(text)) => vec![json!({"text": text})],
            Some(Value::Array(items)) => tool_result_parts(items)?,
            _ => Vec::new(),
        },
    };
    let id = match message.get("tool_call_id") {
        Some(Value::String(id)) => id.clone(),
        Some(Value::Null) => "None".to_owned(),
        Some(other) => other.to_string(),
        None => uuid::Uuid::new_v4().to_string(),
    };
    let mut result = Map::from_iter([
        ("content".to_owned(), Value::Array(content)),
        ("toolUseId".to_owned(), json!(sanitize_tool_use_id(&id))),
    ]);
    if used_search_results {
        result.insert("status".into(), json!("success"));
    }
    Ok(json!({"toolResult": result}))
}

/// `split_concatenated_json_objects`.
fn concatenated_objects(raw: &str) -> Vec<Value> {
    serde_json::Deserializer::from_str(raw.trim())
        .into_iter::<Value>()
        .map_while(Result::ok)
        .map(|value| match value {
            Value::Object(_) => value,
            _ => json!({}),
        })
        .collect()
}

/// `_convert_to_bedrock_tool_call_invoke`.
fn tool_uses(calls: &[Value], model: &BedrockModel<'_>) -> Result<Vec<Value>, Error> {
    let mut blocks = Vec::new();
    for call in calls.iter().filter(|call| call.get("function").is_some()) {
        let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
        let function = &call["function"];
        let name = make_valid_bedrock_tool_name(
            function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        let arguments = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let cache = cache_point(call.get("cache_control"), model);
        if arguments.trim().is_empty() {
            blocks.push(json!({"toolUse": {"input": {}, "name": name, "toolUseId": sanitize_tool_use_id(id)}}));
        } else {
            match serde_json::from_str::<Value>(arguments) {
                Ok(parsed) => {
                    let input = match parsed {
                        Value::Object(_) => parsed,
                        _ => json!({}),
                    };
                    blocks.push(json!({"toolUse": {"input": input, "name": name, "toolUseId": sanitize_tool_use_id(id)}}));
                }
                Err(_) => {
                    let objects = concatenated_objects(arguments);
                    if objects.is_empty() {
                        blocks.push(json!({"toolUse": {"input": {}, "name": name, "toolUseId": sanitize_tool_use_id(id)}}));
                    } else {
                        for (index, object) in objects.into_iter().enumerate() {
                            let block_id = match index {
                                0 => id.to_owned(),
                                _ => format!("{id}_{index}"),
                            };
                            blocks.push(json!({"toolUse": {"input": object, "name": name, "toolUseId": sanitize_tool_use_id(&block_id)}}));
                        }
                    }
                }
            }
        }
        blocks.extend(cache);
    }
    Ok(blocks)
}

/// `translate_thinking_blocks_to_reasoning_content_blocks` and
/// `add_thinking_blocks_to_assistant_content`.
fn add_thinking_blocks(blocks: &[Value], to_claude: bool, out: &mut Vec<Value>) {
    let mut reasoning = Vec::new();
    for block in blocks {
        if to_claude && crate::anthropic::chat::messages::is_encrypted_reasoning(block) {
            continue;
        }
        let thinking = block
            .get("thinking")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let data = block
            .get("data")
            .and_then(Value::as_str)
            .filter(|d| !d.trim().is_empty());
        if block.get("type").and_then(Value::as_str) == Some("redacted_thinking")
            && to_claude
            && let Some(data) = data
        {
            use base64::Engine;
            let content = base64::engine::general_purpose::STANDARD
                .encode(native_redacted_thinking_data(data).as_bytes());
            reasoning.push(json!({"reasoningContent": {"redactedContent": content}}));
            continue;
        }
        match block.get("signature").filter(|s| !s.is_null()) {
            Some(signature) if !signature.as_str().is_some_and(str::is_empty) => reasoning.push(
                json!({"reasoningContent": {"reasoningText": {"text": thinking, "signature": signature}}}),
            ),
            _ => {
                if !thinking.trim().is_empty() {
                    out.push(json!({"text": thinking}));
                }
            }
        }
    }
    out.extend(reasoning);
}

fn assistant_blocks(message: &Value, model: &BedrockModel<'_>) -> Result<Vec<Value>, Error> {
    let mut blocks = Vec::new();
    let to_claude = model.accepts_redacted_thinking();
    if let Some(Value::Array(thinking)) = message.get("thinking_blocks") {
        add_thinking_blocks(thinking, to_claude, &mut blocks);
    }
    match message.get("content") {
        Some(Value::Array(parts)) => {
            let parts = non_empty_parts(parts);
            let mut converted = Vec::new();
            for part in parts.iter().filter(|part| part.is_object()) {
                match part.get("type").and_then(Value::as_str) {
                    Some("thinking" | "redacted_thinking") => {
                        add_thinking_blocks(std::slice::from_ref(part), to_claude, &mut converted)
                    }
                    Some("text") => {
                        let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                        if !text.trim().is_empty() {
                            converted.push(json!({"text": text}));
                        }
                    }
                    Some("image_url") => converted.push(image_url_block(part)?),
                    _ => {}
                }
                converted.extend(cache_point(part.get("cache_control"), model));
            }
            blocks.extend(converted);
        }
        Some(Value::String(text)) => {
            let non_content = message.get("tool_calls").is_some_and(truthy)
                || message.get("function_call").is_some_and(truthy);
            if !text.trim().is_empty() {
                blocks.push(json!({"text": text}));
            }
            // An all-whitespace string becomes `None` when the message carries tool calls,
            // and a `None` content adds no cache point.
            if !(text.trim().is_empty() && non_content) {
                blocks.extend(cache_point(message.get("cache_control"), model));
            }
        }
        _ => {}
    }
    if let Some(Value::Array(calls)) = message.get("tool_calls")
        && !calls.is_empty()
    {
        blocks.extend(tool_uses(calls, model)?);
    }
    Ok(blocks)
}

/// `_deduplicate_bedrock_content_blocks`.
fn dedupe(blocks: Vec<Value>, key: &str) -> Vec<Value> {
    let mut seen: HashSet<String> = HashSet::new();
    blocks
        .into_iter()
        .filter(|block| {
            match block
                .get(key)
                .and_then(|keyed| keyed.get("toolUseId"))
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            {
                Some(id) => seen.insert(id.to_owned()),
                None => true,
            }
        })
        .collect()
}

/// `_sort_bedrock_assistant_content_blocks`.
fn sorted_assistant_blocks(mut blocks: Vec<Value>) -> Vec<Value> {
    blocks.sort_by_key(|block| {
        if block.get("reasoningContent").is_some() {
            0
        } else if block.get("toolUse").is_some() || block.get("cachePoint").is_some() {
            2
        } else {
            1
        }
    });
    blocks
}

fn push_user_turn(contents: &mut Vec<Value>, blocks: Vec<Value>) {
    if let Some(last) = contents.last_mut()
        && last["role"] == "user"
        && let Some(Value::Array(existing)) = last.get_mut("content")
    {
        existing.extend(blocks);
        return;
    }
    contents.push(json!({"role": "user", "content": blocks}));
}

/// `_bedrock_converse_messages_pt`.
pub fn converse_messages(
    messages: &[Value],
    model: &BedrockModel<'_>,
) -> Result<Vec<Value>, Error> {
    if messages.is_empty() {
        return Err(bad_request(format!(
            "{BAD_MESSAGE}bedrock requires at least one non-system message"
        )));
    }
    let mut contents: Vec<Value> = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        let start = index;
        let mut user_content = Vec::new();
        while index < messages.len() && role(&messages[index]) == "user" {
            user_content.extend(user_blocks(&messages[index], model)?);
            index += 1;
        }
        if !user_content.is_empty() {
            push_user_turn(&mut contents, user_content);
        }
        let mut tool_content = Vec::new();
        while index < messages.len() && role(&messages[index]) == "tool" {
            let message = &messages[index];
            tool_content.push(tool_result(message)?);
            let control = message
                .get("cache_control")
                .filter(|c| !c.is_null())
                .or_else(|| {
                    message
                        .get("content")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .find_map(|part| part.get("cache_control").filter(|c| !c.is_null()))
                });
            tool_content.extend(cache_point(control, model));
            index += 1;
        }
        let tool_content = dedupe(tool_content, "toolResult");
        if !tool_content.is_empty() {
            push_user_turn(&mut contents, tool_content);
        }
        let mut assistant_content = Vec::new();
        while index < messages.len() && role(&messages[index]) == "assistant" {
            assistant_content.extend(assistant_blocks(&messages[index], model)?);
            index += 1;
        }
        let assistant_content = sorted_assistant_blocks(dedupe(assistant_content, "toolUse"));
        if !assistant_content.is_empty() {
            contents.push(json!({"role": "assistant", "content": assistant_content}));
        }
        if index == start {
            return Err(bad_request(format!(
                "{BAD_MESSAGE}passed in {}",
                crate::base_llm::chat::params::python_dumps(&messages[index])
            )));
        }
    }
    rename_duplicate_documents(&mut contents);
    let mut contents: Vec<Value> = contents
        .into_iter()
        .map(with_text_when_document_only)
        .collect();
    if model.accepts_redacted_thinking()
        && let Some(last) = contents.last_mut()
        && last["role"] == "assistant"
        && let Some(Value::Array(blocks)) = last.get_mut("content")
        && let Some(position) = blocks.iter().rposition(|block| block.get("text").is_some())
    {
        let text = blocks[position]["text"]
            .as_str()
            .unwrap_or_default()
            .trim_end()
            .to_owned();
        blocks[position] = json!({"text": text});
    }
    if !model.supports_tool_result_images() {
        contents = contents
            .into_iter()
            .map(images_beside_tool_results)
            .collect();
    }
    Ok(contents)
}

/// `_rename_duplicate_bedrock_document_names`.
fn rename_duplicate_documents(contents: &mut [Value]) {
    let names = |contents: &[Value]| -> Vec<String> {
        contents
            .iter()
            .flat_map(|message| message["content"].as_array().cloned().unwrap_or_default())
            .filter_map(|block| {
                block
                    .get("document")?
                    .get("name")?
                    .as_str()
                    .map(str::to_owned)
            })
            .filter(|name| !name.is_empty())
            .collect()
    };
    let mut used: HashSet<String> = names(contents).into_iter().collect();
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for message in contents.iter_mut() {
        let Some(Value::Array(blocks)) = message.get_mut("content") else {
            continue;
        };
        for block in blocks {
            let Some(Value::Object(document)) = block.get_mut("document") else {
                continue;
            };
            let Some(name) = document
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            let count = counts.entry(name.clone()).or_insert(0);
            *count += 1;
            if *count > 1 {
                let mut suffix = *count;
                let mut renamed = format!("{name}_{suffix}");
                while used.contains(&renamed) {
                    suffix += 1;
                    renamed = format!("{name}_{suffix}");
                }
                used.insert(renamed.clone());
                document.insert("name".into(), json!(renamed));
            }
        }
    }
}

/// `_with_text_when_document_only`.
fn with_text_when_document_only(message: Value) -> Value {
    let blocks = message["content"].as_array().cloned().unwrap_or_default();
    let needs_text = message["role"] == "user"
        && blocks.iter().any(|block| block.get("document").is_some())
        && blocks.iter().all(|block| block.get("text").is_none());
    if !needs_text {
        return message;
    }
    let cut = match blocks
        .last()
        .is_some_and(|block| block.get("cachePoint").is_some())
    {
        true => blocks.len() - 1,
        false => blocks.len(),
    };
    let mut content = blocks[..cut].to_vec();
    content.push(json!({"text": DOCUMENT_PLACEHOLDER_TEXT}));
    content.extend(blocks[cut..].iter().cloned());
    json!({"role": "user", "content": content})
}

/// `place_tool_result_images` for a model that refuses an image inside a tool result.
fn images_beside_tool_results(message: Value) -> Value {
    let Some(blocks) = message["content"].as_array() else {
        return message;
    };
    let mut rewritten = Vec::new();
    let mut changed = false;
    for block in blocks {
        let Some(result) = block.get("toolResult") else {
            rewritten.push(block.clone());
            continue;
        };
        let parts = result["content"].as_array().cloned().unwrap_or_default();
        let images: Vec<Value> = parts
            .iter()
            .filter_map(|part| part.get("image").cloned())
            .collect();
        if images.is_empty() {
            rewritten.push(block.clone());
            continue;
        }
        changed = true;
        let mut kept: Vec<Value> = parts
            .into_iter()
            .filter(|part| part.get("image").is_none())
            .collect();
        if kept.is_empty() {
            kept.push(json!({"text": TOOL_RESULT_IMAGE_PLACEHOLDER}));
        }
        let mut result = result.clone();
        result["content"] = Value::Array(kept);
        rewritten.push(json!({"toolResult": result}));
        rewritten.extend(images.into_iter().map(|image| json!({"image": image})));
    }
    match changed {
        true => json!({"role": message["role"].clone(), "content": rewritten}),
        false => message,
    }
}

/// `normalize_json_schema_custom_types_to_object`.
fn normalize_custom_types(schema: &mut Value) {
    let Value::Object(node) = schema else { return };
    if node.get("type").and_then(Value::as_str) == Some("custom") {
        node.insert("type".into(), json!("object"));
    }
    for key in ["items", "additionalProperties"] {
        if let Some(child @ Value::Object(_)) = node.get_mut(key) {
            normalize_custom_types(child);
        }
    }
    for key in ["properties", "$defs", "definitions"] {
        if let Some(Value::Object(children)) = node.get_mut(key) {
            children.values_mut().for_each(normalize_custom_types);
        }
    }
    for key in ["anyOf", "allOf", "oneOf"] {
        if let Some(Value::Array(children)) = node.get_mut(key) {
            children.iter_mut().for_each(normalize_custom_types);
        }
    }
}

/// `_bedrock_tools_pt`, with the names it rewrote mapped back for the response.
pub fn converse_tools(
    tools: &[Value],
    model: &BedrockModel<'_>,
) -> Result<(Vec<Value>, std::collections::HashMap<String, String>), Error> {
    let strict_supported = model.supports_strict_tools();
    let mut blocks = Vec::new();
    let mut renamed = std::collections::HashMap::new();
    for (index, tool) in tools.iter().enumerate() {
        let Some(object) = tool.as_object() else {
            continue;
        };
        if ["systemTool", "toolSpec", "cachePoint"]
            .iter()
            .any(|key| object.contains_key(*key))
        {
            blocks.push(tool.clone());
            continue;
        }
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if kind.starts_with("computer_") {
            return Err(Error::Unsupported(
                "a computer use tool on Bedrock Converse",
            ));
        }
        if matches!(
            kind,
            "tool_search_tool_regex_20251119" | "tool_search_tool_bm25_20251119"
        ) {
            continue;
        }
        if !object.contains_key("function") && !object.contains_key("input_schema") {
            continue;
        }
        let default_schema = json!({"type": "object", "properties": {}});
        let (mut parameters, raw_name, description) =
            if object.contains_key("input_schema") && !object.contains_key("function") {
                (
                    object
                        .get("input_schema")
                        .filter(|v| truthy(v))
                        .cloned()
                        .unwrap_or(default_schema),
                    object
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    object.get("description").cloned(),
                )
            } else {
                let function = object.get("function").cloned().unwrap_or(Value::Null);
                (
                    function
                        .get("parameters")
                        .cloned()
                        .unwrap_or(default_schema),
                    function
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    function.get("description").cloned(),
                )
            };
        let raw_name = match raw_name.trim().is_empty() {
            true => format!("litellm_unnamed_tool_{index}"),
            false => raw_name,
        };
        let name = make_valid_bedrock_tool_name(&raw_name);
        if name != raw_name {
            renamed.insert(name.clone(), raw_name.clone());
        }
        let description = match description {
            Some(Value::String(text)) if !text.is_empty() => json!(text),
            Some(other) if truthy(&other) => other,
            _ => json!(name),
        };
        let defs = match parameters.as_object_mut().and_then(|p| p.remove("$defs")) {
            Some(Value::Object(defs)) => defs,
            _ => Map::new(),
        };
        unpack_defs(&mut parameters, &defs);
        normalize_custom_types(&mut parameters);
        let valid_roots = [
            "array", "boolean", "integer", "null", "number", "object", "string",
        ];
        if !parameters
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| valid_roots.contains(&kind))
            && let Value::Object(object) = &mut parameters
        {
            object.insert("type".into(), json!("object"));
        }
        let mut json_schema = Map::from_iter([
            (
                "type".to_owned(),
                parameters.get("type").cloned().unwrap_or(Value::Null),
            ),
            (
                "properties".to_owned(),
                parameters
                    .get("properties")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
            ),
            (
                "required".to_owned(),
                parameters
                    .get("required")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            ),
        ]);
        if strict_supported
            && let Some(additional) = parameters
                .get("additionalProperties")
                .filter(|v| !v.is_null())
        {
            json_schema.insert("additionalProperties".into(), additional.clone());
        }
        let mut spec = Map::from_iter([
            ("inputSchema".to_owned(), json!({"json": json_schema})),
            ("name".to_owned(), json!(name)),
            ("description".to_owned(), description),
        ]);
        let strict = object.get("function").and_then(|f| f.get("strict"));
        if strict_supported && strict.is_some_and(truthy) {
            spec.insert("strict".into(), strict.cloned().unwrap_or(Value::Null));
        }
        blocks.push(json!({"toolSpec": spec}));
        if let Some(control) = object.get("cache_control").filter(|c| !c.is_null())
            && model.accepts_cache_points()
            && control
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("ephemeral")
                == "ephemeral"
        {
            blocks.push(json!({"cachePoint": cache_point_block(Some(control), model)}));
        }
    }
    Ok((blocks, renamed))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::base_llm::chat::params::ChatRequestContext;

    fn model() -> BedrockModel<'static> {
        BedrockModel::new(
            "anthropic.claude-sonnet-4-5-v1:0",
            &ChatRequestContext {
                drop_params: Some(false),
                model_info: None,
            },
        )
    }

    #[rstest]
    #[case("call_1", "call_1")]
    #[case("call 1", "call_1_")]
    fn tool_use_ids_are_sanitized(#[case] id: &str, #[case] prefix: &str) {
        let sanitized = sanitize_tool_use_id(id);
        assert!(sanitized.starts_with(prefix));
        if id != prefix {
            assert_eq!(sanitized.len(), prefix.len() + TOOL_USE_ID_HASH_LEN);
        }
    }

    #[rstest]
    fn a_tool_round_trip_becomes_tool_use_and_tool_result() {
        let messages = vec![
            json!({"role": "user", "content": "weather?"}),
            json!({"role": "assistant", "content": "checking", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get.weather", "arguments": "{\"city\":\"Paris\"}"}}
            ]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "sunny"}),
        ];
        let converted = converse_messages(&messages, &model()).unwrap();
        assert_eq!(
            converted,
            vec![
                json!({"role": "user", "content": [{"text": "weather?"}]}),
                json!({"role": "assistant", "content": [
                    {"text": "checking"},
                    {"toolUse": {"input": {"city": "Paris"}, "name": "get_weather", "toolUseId": "c1"}}
                ]}),
                json!({"role": "user", "content": [{"toolResult": {"content": [{"text": "sunny"}], "toolUseId": "c1"}}]}),
            ]
        );
    }

    #[rstest]
    fn images_and_documents_become_blocks() {
        let messages = vec![json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBE"}, "cache_control": {"type": "ephemeral"}}
        ]})];
        let converted = converse_messages(&messages, &model()).unwrap();
        let content = converted[0]["content"].as_array().unwrap();
        assert_eq!(
            content[0],
            json!({"image": {"source": {"bytes": "AAAA"}, "format": "png"}})
        );
        assert_eq!(content[1]["document"]["format"], "pdf");
        assert!(
            content[1]["document"]["name"]
                .as_str()
                .unwrap()
                .starts_with("DocumentPDFmessages_")
        );
        assert_eq!(content[2], json!({"text": "."}));
        assert_eq!(content[3], json!({"cachePoint": {"type": "default"}}));
    }

    #[rstest]
    fn system_messages_become_system_blocks_and_operator_notes() {
        let (messages, system) = transform_system_message(
            vec![
                json!({"role": "system", "content": "be terse", "cache_control": {"type": "ephemeral"}}),
                json!({"role": "user", "content": "hi"}),
                json!({"role": "system", "content": "later"}),
            ],
            &model(),
        );
        assert_eq!(
            system,
            vec![
                json!({"text": "be terse"}),
                json!({"cachePoint": {"type": "default"}})
            ]
        );
        assert_eq!(messages[1]["content"][0]["text"], CONVERTED_SYSTEM_NOTE);
        assert_eq!(messages[1]["content"][1]["text"], "later");
    }

    #[rstest]
    fn orphaned_tool_blocks_become_text() {
        let messages = neutralize_orphaned_tool_blocks(
            vec![
                json!({"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]}),
                json!({"role": "tool", "tool_call_id": "c", "content": "ok"}),
            ],
            false,
            false,
        );
        assert_eq!(
            messages[0],
            json!({"role": "assistant", "content": "[tool call c: f({})]"})
        );
        assert_eq!(
            messages[1],
            json!({"role": "user", "content": "[tool result for c: ok]"})
        );
    }

    #[rstest]
    fn tools_become_tool_specs() {
        let (tools, renamed) = converse_tools(
            &[json!({"type": "function", "function": {"name": "get.weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]
            }}})],
            &model(),
        )
        .unwrap();
        assert_eq!(
            tools,
            vec![
                json!({"toolSpec": {"inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}, "name": "get_weather", "description": "get_weather"}})
            ]
        );
        assert_eq!(renamed["get_weather"], "get.weather");
    }
}
