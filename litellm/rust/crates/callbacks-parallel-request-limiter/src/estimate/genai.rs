use serde_json::{Map, Value, json};

use super::{
    Out, PythonServed,
    python::{Py, json_length, py, text, truthy},
};

pub const CALL_TYPES: [&str; 4] = [
    "generate_content",
    "agenerate_content",
    "generate_content_stream",
    "agenerate_content_stream",
];

fn mapping(value: Option<&Value>) -> Option<&Map<String, Value>> {
    value.and_then(Value::as_object)
}

fn config(request: &Map<String, Value>) -> Option<&Value> {
    if request.contains_key("config") {
        request.get("config")
    } else {
        request.get("generationConfig")
    }
}

pub fn request_config(request: &Map<String, Value>) -> Option<&Map<String, Value>> {
    mapping(config(request))
}

fn typed_text_fields(
    value: Option<&Value>,
    keys: &[&str],
    required: Option<&str>,
) -> Out<Option<Map<String, Value>>> {
    let Some(fields) = mapping(value) else {
        return Ok(None);
    };
    if required.is_some_and(|required| !fields.contains_key(required)) {
        return Ok(None);
    }
    let mut kept = Map::new();
    for key in keys {
        if let Some(member) = fields.get(*key) {
            if text(member)?.is_none() {
                return Ok(None);
            }
            kept.insert((*key).to_owned(), member.clone());
        }
    }
    Ok(Some(kept))
}

fn either(
    first: Option<Map<String, Value>>,
    second: Out<Option<Map<String, Value>>>,
) -> Out<Option<Map<String, Value>>> {
    match first {
        Some(fields) if !fields.is_empty() => Ok(Some(fields)),
        _ => second,
    }
}

fn member<'a>(fields: &'a Map<String, Value>, key: &str) -> Out<Option<&'a str>> {
    match fields.get(key) {
        Some(value) => text(value),
        None => Ok(None),
    }
}

fn first_truthy<'a>(fields: &'a Map<String, Value>, keys: &[&str]) -> Out<Option<&'a str>> {
    for key in keys {
        if let Some(found) = member(fields, key)?.filter(|found| !found.is_empty()) {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

fn media_part(part: &Map<String, Value>) -> Out<Option<Value>> {
    let inline_keys = ["data", "mime_type", "mimeType"];
    let file_keys = ["file_uri", "fileUri", "mime_type", "mimeType"];
    let inline = either(
        typed_text_fields(part.get("inline_data"), &inline_keys, Some("data"))?,
        typed_text_fields(part.get("inlineData"), &inline_keys, Some("data")),
    )?;
    let file = either(
        typed_text_fields(part.get("file_data"), &file_keys, None)?,
        typed_text_fields(part.get("fileData"), &file_keys, None),
    )?;
    let Some(media) = inline.as_ref().or(file.as_ref()) else {
        return Ok(None);
    };
    let declared = first_truthy(media, &["mime_type", "mimeType"])?;
    let mime = declared.or(inline.as_ref().map(|_| "image/jpeg"));
    let source_present = match (&inline, &file) {
        (Some(_), _) => true,
        (None, Some(file)) => first_truthy(file, &["file_uri", "fileUri"])?.is_some(),
        (None, None) => false,
    };
    if !source_present {
        return Ok(None);
    }
    let image = match mime {
        Some(mime) if mime.is_ascii() => mime.to_ascii_lowercase().starts_with("image/"),
        Some(_) => return Err(PythonServed),
        None => false,
    };
    Ok(Some(if image {
        json!({"type": "image_url", "image_url": {}})
    } else {
        json!({"type": "file", "file": {}})
    }))
}

fn message_part(part: &Value) -> Out<Option<Value>> {
    match py(part)? {
        Py::Str(text) => Ok(Some(json!({"type": "text", "text": text}))),
        Py::Dict(fields) => match fields.get("text").map(text).transpose()?.flatten() {
            Some(text) => Ok(Some(json!({"type": "text", "text": text}))),
            None => media_part(fields),
        },
        _ => Ok(None),
    }
}

fn content_parts(value: Option<&Value>) -> Vec<&Value> {
    match value {
        Some(Value::Object(_)) => value.into_iter().collect(),
        Some(Value::Array(items)) => items.iter().collect(),
        _ => Vec::new(),
    }
}

fn is_text(part: &Value) -> bool {
    part.get("type").and_then(Value::as_str) == Some("text")
}

fn tool_response(part: &Value) -> Out<Option<Value>> {
    let Some(fields) = part.as_object() else {
        return Ok(None);
    };
    let response = |key: &str| -> Out<Option<Map<String, Value>>> {
        let Some(response) = mapping(fields.get(key)) else {
            return Ok(None);
        };
        if let Some(name) = response.get("name")
            && text(name)?.is_none()
        {
            return Ok(None);
        }
        Ok(Some(
            response
                .iter()
                .filter(|(key, _)| matches!(key.as_str(), "name" | "response" | "parts"))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ))
    };
    let Some(response) = either(response("functionResponse")?, response("function_response"))?
    else {
        return Ok(None);
    };
    let result_length = match response.get("response") {
        Some(value) => json_length(value, false)?,
        None => 2,
    };
    let result = Value::String("r".repeat(result_length as usize));
    let parts: Vec<&Value> = match response.get("parts") {
        None => Vec::new(),
        Some(Value::Array(items)) => items.iter().collect(),
        Some(_) => Vec::new(),
    };
    let media = parts
        .into_iter()
        .filter_map(Value::as_object)
        .map(media_part)
        .filter_map(Result::transpose)
        .collect::<Out<Vec<Value>>>()?;
    let content = if media.is_empty() {
        result
    } else {
        let mut parts = vec![json!({"type": "text", "text": result})];
        parts.extend(media);
        Value::Array(parts)
    };
    Ok(Some(json!({"role": "tool", "content": content})))
}

fn content_messages(content: &Value) -> Out<Vec<Value>> {
    let Some(fields) = content.as_object() else {
        return Ok(Vec::new());
    };
    let parts = content_parts(fields.get("parts"));
    let converted = parts
        .iter()
        .map(|part| message_part(part))
        .filter_map(Result::transpose)
        .collect::<Out<Vec<Value>>>()?;
    let role = fields.get("role").map(text).transpose()?.flatten();
    let role = if fields.contains_key("role") {
        role
    } else {
        Some("user")
    };
    match role {
        Some("user") => {
            let mut messages = Vec::new();
            if !converted.is_empty() {
                let content = if converted.len() == 1 && is_text(&converted[0]) {
                    converted[0]["text"].clone()
                } else {
                    Value::Array(converted)
                };
                messages.push(json!({"role": "user", "content": content}));
            }
            for part in &parts {
                messages.extend(tool_response(part)?);
            }
            Ok(messages)
        }
        Some("model") => {
            let joined: String = converted
                .iter()
                .filter(|part| is_text(part))
                .filter_map(|part| part["text"].as_str())
                .collect();
            let content = if converted.iter().any(|part| !is_text(part)) {
                Value::Array(converted)
            } else if joined.is_empty() {
                Value::Null
            } else {
                Value::String(joined)
            };
            Ok(vec![json!({"role": "assistant", "content": content})])
        }
        _ => Ok(Vec::new()),
    }
}

fn system_messages(instruction: Option<&Value>) -> Out<Vec<Value>> {
    let Some(instruction) = instruction else {
        return Ok(Vec::new());
    };
    if !truthy(instruction)? {
        return Ok(Vec::new());
    }
    let Some(fields) = instruction.as_object() else {
        return Err(PythonServed);
    };
    let parts = fields.get("parts");
    let Some(parts) = parts.filter(|parts| truthy(parts).unwrap_or(true)) else {
        return Ok(Vec::new());
    };
    let first = match py(parts)? {
        Py::List(items) => &items[0],
        Py::Str(_) => return Ok(Vec::new()),
        _ => return Err(PythonServed),
    };
    let found = match py(first)? {
        Py::Dict(fields) => fields.get("text").cloned(),
        Py::Str(text) if text.contains("text") => return Err(PythonServed),
        Py::Str(_) => None,
        Py::List(items) if items.iter().any(|item| text(item) == Ok(Some("text"))) => {
            return Err(PythonServed);
        }
        Py::List(_) => None,
        _ => return Err(PythonServed),
    };
    Ok(found
        .map(|content| vec![json!({"role": "system", "content": content})])
        .unwrap_or_default())
}

fn first_present<'a>(fields: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .find_map(|key| fields.get(*key).filter(|value| !value.is_null()))
}

fn side_paths_need_python(
    request: &Map<String, Value>,
    config: Option<&Map<String, Value>>,
) -> Out<bool> {
    if let Some(config) = config {
        let mime = first_present(config, &["responseMimeType", "response_mime_type"]);
        let json_mime = mime.is_none_or(|mime| mime.as_str() == Some("application/json"));
        let schema = first_present(
            config,
            &[
                "responseJsonSchema",
                "response_json_schema",
                "responseSchema",
                "response_schema",
            ],
        );
        if json_mime && schema.is_some_and(Value::is_object) {
            return Ok(true);
        }
    }
    if let Some(Value::Array(tools)) = request.get("tools")
        && tools.iter().any(|tool| {
            tool.as_object()
                .is_none_or(|tool| tool.contains_key("functionDeclarations"))
        })
    {
        return Ok(true);
    }
    let tool_config = match request.get("toolConfig") {
        Some(value) if truthy(value)? => Some(value),
        _ => request.get("tool_config"),
    };
    if let Some(tool_config) = tool_config
        && truthy(tool_config)?
    {
        let Some(tool_config) = tool_config.as_object() else {
            return Ok(true);
        };
        let calling = tool_config.get("functionCallingConfig");
        let mode = match calling {
            None => None,
            Some(Value::Object(calling)) => calling.get("mode"),
            Some(_) => return Ok(true),
        };
        if matches!(mode, Some(Value::Array(_) | Value::Object(_))) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn translated_messages(request: &Map<String, Value>, call_type: &str) -> Out<Option<Value>> {
    if !CALL_TYPES.contains(&call_type) {
        return Ok(None);
    }
    let contents: Vec<&Value> = match request.get("contents") {
        Some(contents @ Value::Object(_)) => vec![contents],
        Some(Value::Array(items)) => items.iter().collect(),
        _ => return Ok(None),
    };
    if side_paths_need_python(request, request_config(request))? {
        return Err(PythonServed);
    }
    let instruction = match request.get("systemInstruction") {
        Some(value) if truthy(value)? => Some(value),
        _ => request.get("system_instruction"),
    };
    let mut messages = system_messages(instruction)?;
    for content in contents {
        messages.extend(content_messages(content)?);
    }
    Ok(Some(Value::Array(messages)))
}
