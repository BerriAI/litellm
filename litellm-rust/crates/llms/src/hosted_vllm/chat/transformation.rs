//! `litellm/llms/hosted_vllm/chat/transformation.py`: a vLLM server at the deployment's
//! `api_base`. Tools lose `additionalProperties: false` and `strict` and custom tools become
//! function tools, `thinking` becomes a `reasoning_effort`, replayed assistant turns lose the
//! fields vLLM rejects with their content lists flattened, and video files go as `video_url`.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::chat::params::python_dumps,
    openai::chat::compatible::{
        CompatibleContext, GPT_BASE_PARAMS, OpenAiCompatibleChatConfig, copy_supported,
        gpt_messages, reasoning_effort_from_thinking_budget, role,
    },
    vertex_ai::gemini::schema::{remove_additional_properties, remove_strict_from_schema},
};

pub struct HostedVllmChatConfig;

pub const HOSTED_VLLM_CHAT_CONFIG: HostedVllmChatConfig = HostedVllmChatConfig;

impl OpenAiCompatibleChatConfig for HostedVllmChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::HostedVllm
    }

    fn label(&self) -> &'static str {
        "hosted vLLM"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["HOSTED_VLLM_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        None
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &["HOSTED_VLLM_API_KEY"]
    }

    /// vLLM needs no key, and Python sends this one when none is set.
    fn default_api_key(&self) -> Option<&'static str> {
        Some("fake-api-key")
    }

    fn supported_params(&self, _context: &CompatibleContext<'_>) -> Vec<&'static str> {
        GPT_BASE_PARAMS
            .iter()
            .copied()
            .chain(["response_format", "reasoning_effort", "thinking"])
            .collect()
    }

    fn map_params(
        &self,
        non_default: &Map<String, Value>,
        optional: &mut Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        let mut params = non_default.clone();
        if let Some(mut tools) = params.remove("tools").filter(|tools| !tools.is_null()) {
            remove_additional_properties(&mut tools);
            remove_strict_from_schema(&mut tools);
            if let Value::Array(items) = tools {
                tools = Value::Array(
                    items
                        .into_iter()
                        .enumerate()
                        .map(|(index, tool)| custom_tool_as_function(index, tool))
                        .collect(),
                );
            }
            params.insert("tools".into(), tools);
        }
        if let Some(Value::Object(thinking)) = params.remove("thinking")
            && thinking.get("type").and_then(Value::as_str) == Some("enabled")
            && !params.contains_key("reasoning_effort")
        {
            let budget = thinking
                .get("budget_tokens")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            params.insert(
                "reasoning_effort".into(),
                json!(reasoning_effort_from_thinking_budget(budget)),
            );
        }
        copy_supported(&params, optional, &self.supported_params(context));
        Ok(())
    }

    fn transform_messages(
        &self,
        messages: Vec<Value>,
        _optional: &Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<Vec<Value>, Error> {
        let messages = messages
            .into_iter()
            .map(|mut message| {
                match role(&message) {
                    Some("assistant") => assistant_for_vllm(&mut message),
                    Some("user") => video_files_as_video_urls(&mut message),
                    _ => {}
                }
                message
            })
            .collect();
        gpt_messages(messages)
    }
}

/// `_convert_custom_tools_to_function_tools` for one tool.
fn custom_tool_as_function(index: usize, tool: Value) -> Value {
    if tool.get("type").and_then(Value::as_str) != Some("custom") {
        return tool;
    }
    let custom = tool
        .get("custom")
        .filter(|custom| custom.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let pick = |key: &str| {
        custom
            .get(key)
            .filter(|value| crate::base_llm::chat::params::truthy(value))
            .or_else(|| tool.get(key).filter(|value| !value.is_null()))
            .cloned()
    };
    let name = match pick("name") {
        Some(Value::String(name)) => name,
        Some(other) => python_str(&other),
        None => format!("custom_tool_{index}"),
    };
    let parameters = match pick("input_schema") {
        Some(schema @ Value::Object(_)) => schema,
        _ => json!({
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Raw tool input payload."}},
            "required": ["input"],
        }),
    };
    let mut function = Map::from_iter([
        ("name".to_owned(), Value::String(name)),
        ("parameters".to_owned(), parameters),
    ]);
    if let Some(Value::String(description)) = pick("description") {
        function.insert("description".into(), Value::String(description));
    }
    json!({"type": "function", "function": function})
}

fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => python_dumps(other),
    }
}

/// The assistant half of `HostedVLLMChatConfig._transform_messages`.
fn assistant_for_vllm(message: &mut Value) {
    let Some(object) = message.as_object_mut() else {
        return;
    };
    object.remove("thinking_blocks");
    if !object
        .get("reasoning_content")
        .is_some_and(Value::is_string)
    {
        object.remove("reasoning_content");
    }
    let Some(Value::Array(parts)) = object.get("content").cloned() else {
        return;
    };
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut blocks = Vec::new();
    let mut structured = false;
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("text") if part.is_object() => {
                texts.push(
                    part.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                );
                blocks.push(part);
            }
            Some("tool_use") if part.is_object() => {
                let input = part.get("input").cloned().unwrap_or_else(|| json!({}));
                let arguments = match input {
                    Value::String(text) => text,
                    other => python_dumps(&other),
                };
                tool_calls.push(json!({
                    "id": part.get("id").cloned().unwrap_or(Value::Null),
                    "type": "function",
                    "function": {"name": part.get("name").cloned().unwrap_or(Value::Null), "arguments": arguments},
                }));
            }
            _ => {
                blocks.push(part);
                structured = true;
            }
        }
    }
    if !tool_calls.is_empty() {
        match object.get_mut("tool_calls") {
            Some(Value::Array(existing)) => {
                let known: Vec<Value> = existing
                    .iter()
                    .filter_map(|call| call.get("id").filter(|id| !id.is_null()).cloned())
                    .collect();
                existing.extend(
                    tool_calls
                        .into_iter()
                        .filter(|call| !known.contains(&call["id"])),
                );
            }
            _ => {
                object.insert("tool_calls".into(), Value::Array(tool_calls));
            }
        }
    }
    let content = match structured {
        true => Value::Array(blocks),
        false => Value::String(texts.join("\n")),
    };
    object.insert("content".into(), content);
}

/// `_is_video_file`: a `file` part whose format, data URL or file id names a video.
fn is_video_file(part: &Value) -> bool {
    if part.get("type").and_then(Value::as_str) != Some("file") {
        return false;
    }
    let file = part.get("file");
    let field = |key: &str| {
        file.and_then(|file| file.get(key))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
    };
    if field("format").is_some_and(|format| format.starts_with("video/")) {
        return true;
    }
    if let Some(data) = field("file_data") {
        return data_url_mime(data).is_some_and(|mime| mime.starts_with("video/"));
    }
    if let Some(id) = field("file_id") {
        return url_video_mime(id);
    }
    false
}

/// `parse_mime_type`: the media type of a `data:` URL.
fn data_url_mime(data: &str) -> Option<&str> {
    let rest = data.strip_prefix("data:")?;
    let end = rest.find([';', ','])?;
    Some(&rest[..end])
}

/// `get_image_mime_type_from_url` for the video extensions it knows.
fn url_video_mime(url: &str) -> bool {
    let path = url.split(['?', '#']).next().unwrap_or(url).to_lowercase();
    [
        ".mp4", ".mov", ".avi", ".flv", ".mpeg", ".mpg", ".webm", ".wmv", ".3gpp",
    ]
    .iter()
    .any(|extension| path.ends_with(extension))
}

fn video_files_as_video_urls(message: &mut Value) {
    let Some(Value::Array(parts)) = message.get_mut("content") else {
        return;
    };
    for part in parts.iter_mut() {
        if !is_video_file(part) {
            continue;
        }
        let file = part.get("file").cloned().unwrap_or_default();
        let url = ["file_id", "file_data"]
            .iter()
            .find_map(|key| {
                file.get(*key)
                    .filter(|value| crate::base_llm::chat::params::truthy(value))
            })
            .cloned();
        if let Some(url) = url {
            *part = json!({"type": "video_url", "video_url": {"url": url}});
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::openai::chat::compatible::transform_request;

    fn send(params: Value) -> Map<String, Value> {
        let Value::Object(body) = params else {
            panic!("an object")
        };
        transform_request(
            &HOSTED_VLLM_CHAT_CONFIG,
            &CompatibleContext {
                model: "Qwen/Qwen3-8B",
                drop_params: false,
                model_info: None,
            },
            body,
        )
        .unwrap()
        .body
    }

    #[rstest]
    fn tools_lose_strict_and_closed_objects_and_custom_tools_become_functions() {
        let sent = send(json!({"messages": [], "tools": [
            {"type": "function", "function": {"name": "f", "strict": true,
                "parameters": {"type": "object", "additionalProperties": false, "properties": {}}}},
            {"type": "custom", "custom": {"name": "grep", "description": "search"}},
        ]}));
        assert_eq!(
            sent["tools"],
            json!([
                {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}},
                {"type": "function", "function": {"name": "grep", "description": "search", "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": "Raw tool input payload."}},
                    "required": ["input"]}}},
            ])
        );
    }

    #[rstest]
    #[case::budget(json!({"thinking": {"type": "enabled", "budget_tokens": 3000}}), Some("medium"))]
    #[case::effort_wins(json!({"thinking": {"type": "enabled", "budget_tokens": 3000}, "reasoning_effort": "low"}), Some("low"))]
    #[case::disabled(json!({"thinking": {"type": "disabled"}}), None)]
    fn thinking_becomes_a_reasoning_effort(#[case] params: Value, #[case] effort: Option<&str>) {
        let mut body = json!({"messages": []});
        body.as_object_mut()
            .unwrap()
            .extend(params.as_object().unwrap().clone());
        let sent = send(body);
        assert!(!sent.contains_key("thinking"));
        assert_eq!(sent.get("reasoning_effort").and_then(Value::as_str), effort);
    }

    #[rstest]
    fn assistant_turns_are_flattened_for_vllm() {
        let sent = send(json!({"messages": [
            {"role": "assistant", "thinking_blocks": [{"type": "thinking"}], "reasoning_content": {"x": 1},
             "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"q": 1}}]},
            {"role": "user", "content": [{"type": "file", "file": {"file_data": "data:video/mp4;base64,AA"}}]},
        ]}));
        assert_eq!(
            sent["messages"],
            json!([
                {"role": "assistant", "content": "a\nb", "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{\"q\": 1}"}}]},
                {"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}]},
            ])
        );
    }

    #[rstest]
    fn the_key_defaults_for_a_server_that_takes_none() {
        assert_eq!(
            HOSTED_VLLM_CHAT_CONFIG.api_key(None, &|_| None).unwrap(),
            "fake-api-key"
        );
    }

    #[rstest]
    fn an_api_base_is_required() {
        assert!(HOSTED_VLLM_CHAT_CONFIG.api_base(None, &|_| None).is_err());
    }
}
