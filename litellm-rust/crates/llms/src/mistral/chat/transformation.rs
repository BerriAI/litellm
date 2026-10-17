//! `litellm/llms/mistral/chat/transformation.py`: Mistral at `https://api.mistral.ai/v1`.
//! Tool choices narrow to Mistral's `auto`/`any`/`none`, `seed` goes as `random_seed`,
//! magistral models get Mistral's reasoning system prompt, replayed messages lose what
//! Mistral's schema forbids, and content lists in the answer become text plus
//! `reasoning_content`.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::{
    Error,
    openai::chat::compatible::{
        CompatibleContext, OpenAiCompatibleChatConfig, ResponseShape, content_list_to_str,
        gpt_messages, messages_mut, nearest_declared_reasoning_effort, role, strip_internal_fields,
    },
};

const REASONING_PROMPT_FLAG: &str = "_add_reasoning_prompt";
/// `_get_mistral_reasoning_system_prompt`, verbatim with its indentation.
const REASONING_SYSTEM_PROMPT: &str = r"
        <s>[SYSTEM_PROMPT]system_prompt
        A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \boxed{} in your response.

        Your thinking process must follow the template below:
        <think>
        Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
        </think>

        Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.

        Problem:

        [/SYSTEM_PROMPT][INST]user_message[/INST]<think>
        reasoning_traces
        </think>
        assistant_response</s>[INST]user_message[/INST]
        ";
/// The keys of `ChatCompletionAssistantMessage` besides `role`.
const ASSISTANT_MESSAGE_KEYS: &[&str] = &[
    "content",
    "name",
    "tool_calls",
    "function_call",
    "reasoning_content",
    "cache_control",
    "thinking_blocks",
    "reasoning_items",
];

pub struct MistralChatConfig;

pub const MISTRAL_CHAT_CONFIG: MistralChatConfig = MistralChatConfig;

impl OpenAiCompatibleChatConfig for MistralChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::Mistral
    }

    fn label(&self) -> &'static str {
        "Mistral"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["MISTRAL_AZURE_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://api.mistral.ai/v1")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &["MISTRAL_AZURE_API_KEY", "MISTRAL_API_KEY"]
    }

    /// Mistral always needs `/v1` at the end of the base.
    fn api_base(
        &self,
        configured: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let base = configured
            .filter(|base| !base.is_empty())
            .map(str::to_owned)
            .or_else(|| lookup("MISTRAL_AZURE_API_BASE"))
            .unwrap_or_else(|| "https://api.mistral.ai/v1".to_owned());
        Ok(match base.ends_with("/v1") {
            true => base,
            false => format!("{base}/v1"),
        })
    }

    fn supported_params(&self, context: &CompatibleContext<'_>) -> Vec<&'static str> {
        let magistral = is_magistral(context.model);
        let mut params = vec![
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "seed",
            "stop",
            "response_format",
            "parallel_tool_calls",
        ];
        if magistral {
            params.push("thinking");
        }
        if magistral || context.supports(self.name(), "supports_reasoning") {
            params.push("reasoning_effort");
        }
        params
    }

    fn map_params(
        &self,
        non_default: &Map<String, Value>,
        optional: &mut Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        let magistral = is_magistral(context.model);
        for (name, value) in non_default {
            match name.as_str() {
                "max_tokens" if !non_default.contains_key("max_completion_tokens") => {
                    optional.insert("max_tokens".into(), value.clone());
                }
                "max_completion_tokens" => {
                    optional.insert("max_tokens".into(), value.clone());
                }
                "tools" => {
                    let mut tools = value.clone();
                    remove_json_schema_refs(&mut tools, 10);
                    optional.insert("tools".into(), tools);
                }
                "stream" if value == &Value::Bool(true) => {
                    optional.insert("stream".into(), value.clone());
                }
                "temperature" | "top_p" | "stop" | "response_format" | "parallel_tool_calls" => {
                    optional.insert(name.clone(), value.clone());
                }
                "tool_choice" => {
                    if let Some(choice) = value.as_str() {
                        let choice = match choice {
                            "auto" | "none" => choice,
                            _ => "any",
                        };
                        optional.insert("tool_choice".into(), json!(choice));
                    }
                }
                "seed" => {
                    optional.insert("extra_body".into(), json!({"random_seed": value}));
                }
                "reasoning_effort" | "thinking" if magistral => {
                    optional.insert(REASONING_PROMPT_FLAG.into(), Value::Bool(true));
                }
                "reasoning_effort" => {
                    let effort = match (
                        value.as_str(),
                        context.declared_reasoning_efforts(self.name()),
                    ) {
                        (Some(requested), Some(declared)) => {
                            json!(nearest_declared_reasoning_effort(requested, &declared))
                        }
                        _ => value.clone(),
                    };
                    optional.insert("reasoning_effort".into(), effort);
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn map_extra_body(
        &self,
        extra_body: &mut Map<String, Value>,
        _optional: &mut Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        extra_body.remove("client_metadata");
        Ok(())
    }

    fn transform_messages(
        &self,
        messages: Vec<Value>,
        optional: &Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<Vec<Value>, Error> {
        let mut messages = messages;
        if is_magistral(context.model) && optional.contains_key(REASONING_PROMPT_FLAG) {
            add_reasoning_system_prompt(&mut messages);
        }
        for message in messages.iter_mut() {
            if role(message) == Some("assistant") {
                strip_internal_fields(message);
            }
        }
        let has_media = messages.iter().any(|message| {
            message
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(|parts| {
                    parts.iter().any(|part| {
                        matches!(
                            part.get("type").and_then(Value::as_str),
                            Some("image_url" | "file")
                        )
                    })
                })
        });
        if has_media {
            let mut messages = gpt_messages(messages)?;
            messages.iter_mut().for_each(file_parts_to_file_ids);
            return Ok(messages);
        }
        let messages = messages
            .into_iter()
            .map(|mut message| {
                let text = content_list_to_str(&message);
                if let Some(object) = message.as_object_mut() {
                    if !text.is_empty() {
                        object.insert("content".into(), Value::String(text));
                    }
                    keep_name_on_tool_messages(object);
                    rebuild_tool_calls(object);
                }
                message
            })
            .filter(|message| !is_empty_assistant_message(message))
            .map(|mut message| {
                if let Some(object) = message.as_object_mut() {
                    object.retain(|_, value| !value.is_null());
                }
                message
            })
            .collect();
        gpt_messages(messages)
    }

    fn finish_request(
        &self,
        body: &mut Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        body.remove(REASONING_PROMPT_FLAG);
        body.remove("client_metadata");
        Ok(())
    }

    fn fix_response(&self, body: &mut Map<String, Value>, _shape: &ResponseShape) -> bool {
        let mut changed = false;
        for message in messages_mut(body) {
            match message.get("content") {
                Some(Value::String(content)) if content.is_empty() => {
                    message.insert("content".into(), Value::Null);
                    changed = true;
                }
                Some(Value::Array(blocks)) if !blocks.is_empty() => {
                    let (text, thinking) = split_content_blocks(blocks);
                    message.insert("content".into(), Value::String(text));
                    if !thinking.is_empty() {
                        message.insert("reasoning_content".into(), Value::String(thinking));
                    }
                    changed = true;
                }
                _ => {}
            }
        }
        changed
    }

    fn fix_chunk(&self, chunk: &mut Map<String, Value>, _shape: &ResponseShape) -> bool {
        let mut changed = false;
        let Some(Value::Array(choices)) = chunk.get_mut("choices") else {
            return false;
        };
        for delta in choices
            .iter_mut()
            .filter_map(|choice| choice.get_mut("delta")?.as_object_mut())
        {
            let Some(Value::Array(blocks)) = delta.get("content") else {
                continue;
            };
            let (text, thinking_blocks, reasoning) = normalize_stream_blocks(blocks);
            delta.insert("content".into(), text.map_or(Value::Null, Value::String));
            match reasoning {
                Some(reasoning) if !thinking_blocks.is_empty() => {
                    delta.insert("thinking_blocks".into(), Value::Array(thinking_blocks));
                    delta.insert("reasoning_content".into(), Value::String(reasoning));
                }
                _ => {
                    delta.remove("thinking_blocks");
                    delta.remove("reasoning_content");
                }
            }
            changed = true;
        }
        changed
    }
}

fn is_magistral(model: &str) -> bool {
    model.to_lowercase().contains("magistral")
}

/// `remove_json_schema_refs`.
fn remove_json_schema_refs(value: &mut Value, depth: usize) {
    if depth == 0 {
        return;
    }
    match value {
        Value::Object(object) => {
            object.remove("$id");
            object.remove("$schema");
            object
                .values_mut()
                .for_each(|nested| remove_json_schema_refs(nested, depth - 1));
        }
        Value::Array(items) => items
            .iter_mut()
            .for_each(|item| remove_json_schema_refs(item, depth - 1)),
        _ => {}
    }
}

/// `_add_reasoning_system_prompt_if_needed`.
fn add_reasoning_system_prompt(messages: &mut Vec<Value>) {
    let Some(system) = messages
        .iter_mut()
        .find(|message| role(message) == Some("system"))
        .and_then(Value::as_object_mut)
    else {
        messages.insert(
            0,
            json!({"role": "system", "content": REASONING_SYSTEM_PROMPT}),
        );
        return;
    };
    let content = match system.remove("content") {
        Some(Value::String(existing)) => json!(format!("{REASONING_SYSTEM_PROMPT}\n\n{existing}")),
        Some(Value::Array(existing)) => Value::Array(
            std::iter::once(
                json!({"type": "text", "text": format!("{REASONING_SYSTEM_PROMPT}\n\n")}),
            )
            .chain(existing)
            .collect(),
        ),
        Some(Value::Null) => json!(format!("{REASONING_SYSTEM_PROMPT}\n\nNone")),
        Some(other) => json!(format!("{REASONING_SYSTEM_PROMPT}\n\n{other}")),
        None => json!(format!("{REASONING_SYSTEM_PROMPT}\n\n")),
    };
    system.insert("content".into(), content);
}

/// `_handle_message_with_file`: Mistral takes a file part by its `file_id`.
fn file_parts_to_file_ids(message: &mut Value) {
    let Some(Value::Array(parts)) = message.get_mut("content") else {
        return;
    };
    for part in parts
        .iter_mut()
        .filter_map(Value::as_object_mut)
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("file"))
    {
        let file_id = part
            .get("file")
            .and_then(|file| file.get("file_id"))
            .filter(|id| crate::base_llm::chat::params::truthy(id))
            .cloned();
        if let Some(file_id) = file_id {
            part.insert("file_id".into(), file_id);
            part.remove("file");
        }
    }
}

/// `_handle_name_in_message`: Mistral takes `name` only on tool messages.
fn keep_name_on_tool_messages(message: &mut Map<String, Value>) {
    let Some(name) = message.get("name").filter(|name| !name.is_null()) else {
        return;
    };
    let blank = name.as_str().is_some_and(|name| name.trim().is_empty());
    if message.get("role").and_then(Value::as_str) != Some("tool") || blank {
        message.remove("name");
    }
}

/// `_handle_tool_call_message`: each call as `{id, type: "function", function}`.
fn rebuild_tool_calls(message: &mut Map<String, Value>) {
    let Some(Value::Array(calls)) = message.get_mut("tool_calls") else {
        return;
    };
    for call in calls.iter_mut() {
        *call = json!({
            "id": call.get("id").cloned().unwrap_or(Value::Null),
            "type": "function",
            "function": call.get("function").cloned().unwrap_or(Value::Null),
        });
    }
}

/// `_is_empty_assistant_message`.
fn is_empty_assistant_message(message: &Value) -> bool {
    role(message) == Some("assistant")
        && ASSISTANT_MESSAGE_KEYS
            .iter()
            .all(|key| match message.get(*key) {
                None | Some(Value::Null) => true,
                Some(Value::String(content)) => *key == "content" && content.is_empty(),
                Some(_) => false,
            })
}

/// `_handle_content_list_to_str_conversion` for one message: the last text block, and the
/// last thinking block's text parts joined by newlines.
fn split_content_blocks(blocks: &[Value]) -> (String, String) {
    let mut text = String::new();
    let mut thinking = String::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("thinking") => thinking = thinking_texts(block).join("\n"),
            Some("text") => {
                text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
            }
            _ => {}
        }
    }
    (text, thinking)
}

fn thinking_texts(block: &Value) -> Vec<&str> {
    block
        .get("thinking")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
        .map(|part| part.get("text").and_then(Value::as_str).unwrap_or_default())
        .collect()
}

/// `MistralChatResponseIterator._normalize_content_blocks`.
fn normalize_stream_blocks(blocks: &[Value]) -> (Option<String>, Vec<Value>, Option<String>) {
    let mut text: Option<String> = None;
    let mut thinking_blocks = Vec::new();
    let mut reasoning = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("thinking") => {
                let thought = thinking_texts(block).concat();
                if !thought.is_empty() {
                    thinking_blocks.push(
                        json!({"type": "thinking", "thinking": thought, "signature": "mistral"}),
                    );
                    reasoning.push(thought);
                }
            }
            Some("text") => text.get_or_insert_with(String::new).push_str(
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ),
            _ => {}
        }
    }
    let reasoning = (!reasoning.is_empty()).then(|| reasoning.join("\n"));
    (text, thinking_blocks, reasoning)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::openai::chat::compatible::{fix_response, transform_request};

    fn object(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("an object")
        };
        map
    }

    fn send(model: &str, params: Value) -> Map<String, Value> {
        transform_request(
            &MISTRAL_CHAT_CONFIG,
            &CompatibleContext {
                model,
                drop_params: false,
                model_info: None,
            },
            object(params),
        )
        .unwrap()
        .body
    }

    #[rstest]
    #[case::bare("https://example.ai", "https://example.ai/v1")]
    #[case::versioned("https://example.ai/v1", "https://example.ai/v1")]
    fn the_base_always_ends_in_v1(#[case] configured: &str, #[case] expected: &str) {
        assert_eq!(
            MISTRAL_CHAT_CONFIG
                .api_base(Some(configured), &|_| None)
                .unwrap(),
            expected
        );
    }

    #[rstest]
    fn params_take_mistral_names() {
        let sent = send(
            "mistral-large-latest",
            json!({"messages": [], "max_tokens": 1,
            "max_completion_tokens": 9, "seed": 3, "stream": false, "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"$schema": "s", "$id": "i", "type": "object"}}}]}),
        );
        assert_eq!(
            Value::Object(sent),
            json!({"model": "mistral-large-latest", "messages": [], "max_tokens": 9,
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                "tool_choice": "any", "random_seed": 3})
        );
    }

    #[rstest]
    fn magistral_reasoning_adds_the_system_prompt_instead_of_the_param() {
        let sent = send(
            "magistral-medium-latest",
            json!({"reasoning_effort": "high",
            "messages": [{"role": "system", "content": "be brief"}]}),
        );
        assert_eq!(
            sent["messages"],
            json!([{"role": "system", "content": format!("{REASONING_SYSTEM_PROMPT}\n\nbe brief")}])
        );
        assert!(!sent.contains_key("reasoning_effort"));
        assert!(!sent.contains_key(REASONING_PROMPT_FLAG));
    }

    #[rstest]
    fn replayed_messages_lose_what_mistral_forbids() {
        let sent = send(
            "mistral-small",
            json!({"messages": [
            {"role": "user", "name": "u", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "", "reasoning_content": "r", "thinking_blocks": []},
            {"role": "assistant", "content": null, "tool_calls": [{"index": 0, "id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "name": "f", "tool_call_id": "c", "content": "ok"}]}),
        );
        assert_eq!(
            sent["messages"],
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "name": "f", "tool_call_id": "c", "content": "ok"}])
        );
    }

    #[rstest]
    fn file_parts_go_by_file_id() {
        let sent = send(
            "mistral-small",
            json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "read"}, {"type": "file", "file": {"file_id": "abc"}}]}]}),
        );
        assert_eq!(
            sent["messages"][0]["content"],
            json!([{"type": "text", "text": "read"}, {"type": "file", "file_id": "abc"}])
        );
    }

    #[rstest]
    #[case::empty(json!(""), json!({"role": "assistant", "content": null}))]
    #[case::blocks(json!([{"type": "thinking", "thinking": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}, {"type": "text", "text": "done"}]),
        json!({"role": "assistant", "content": "done", "reasoning_content": "a\nb"}))]
    fn answers_become_text_and_reasoning(#[case] content: Value, #[case] message: Value) {
        let mut body =
            object(json!({"choices": [{"message": {"role": "assistant", "content": content}}]}));
        assert!(fix_response(
            &MISTRAL_CHAT_CONFIG,
            &mut body,
            &ResponseShape::default()
        ));
        assert_eq!(body["choices"][0]["message"], message);
    }

    #[rstest]
    fn streamed_blocks_become_text_and_thinking() {
        let mut chunk = object(json!({"choices": [{"index": 0, "delta": {"content": [
            {"type": "thinking", "thinking": [{"type": "text", "text": "hm"}]}, {"type": "text", "text": "ok"}]}}]}));
        assert!(MISTRAL_CHAT_CONFIG.fix_chunk(&mut chunk, &ResponseShape::default()));
        assert_eq!(
            chunk["choices"][0]["delta"],
            json!({"content": "ok", "thinking_blocks": [{"type": "thinking", "thinking": "hm", "signature": "mistral"}], "reasoning_content": "hm"})
        );
    }
}
