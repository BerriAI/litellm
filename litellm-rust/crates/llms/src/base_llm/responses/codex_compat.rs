//! Codex CLI wire-format quirks shared by the Responses providers that need them, matching
//! `litellm/llms/base_llm/responses/codex_compat.py`

use litellm_python_compat::{
    json::{dumps, from_json},
    repr::to_str,
    truthy::truthy,
};
use serde_json::{Map, Value, json};

/// Codex history item types that api.openai.com accepts and other Responses backends reject,
/// rewritten to the spelling those backends take. An item that carries nothing to rewrite is
/// dropped
pub fn normalize_codex_input_items(input: Value) -> Value {
    match input {
        Value::Array(items) => Value::Array(items.into_iter().filter_map(normalized).collect()),
        other => other,
    }
}

/// The tools whose `type` the backend accepts; a tool that is not an object passes through
pub fn drop_unsupported_tools(tools: Vec<Value>, supported: &[&str]) -> Vec<Value> {
    tools
        .into_iter()
        .filter(|tool| match tool {
            Value::Object(fields) => fields
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| supported.contains(&kind)),
            _ => true,
        })
        .collect()
}

fn normalized(item: Value) -> Option<Value> {
    let Value::Object(fields) = &item else {
        return Some(item);
    };
    match fields.get("type").and_then(Value::as_str) {
        Some("agent_message") => agent_message(fields),
        Some("context_compaction") => context_compaction(fields),
        Some("local_shell_call") => local_shell_call(fields),
        _ => Some(item),
    }
}

fn agent_message(fields: &Map<String, Value>) -> Option<Value> {
    let text: String = fields
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .map(|block| {
            ["text", "encrypted_content"]
                .iter()
                .filter_map(|name| block.get(*name))
                .map(|value| from_json(value.clone()))
                .find(truthy)
                .map(|value| to_str(&value))
                .unwrap_or_default()
        })
        .collect();
    (!text.is_empty()).then(|| {
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    })
}

fn context_compaction(fields: &Map<String, Value>) -> Option<Value> {
    fields
        .get("encrypted_content")
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
        .map(|content| json!({"type": "compaction", "encrypted_content": content}))
}

fn local_shell_call(fields: &Map<String, Value>) -> Option<Value> {
    let call_id = fields
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|call_id| !call_id.is_empty())?;
    let arguments = match fields.get("action") {
        Some(action @ Value::Object(_)) => dumps(&from_json(action.clone())).ok()?,
        _ => "{}".to_owned(),
    };
    Some(json!({
        "type": "function_call",
        "call_id": call_id,
        "name": "local_shell",
        "arguments": arguments,
    }))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::agent_message_joins_text_and_plaintext_args(
        json!({"type": "agent_message", "content": [{"text": "a"}, {"encrypted_content": "b"}, {"text": "", "encrypted_content": "c"}, "skipped"]}),
        Some(json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "abc"}]}))
    )]
    #[case::agent_message_renders_non_string_text_like_python(
        json!({"type": "agent_message", "content": [{"text": 1.0}, {"text": true}]}),
        Some(json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "1.0True"}]}))
    )]
    #[case::empty_agent_message_is_dropped(json!({"type": "agent_message", "content": [{"text": ""}]}), None)]
    #[case::context_compaction(
        json!({"type": "context_compaction", "encrypted_content": "enc"}),
        Some(json!({"type": "compaction", "encrypted_content": "enc"}))
    )]
    #[case::empty_context_compaction_is_dropped(json!({"type": "context_compaction", "encrypted_content": ""}), None)]
    #[case::local_shell_call_dumps_its_action_like_python(
        json!({"type": "local_shell_call", "call_id": "c1", "action": {"command": ["ls", "é"], "timeout": 1}}),
        Some(json!({"type": "function_call", "call_id": "c1", "name": "local_shell", "arguments": "{\"command\": [\"ls\", \"\\u00e9\"], \"timeout\": 1}"}))
    )]
    #[case::local_shell_call_without_action(
        json!({"type": "local_shell_call", "call_id": "c1", "action": "ls"}),
        Some(json!({"type": "function_call", "call_id": "c1", "name": "local_shell", "arguments": "{}"}))
    )]
    #[case::local_shell_call_without_call_id_is_dropped(json!({"type": "local_shell_call", "call_id": ""}), None)]
    #[case::other_items_pass_through(json!({"type": "message", "role": "user", "content": "hi"}), Some(json!({"type": "message", "role": "user", "content": "hi"})))]
    fn codex_items_take_the_backend_spelling(#[case] item: Value, #[case] expected: Option<Value>) {
        assert_eq!(
            normalize_codex_input_items(json!([item])),
            Value::Array(expected.into_iter().collect())
        );
    }

    #[test]
    fn string_input_is_untouched() {
        assert_eq!(normalize_codex_input_items(json!("hi")), json!("hi"));
    }

    #[test]
    fn unsupported_tool_types_are_dropped() {
        assert_eq!(
            drop_unsupported_tools(
                vec![
                    json!({"type": "function", "name": "f"}),
                    json!({"type": "web_search"}),
                    json!({"name": "untyped"}),
                    json!("opaque"),
                ],
                &["function"],
            ),
            vec![json!({"type": "function", "name": "f"}), json!("opaque")]
        );
    }
}
