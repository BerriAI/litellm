//! The LiteLLM UI content format: span input / output reduced to messages, key/value fields or
//! plain text.

use serde::{Deserialize, Deserializer};
use serde_json::Value;

use crate::normalize::{HIDDEN_BLOCK_TYPES, MessagePayload, encode};

#[macro_rules_attribute::apply(crate::response_type)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[macro_rules_attribute::apply(crate::response_type)]
#[derive(Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", schemars(rename = "UIContent"))]
pub enum UiContent {
    #[cfg_attr(feature = "schema", schemars(title = "UIMessages"))]
    Messages { messages: Vec<UiMessage> },
    #[cfg_attr(feature = "schema", schemars(title = "UIFields"))]
    Fields { fields: Vec<UiField> },
    #[cfg_attr(feature = "schema", schemars(title = "UIText"))]
    Text { text: String },
}

#[macro_rules_attribute::apply(crate::response_type)]
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "schema", schemars(rename = "UIMessage"))]
pub struct UiMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<UiToolCall>>,
}

#[macro_rules_attribute::apply(crate::response_type)]
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "schema", schemars(rename = "UIToolCall"))]
pub struct UiToolCall {
    pub name: String,
    pub arguments: String,
}

#[macro_rules_attribute::apply(crate::response_type)]
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "schema", schemars(rename = "UIField"))]
pub struct UiField {
    pub key: String,
    pub value: String,
}

#[derive(Deserialize)]
struct ToolFunction {
    #[serde(default)]
    name: String,
    arguments: Option<Value>,
}

#[derive(Deserialize)]
struct RawToolCall {
    #[serde(default)]
    name: String,
    args: Option<Value>,
    arguments: Option<Value>,
    function: Option<ToolFunction>,
}

#[derive(Deserialize)]
struct RawMessage {
    role: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(default, deserialize_with = "present")]
    content: Option<Value>,
    name: Option<String>,
    tool_calls: Option<Vec<RawToolCall>>,
    kwargs: Option<Box<RawMessage>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssistantSummary {
    content: Option<String>,
    tool_names: Vec<String>,
}

impl AssistantSummary {
    fn into_ui(self) -> UiMessage {
        let calls: Vec<UiToolCall> = self
            .tool_names
            .into_iter()
            .map(|name| UiToolCall {
                name,
                arguments: "{}".to_owned(),
            })
            .collect();
        UiMessage {
            role: ChatRole::Assistant,
            content: self.content.unwrap_or_default(),
            name: None,
            tool_calls: (!calls.is_empty()).then_some(calls),
        }
    }
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type", default)]
    kind: String,
    text: Option<String>,
}

fn present<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
}

fn known_role(role: &str) -> Option<ChatRole> {
    match role {
        "human" | "user" => Some(ChatRole::User),
        "ai" | "assistant" => Some(ChatRole::Assistant),
        "system" => Some(ChatRole::System),
        "tool" => Some(ChatRole::Tool),
        _ => None,
    }
}

impl RawMessage {
    fn unwrapped(mut self) -> Self {
        match self.kwargs.take() {
            Some(kwargs) => *kwargs,
            None => self,
        }
    }

    fn is_message(&self) -> bool {
        let has_role = self.role.is_some() || self.kind.as_deref().and_then(known_role).is_some();
        has_role
            && (self.content.is_some()
                || self
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty()))
    }

    fn into_ui(self) -> UiMessage {
        let calls: Vec<UiToolCall> = self
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(RawToolCall::into_ui)
            .collect();
        let label = self
            .role
            .as_deref()
            .filter(|role| !role.is_empty())
            .or(self.kind.as_deref())
            .unwrap_or_default();
        let role = known_role(label).unwrap_or(if calls.is_empty() {
            ChatRole::User
        } else {
            ChatRole::Assistant
        });
        UiMessage {
            role,
            content: content_text(self.content),
            name: self.name.filter(|name| !name.is_empty()),
            tool_calls: (!calls.is_empty()).then_some(calls),
        }
    }
}

impl RawToolCall {
    fn into_ui(self) -> UiToolCall {
        match self.function {
            Some(function) => UiToolCall {
                name: if function.name.is_empty() {
                    self.name
                } else {
                    function.name
                },
                arguments: arguments_text(function.arguments),
            },
            None => UiToolCall {
                name: self.name,
                arguments: arguments_text(self.args.or(self.arguments)),
            },
        }
    }
}

fn arguments_text(arguments: Option<Value>) -> String {
    match arguments {
        Some(Value::String(text)) => text,
        None => "{}".to_owned(),
        Some(value) => encode(&value),
    }
}

/// Message content as display text: block lists keep only their text blocks.
fn content_text(content: Option<Value>) -> String {
    match content {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text,
        Some(value) => match Vec::<ContentBlock>::deserialize(&value) {
            Ok(blocks)
                if blocks.iter().all(|block| {
                    block.text.is_some() || HIDDEN_BLOCK_TYPES.contains(&block.kind.as_str())
                }) =>
            {
                blocks
                    .into_iter()
                    .filter_map(|block| block.text)
                    .collect::<Vec<_>>()
                    .join("\n\n")
            }
            _ => encode(&value),
        },
    }
}

fn messages(parsed: &Value) -> Option<Vec<UiMessage>> {
    let raw = MessagePayload::<RawMessage>::deserialize(parsed)
        .ok()?
        .into_messages();
    let unwrapped: Vec<RawMessage> = raw.into_iter().map(RawMessage::unwrapped).collect();
    if unwrapped.is_empty() || !unwrapped.iter().all(RawMessage::is_message) {
        return None;
    }
    Some(unwrapped.into_iter().map(RawMessage::into_ui).collect())
}

fn assistant_summaries(parsed: &Value) -> Option<Vec<UiMessage>> {
    let summaries = Vec::<AssistantSummary>::deserialize(parsed).ok()?;
    if summaries.is_empty() {
        return None;
    }
    Some(
        summaries
            .into_iter()
            .map(AssistantSummary::into_ui)
            .collect(),
    )
}

pub fn to_ui_content(raw: &str) -> UiContent {
    let text = || UiContent::Text {
        text: raw.to_owned(),
    };
    if raw.is_empty() {
        return text();
    }
    let parsed = match serde_json::from_str::<Value>(raw) {
        Ok(Value::String(text)) => return UiContent::Text { text },
        Ok(parsed @ (Value::Array(_) | Value::Object(_))) => parsed,
        _ => return text(),
    };
    if let Some(messages) = messages(&parsed).or_else(|| assistant_summaries(&parsed)) {
        return UiContent::Messages { messages };
    }
    match parsed {
        Value::Object(fields) => UiContent::Fields {
            fields: fields
                .into_iter()
                .map(|(key, value)| UiField {
                    key,
                    value: match value {
                        Value::String(text) => text,
                        value => encode(&value),
                    },
                })
                .collect(),
        },
        _ => text(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn message(role: &'static str, content: &str) -> UiMessage {
        UiMessage {
            role: known_role(role).unwrap(),
            content: content.to_owned(),
            name: None,
            tool_calls: None,
        }
    }

    fn call(name: &str, arguments: &str) -> UiToolCall {
        UiToolCall {
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        }
    }

    #[rstest]
    fn message_arrays_map_roles_and_keep_order() {
        let raw = json!([
            {"role": "system", "content": "be brief"},
            {"role": "human", "content": "hi"},
            {"role": "tool", "name": "lookup", "content": "42"},
            {"role": "narrator", "content": "aside"},
        ]);
        assert_eq!(
            to_ui_content(&raw.to_string()),
            UiContent::Messages {
                messages: vec![
                    message("system", "be brief"),
                    message("user", "hi"),
                    UiMessage {
                        name: Some("lookup".into()),
                        ..message("tool", "42")
                    },
                    message("user", "aside"),
                ]
            }
        );
    }

    #[rstest]
    #[case::args(json!({"name": "get_plan", "args": {"customer_id": "c-1"}}))]
    #[case::arguments(json!({"name": "get_plan", "arguments": "{\"customer_id\": \"c-1\"}"}))]
    #[case::openai(json!({"id": "call_1", "type": "function", "function": {"name": "get_plan", "arguments": "{\"customer_id\": \"c-1\"}"}}))]
    fn assistant_tool_calls_keep_name_and_arguments(#[case] tool_call: Value) {
        let raw = json!({"role": "assistant", "content": null, "tool_calls": [tool_call]});
        assert_eq!(
            to_ui_content(&raw.to_string()),
            UiContent::Messages {
                messages: vec![UiMessage {
                    tool_calls: Some(vec![call("get_plan", "{\"customer_id\": \"c-1\"}")]),
                    ..message("assistant", "")
                }]
            }
        );
    }

    #[rstest]
    fn unknown_role_with_tool_calls_is_the_assistant() {
        let raw =
            json!({"role": "model", "content": "", "tool_calls": [{"name": "f", "args": null}]});
        assert_eq!(
            to_ui_content(&raw.to_string()),
            UiContent::Messages {
                messages: vec![UiMessage {
                    tool_calls: Some(vec![call("f", "{}")]),
                    ..message("assistant", "")
                }]
            }
        );
    }

    #[rstest]
    #[case::text_blocks(json!([{"type": "reasoning", "encrypted_content": "opaque"}, {"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "first"}, {"type": "text", "text": "second"}]), "first\n\nsecond")]
    #[case::unrecognized_block(json!([{"type": "image_url", "image_url": {"url": "u"}}]), r#"[{"type": "image_url", "image_url": {"url": "u"}}]"#)]
    #[case::number(json!(42), "42")]
    fn block_content_keeps_only_display_text(#[case] content: Value, #[case] expected: &str) {
        let raw = json!({"role": "assistant", "content": content});
        assert_eq!(
            to_ui_content(&raw.to_string()),
            UiContent::Messages {
                messages: vec![message("assistant", expected)]
            }
        );
    }

    #[rstest]
    fn langchain_kwargs_are_unwrapped() {
        let raw = json!([
            {"lc": 1, "type": "constructor", "kwargs": {"type": "human", "content": "question"}},
            {"kwargs": {"type": "ai", "content": "", "tool_calls": [{"name": "search", "args": {"q": "x"}}]}},
        ]);
        assert_eq!(
            to_ui_content(&raw.to_string()),
            UiContent::Messages {
                messages: vec![
                    message("user", "question"),
                    UiMessage {
                        tool_calls: Some(vec![call("search", "{\"q\": \"x\"}")]),
                        ..message("assistant", "")
                    },
                ]
            }
        );
    }

    #[rstest]
    fn assistant_summaries_become_assistant_messages() {
        let raw = json!([
            {"content": "`/etc/hosts` has 11 lines.", "tool_names": []},
            {"content": null, "tool_names": ["terminal"]},
        ]);
        assert_eq!(
            to_ui_content(&raw.to_string()),
            UiContent::Messages {
                messages: vec![
                    message("assistant", "`/etc/hosts` has 11 lines."),
                    UiMessage {
                        tool_calls: Some(vec![call("terminal", "{}")]),
                        ..message("assistant", "")
                    },
                ]
            }
        );
    }

    #[rstest]
    #[case::extra_field(r#"[{"content": "x", "tool_names": [], "score": 1}]"#)]
    #[case::missing_tool_names(r#"[{"content": "x"}]"#)]
    fn near_summaries_stay_text(#[case] raw: &str) {
        assert!(matches!(to_ui_content(raw), UiContent::Text { .. }));
    }

    #[rstest]
    fn plain_objects_become_fields_in_key_order() {
        let raw = r#"{"zeta": "plain", "alpha": {"nested": [1, 2]}, "count": 3, "missing": null}"#;
        let field = |key: &str, value: &str| UiField {
            key: key.into(),
            value: value.into(),
        };
        assert_eq!(
            to_ui_content(raw),
            UiContent::Fields {
                fields: vec![
                    field("zeta", "plain"),
                    field("alpha", r#"{"nested": [1, 2]}"#),
                    field("count", "3"),
                    field("missing", "null"),
                ]
            }
        );
    }

    #[rstest]
    #[case::role_without_content(r#"{"role": "admin", "user_id": "u1"}"#)]
    #[case::kwargs_not_a_message(r#"{"kwargs": [], "content": "x"}"#)]
    fn objects_that_are_not_messages_are_fields(#[case] raw: &str) {
        assert!(matches!(to_ui_content(raw), UiContent::Fields { .. }));
    }

    #[rstest]
    #[case::json_string(r#""line one\n\"quoted\"""#, "line one\n\"quoted\"")]
    #[case::cut_json(
        r#"[{"role": "user", "content": "cut of"#,
        r#"[{"role": "user", "content": "cut of"#
    )]
    #[case::plain_words("plain words", "plain words")]
    #[case::number("42", "42")]
    #[case::non_message_list("[1, 2]", "[1, 2]")]
    #[case::message_fields_are_not_a_message(
        r#"["user",null,"hello",null,null,null]"#,
        r#"["user",null,"hello",null,null,null]"#
    )]
    #[case::empty_list("[]", "[]")]
    #[case::empty("", "")]
    fn other_payloads_are_text(#[case] raw: &str, #[case] expected: &str) {
        assert_eq!(
            to_ui_content(raw),
            UiContent::Text {
                text: expected.to_owned()
            }
        );
    }
}
