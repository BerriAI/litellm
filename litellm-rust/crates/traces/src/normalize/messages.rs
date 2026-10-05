//! The common message format normalizers emit for span input and output: a JSON array of
//! `{role, content, tool_calls?, name?}` that the UI renders as a conversation.

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, ser::Formatter};
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
};

use litellm_llms_types::{formats::chat_completions::ChatMessageContent, recognized::Recognized};

use super::{CallEvidence, CallKey, attr};

/// Characters of a span's input kept for list views.
pub(super) const PREVIEW_CHARS: usize = 240;

/// Content blocks that carry no display text: reasoning and the model's own tool requests.
pub(crate) const HIDDEN_BLOCK_TYPES: [&str; 6] = [
    "reasoning",
    "thinking",
    "redacted_thinking",
    "function_call",
    "tool_use",
    "tool_call",
];

fn display_text(content: &Recognized<ChatMessageContent>) -> String {
    match content {
        Recognized::Known(ChatMessageContent::Text(text)) => text.clone(),
        Recognized::Known(ChatMessageContent::Parts(blocks)) => blocks
            .iter()
            .filter(|block| {
                !block
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| HIDDEN_BLOCK_TYPES.contains(&kind))
            })
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n\n"),
        Recognized::Unrecognized(value) => encode(value),
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(transparent)]
pub(super) struct ToolCall(IndexMap<String, Value>);

#[derive(Deserialize)]
pub(super) struct ResponseMetadata {
    pub id: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum MessagePayload<T> {
    Single {
        #[serde(flatten)]
        message: T,
    },
    Batch(Vec<T>),
}

impl<T> MessagePayload<T> {
    pub(crate) fn into_messages(self) -> Vec<T> {
        match self {
            Self::Single { message } => vec![message],
            Self::Batch(messages) => messages,
        }
    }
}

pub(super) fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Default, Deserialize)]
struct EventFields {
    #[serde(default, deserialize_with = "present")]
    role: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    content: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    tool_calls: Option<Value>,
    #[serde(flatten)]
    indexed: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
pub(super) struct EventMessage {
    #[serde(rename = "event.name")]
    name: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "present")]
    message: Option<Recognized<EventFields>>,
    #[serde(rename = "message.role", default, deserialize_with = "present")]
    role: Option<Value>,
    #[serde(rename = "message.content", default, deserialize_with = "present")]
    content: Option<Value>,
    #[serde(flatten)]
    body: EventFields,
}

impl EventMessage {
    pub(super) fn recorded(&self) -> Option<(bool, Value)> {
        self.normalized(self.name.as_ref()?.known()?)
    }

    fn normalized(&self, name: &str) -> Option<(bool, Value)> {
        let (output, role) = match name {
            "gen_ai.system.message" => (false, "system"),
            "gen_ai.user.message" | "gen_ai.content.prompt" => (false, "user"),
            "gen_ai.assistant.message" | "gen_ai.choice" | "gen_ai.content.completion" => {
                (true, "assistant")
            }
            "gen_ai.tool.message" => (true, "tool"),
            _ => return None,
        };
        let empty = EventFields::default();
        let body = match &self.message {
            Some(Recognized::Known(message)) => message,
            Some(Recognized::Unrecognized(_)) => &empty,
            None => &self.body,
        };
        let content = body.content.as_ref().or(self.content.as_ref());
        let calls = event_tool_calls(body);
        if content.is_none() && calls.is_none() {
            return None;
        }
        Some((
            output,
            serde_json::json!({
                "role": body.role.as_ref().or(self.role.as_ref()).cloned().unwrap_or(Value::from(role)),
                "content": content.cloned().unwrap_or(Value::from("")),
                "tool_calls": calls,
            }),
        ))
    }
}

/// One part of an OpenTelemetry GenAI (`type` + `content`) or Gemini (`text`) message.
#[derive(Deserialize)]
struct Part {
    #[serde(rename = "type")]
    kind: Option<String>,
    content: Option<Value>,
    text: Option<String>,
    id: Option<Value>,
    name: Option<String>,
    arguments: Option<Value>,
    response: Option<Value>,
}

/// A message as instrumentations record it: OpenAI chat (`role` + `content`), LangChain
/// (`type`, wrapped in `kwargs` by `dumpd` or `data` by `messages_to_dict`), or OpenTelemetry
/// GenAI and Gemini (`role` + `parts`).
#[derive(Deserialize)]
pub(super) struct RawMessage {
    kwargs: Option<Box<RawMessage>>,
    data: Option<Box<RawMessage>>,
    #[serde(rename = "type")]
    kind: Option<String>,
    role: Option<String>,
    content: Option<Recognized<ChatMessageContent>>,
    parts: Option<Vec<Part>>,
    tool_calls: Option<Vec<ToolCall>>,
    name: Option<Value>,
    pub response_metadata: Option<ResponseMetadata>,
}

#[derive(Serialize)]
pub(super) struct Message {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<Value>,
}

impl RawMessage {
    pub(super) fn unwrapped(&self) -> &Self {
        self.kwargs
            .as_deref()
            .or(self.data.as_deref())
            .unwrap_or(self)
    }

    fn role(&self) -> &str {
        let fields = self.unwrapped();
        let raw = fields
            .kind
            .as_deref()
            .filter(|role| !role.is_empty())
            .or_else(|| fields.role.as_deref().filter(|role| !role.is_empty()))
            .or_else(|| self.kind.as_deref().filter(|role| !role.is_empty()))
            .unwrap_or_default();
        match raw {
            "human" => "user",
            "ai" | "model" => "assistant",
            other => other,
        }
    }

    fn is_message(&self) -> bool {
        let fields = self.unwrapped();
        !self.role().is_empty()
            && (fields.content.is_some() || fields.parts.is_some() || fields.tool_calls.is_some())
    }

    pub(super) fn normalized(&self) -> Message {
        let fields = self.unwrapped();
        let role = self.role().to_owned();
        let parts = fields.parts.as_deref().unwrap_or_default();
        let content = match &fields.content {
            Some(content) => display_text(content),
            None => parts
                .iter()
                .filter_map(Part::text)
                .collect::<Vec<_>>()
                .join("\n\n"),
        };
        let tool_calls = fields
            .tool_calls
            .clone()
            .unwrap_or_else(|| parts.iter().filter_map(Part::tool_call).collect());
        Message {
            name: (role == "tool")
                .then_some(fields.name.clone())
                .flatten()
                .filter(|name| !name.is_null() && name != &Value::String(String::new())),
            role,
            content,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        }
    }
}

impl Part {
    fn text(&self) -> Option<String> {
        match self.kind.as_deref().unwrap_or("text") {
            "text" => self
                .text
                .clone()
                .or_else(|| self.content.as_ref().map(display_value)),
            "tool_call_response" => self.response.as_ref().map(display_value),
            _ => None,
        }
    }

    fn tool_call(&self) -> Option<ToolCall> {
        (self.kind.as_deref() == Some("tool_call")).then(|| {
            ToolCall(IndexMap::from([
                (
                    "name".to_owned(),
                    Value::from(self.name.clone().unwrap_or_default()),
                ),
                (
                    "arguments".to_owned(),
                    self.arguments.clone().unwrap_or(Value::Null),
                ),
                ("id".to_owned(), self.id.clone().unwrap_or(Value::Null)),
            ]))
        })
    }
}

fn display_value(value: &Value) -> String {
    value.as_str().map_or_else(|| encode(value), str::to_owned)
}

/// The conversation `value` holds: an array of messages or a single message.
pub(super) fn parse(value: &Value) -> Option<Vec<Message>> {
    let raw = MessagePayload::<RawMessage>::deserialize(value)
        .ok()?
        .into_messages();
    (!raw.is_empty() && raw.iter().all(RawMessage::is_message))
        .then(|| raw.iter().map(RawMessage::normalized).collect())
}

/// OpenInference's flattened `<prefix>.<i>.message.{role,content,contents,tool_calls}` attributes.
pub(super) fn flattened(
    attributes: &BTreeMap<String, String>,
    prefix: &str,
) -> Option<Vec<Message>> {
    let messages: Vec<Message> = (0..)
        .map(|index| format!("{prefix}.{index}.message."))
        .take_while(|message| {
            attributes
                .keys()
                .any(|key| key.starts_with(message.as_str()))
        })
        .map(|message| {
            let field = |name: &str| attr(attributes, &format!("{message}{name}")).to_owned();
            let content = if field("content").is_empty() {
                (0..)
                    .map(|part| field(&format!("contents.{part}.message_content.text")))
                    .take_while(|text| !text.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n")
            } else {
                field("content")
            };
            let tool_calls: Vec<ToolCall> = (0..)
                .map(|call| format!("tool_calls.{call}.tool_call."))
                .take_while(|call| !field(&format!("{call}function.name")).is_empty())
                .map(|call| {
                    ToolCall(IndexMap::from([
                        (
                            "name".to_owned(),
                            Value::from(field(&format!("{call}function.name"))),
                        ),
                        (
                            "arguments".to_owned(),
                            Value::from(field(&format!("{call}function.arguments"))),
                        ),
                        ("id".to_owned(), Value::from(field(&format!("{call}id")))),
                    ]))
                })
                .collect();
            Message {
                role: field("role"),
                content,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                name: Some(field("name"))
                    .filter(|name| !name.is_empty())
                    .map(Value::from),
            }
        })
        .collect();
    (!messages.is_empty()).then_some(messages)
}

pub(super) fn indexed(attributes: &BTreeMap<String, String>, prefix: &str) -> Option<String> {
    let indices: BTreeSet<usize> = attributes
        .keys()
        .filter_map(|key| {
            key.strip_prefix(prefix)?
                .strip_prefix('.')?
                .split('.')
                .next()?
                .parse()
                .ok()
        })
        .collect();
    let values: Vec<Value> = indices
        .into_iter()
        .filter_map(|index| {
            let base = format!("{prefix}.{index}.");
            let fields = Value::Object(
                attributes
                    .range(base.clone()..)
                    .take_while(|(key, _)| key.starts_with(&base))
                    .filter_map(|(key, value)| {
                        let suffix = key.strip_prefix(&base)?;
                        Some((
                            suffix.strip_prefix("message.").unwrap_or(suffix).to_owned(),
                            Value::from(value.clone()),
                        ))
                    })
                    .collect(),
            );
            let message = EventFields::deserialize(&fields).ok()?;
            let calls = event_tool_calls(&message);
            if message.content.is_none() && calls.is_none() {
                return None;
            }
            Some(serde_json::json!({
                "role": message.role?,
                "content": message.content.unwrap_or(Value::from("")),
                "tool_calls": calls,
            }))
        })
        .collect();
    (!values.is_empty()).then(|| canonical(&encode(&values)))
}

fn event_tool_calls(value: &EventFields) -> Option<Value> {
    if let Some(calls) = &value.tool_calls {
        return Some(calls.clone());
    }
    let indices: BTreeSet<usize> = value
        .indexed
        .keys()
        .filter_map(|key| {
            key.strip_prefix("tool_calls.")?
                .split('.')
                .next()?
                .parse()
                .ok()
        })
        .collect();
    let calls: Vec<Value> = indices
        .into_iter()
        .filter_map(|index| {
            let prefix = format!("tool_calls.{index}");
            Some(serde_json::json!({
                "id": value.indexed.get(&format!("{prefix}.id")),
                "name": value.indexed.get(&format!("{prefix}.function.name"))?,
                "arguments": value.indexed.get(&format!("{prefix}.function.arguments")),
            }))
        })
        .collect();
    (!calls.is_empty()).then_some(Value::Array(calls))
}

pub(super) fn event_message(name: &str, value: &Value) -> Option<(bool, Value)> {
    EventMessage::deserialize(value).ok()?.normalized(name)
}

pub(super) fn event_payload(events: &[(bool, Value)], output: bool) -> Option<String> {
    let values: Vec<&Value> = events
        .iter()
        .filter(|(direction, _)| *direction == output)
        .map(|(_, value)| value)
        .collect();
    (!values.is_empty()).then(|| canonical(&encode(&values)))
}

/// The latest user message with text.
pub(super) fn preview(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user" && !message.content.is_empty())
        .map_or("", |message| message.content.as_str())
        .chars()
        .take(PREVIEW_CHARS)
        .collect()
}

/// The latest user message when `input` is a conversation, else the input itself.
pub(super) fn input_preview(input: &str) -> String {
    match serde_json::from_str::<Value>(input)
        .ok()
        .and_then(|value| parse(&value))
    {
        Some(messages) => preview(&messages),
        None => input.chars().take(PREVIEW_CHARS).collect(),
    }
}

/// `raw` in the common format when it holds a conversation, else unchanged.
pub(super) fn canonical(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|value| parse(&value))
        .map_or_else(|| raw.to_owned(), |messages| encode(&messages))
}

#[derive(Deserialize)]
struct LlmOutput {
    id: Option<String>,
}

#[derive(Deserialize)]
struct Generation {
    message: RawMessage,
}

#[derive(Deserialize)]
struct LlmResult {
    generations: Vec<Recognized<Vec<Recognized<Generation>>>>,
    llm_output: Option<LlmOutput>,
}

/// A LangChain `LLMResult`'s first generation and the requests behind it.
pub(super) struct Generations {
    pub first: Option<Message>,
    pub calls: CallEvidence,
}

/// LangChain `LLMResult`: `generations[prompt][candidate]`. Each prompt is one provider request,
/// whose candidates share its response id (`response_metadata.id`; `llm_output.id` for a single
/// prompt). The evidence is complete only when every prompt yields exactly one id and no entry
/// failed to parse.
pub(super) fn langchain_result(value: &Value) -> Option<Generations> {
    let result = LlmResult::deserialize(value).ok()?;
    let mut complete = true;
    let mut first = None;
    let mut keys = BTreeSet::new();
    for prompt in &result.generations {
        let Recognized::Known(candidates) = prompt else {
            complete = false;
            continue;
        };
        let mut ids = BTreeSet::new();
        for candidate in candidates {
            match candidate {
                Recognized::Known(generation) => {
                    let message = generation.message.unwrapped();
                    if let Some(id) = message
                        .response_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.id.clone())
                    {
                        ids.insert(id);
                    }
                    if first.is_none() {
                        first = Some(generation.message.normalized());
                    }
                }
                Recognized::Unrecognized(_) => complete = false,
            }
        }
        if ids.is_empty()
            && result.generations.len() == 1
            && let Some(id) = result
                .llm_output
                .as_ref()
                .and_then(|output| output.id.clone())
        {
            ids.insert(id);
        }
        complete &= ids.len() == 1;
        keys.extend(ids.into_iter().map(CallKey::ProviderResponse));
    }
    let calls = match (keys.is_empty(), complete && !result.generations.is_empty()) {
        (true, _) => CallEvidence::Unknown,
        (false, true) => CallEvidence::Complete(keys),
        (false, false) => CallEvidence::Partial(keys),
    };
    Some(Generations { first, calls })
}

struct PythonJsonFormatter;

impl Formatter for PythonJsonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

pub(crate) fn encode<T: Serialize>(value: &T) -> String {
    let mut output = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut output, PythonJsonFormatter);
    if value.serialize(&mut serializer).is_err() {
        return String::new();
    }
    String::from_utf8(output).unwrap_or_default()
}

pub(super) fn state_preview(input: &str, key: &str) -> Option<String> {
    let object = serde_json::from_str::<serde_json::Map<String, Value>>(input).ok()?;
    let conversation = parse(object.get(key)?)?;
    Some(preview(&conversation))
}

pub(super) fn state_conversation(input: &str) -> Option<Vec<Message>> {
    let value: Value = serde_json::from_str(input).ok()?;
    let items = value.get("messages")?.as_array()?;
    if items.first().is_some_and(Value::is_array) {
        return None;
    }
    let conversation: Vec<Message> = items
        .iter()
        .filter_map(|item| RawMessage::deserialize(item).ok())
        .map(|message| message.normalized())
        .collect();
    (!conversation.is_empty()).then_some(conversation)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{state_conversation, state_preview};
    use serde_json::Value;

    #[rstest]
    #[case::latest_user(r#"{"messages":[{"role":"user","content":"first"},{"role":"assistant","content":"reply"},{"role":"user","content":"last"}]}"#, Some("last"))]
    #[case::malformed("not-json", None)]
    #[case::missing("{}", None)]
    #[case::not_messages(r#"{"messages":[{"role":"user"}]}"#, None)]
    fn state_preview_requires_a_valid_conversation(
        #[case] input: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(state_preview(input, "messages").as_deref(), expected);
    }

    #[rstest]
    #[case::lenient_flat(
        r#"{"messages":[null,{"type":"human","content":"hello"}]}"#,
        Some(r#"[{"role":"user","content":"hello"}]"#)
    )]
    #[case::nested(r#"{"messages":[[{"role":"user","content":"hello"}]]}"#, None)]
    #[case::empty(r#"{"messages":[]}"#, None)]
    fn state_conversation_preserves_flat_batch_semantics(
        #[case] input: &str,
        #[case] expected: Option<&str>,
    ) {
        let observed =
            state_conversation(input).map(|messages| serde_json::to_value(messages).unwrap());
        let expected_value = expected.map(|value| serde_json::from_str::<Value>(value).unwrap());
        assert_eq!(observed, expected_value);
    }
}
