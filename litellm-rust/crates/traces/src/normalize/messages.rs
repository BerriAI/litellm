//! The common message format normalizers emit for span input and output: a JSON array of
//! `{role, content, tool_calls?, name?}` that the UI renders as a conversation.

use std::{
    collections::{BTreeMap, BTreeSet},
    io,
};

use indexmap::IndexMap;
use litellm_llms_types::{formats::chat_completions::ChatMessageContent, recognized::Recognized};
use serde::{Deserialize, Serialize};
use serde_json::{Value, ser::Formatter};

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
    let raw: Vec<RawMessage> = match value {
        Value::Array(items) => items
            .iter()
            .map(|item| serde_json::from_value(item.clone()).ok())
            .collect::<Option<_>>()?,
        Value::Object(_) => vec![serde_json::from_value(value.clone()).ok()?],
        _ => return None,
    };
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
    generations: Vec<Value>,
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
    let result: LlmResult = serde_json::from_value(value.clone()).ok()?;
    let mut complete = true;
    let mut first = None;
    let mut keys = BTreeSet::new();
    for prompt in &result.generations {
        let Some(candidates) = prompt.as_array() else {
            complete = false;
            continue;
        };
        let mut ids = BTreeSet::new();
        for candidate in candidates {
            match serde_json::from_value::<Generation>(candidate.clone()) {
                Ok(generation) => {
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
                Err(_) => complete = false,
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
