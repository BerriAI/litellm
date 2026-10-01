use std::{collections::BTreeMap, io};

use serde::{Deserialize, Serialize};
use serde_json::{Value, ser::Formatter};

use super::{NormalizedSpan, ObservationType, SpanNormalizer, attr};

pub(super) struct LangSmithNormalizer;

#[derive(Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
    Other(Value),
}

impl MessageContent {
    fn display_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    ContentBlock::Hidden(kind) => match kind {
                        HiddenBlock::Reasoning
                        | HiddenBlock::Thinking
                        | HiddenBlock::RedactedThinking
                        | HiddenBlock::FunctionCall
                        | HiddenBlock::ToolUse
                        | HiddenBlock::ToolCall => None,
                    },
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
            Self::Other(value) => encode(value),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ContentBlock {
    Text { text: String },
    Hidden(HiddenBlock),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum HiddenBlock {
    Reasoning,
    Thinking,
    RedactedThinking,
    FunctionCall,
    ToolUse,
    ToolCall,
}

#[derive(Deserialize, Serialize)]
struct RawToolCall {
    name: Option<Value>,
    args: Option<Value>,
}

#[derive(Deserialize)]
struct ResponseMetadata {
    id: Option<String>,
}

#[derive(Deserialize)]
struct RawMessage {
    kwargs: Option<Box<RawMessage>>,
    #[serde(rename = "type")]
    kind: Option<String>,
    role: Option<String>,
    content: Option<MessageContent>,
    tool_calls: Option<Vec<RawToolCall>>,
    name: Option<Value>,
    response_metadata: Option<ResponseMetadata>,
}

impl RawMessage {
    fn unwrapped(&self) -> &Self {
        self.kwargs.as_deref().unwrap_or(self)
    }

    fn normalized(&self) -> NormalizedMessage<'_> {
        let fields = self.unwrapped();
        let raw_role = fields
            .kind
            .as_deref()
            .filter(|role| !role.is_empty())
            .or_else(|| fields.role.as_deref().filter(|role| !role.is_empty()))
            .unwrap_or_default();
        let role = match raw_role {
            "human" => "user",
            "ai" => "assistant",
            other => other,
        };
        NormalizedMessage {
            role,
            content: fields
                .content
                .as_ref()
                .map_or_else(String::new, MessageContent::display_text),
            tool_calls: fields
                .tool_calls
                .as_deref()
                .filter(|calls| !calls.is_empty()),
            name: (role == "tool")
                .then_some(fields.name.as_ref())
                .flatten()
                .filter(|name| !name.is_null() && name != &&Value::String(String::new())),
        }
    }
}

#[derive(Serialize)]
struct NormalizedMessage<'a> {
    role: &'a str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a [RawToolCall]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MessageBatch {
    Flat(Vec<RawMessage>),
    Nested(Vec<Vec<RawMessage>>),
}

impl MessageBatch {
    fn first_batch(&self) -> &[RawMessage] {
        match self {
            Self::Flat(messages) => messages,
            Self::Nested(batches) => batches.first().map(Vec::as_slice).unwrap_or_default(),
        }
    }

    fn agent_messages(&self) -> &[RawMessage] {
        match self {
            Self::Flat(messages) => messages,
            Self::Nested(_) => &[],
        }
    }
}

#[derive(Deserialize)]
struct GenerationMessage {
    kwargs: Option<RawMessage>,
}

#[derive(Deserialize)]
struct Generation {
    message: Option<GenerationMessage>,
}

#[derive(Default, Deserialize)]
struct Payload {
    messages: Option<MessageBatch>,
    generations: Option<Vec<Vec<Generation>>>,
    output: Option<Value>,
}

#[derive(Deserialize)]
struct Command {
    update: CommandUpdate,
}

#[derive(Deserialize)]
struct CommandUpdate {
    messages: Vec<Value>,
}

#[derive(Deserialize)]
struct ContentValue {
    content: Value,
}

struct SpanIo {
    input: String,
    output: String,
    request_id: String,
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

fn encode<T: Serialize>(value: &T) -> String {
    let mut output = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut output, PythonJsonFormatter);
    if value.serialize(&mut serializer).is_err() {
        return String::new();
    }
    String::from_utf8(output).unwrap_or_default()
}

fn normalized_messages(messages: &[RawMessage]) -> String {
    encode(
        &messages
            .iter()
            .map(RawMessage::normalized)
            .collect::<Vec<_>>(),
    )
}

fn span_type(
    name: &str,
    parent_span_id: &str,
    attributes: &BTreeMap<String, String>,
) -> ObservationType {
    if parent_span_id.is_empty() || name == attr(attributes, "langsmith.metadata.lc_agent_name") {
        return ObservationType::Agent;
    }
    match attr(attributes, "langsmith.span.kind") {
        "llm" => ObservationType::Llm,
        "tool" => ObservationType::Tool,
        _ if [
            ".wrap_model_call",
            ".wrap_tool_call",
            ".before_agent",
            ".after_agent",
            ".before_model",
            ".after_model",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix)) =>
        {
            ObservationType::Framework
        }
        _ => ObservationType::Chain,
    }
}

fn tool_output(payload: Payload, raw_completion: &str) -> String {
    let completion = serde_json::from_str::<Value>(raw_completion).unwrap_or(Value::Null);
    let raw = payload.output.unwrap_or(completion);
    let selected = serde_json::from_value::<Command>(raw.clone())
        .ok()
        .and_then(|command| command.update.messages.into_iter().last())
        .unwrap_or(raw);
    let output = serde_json::from_value::<ContentValue>(selected.clone())
        .map(|message| message.content)
        .unwrap_or(selected);
    output
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| encode(&output))
}

fn span_io(kind: ObservationType, attributes: &BTreeMap<String, String>) -> SpanIo {
    let raw_prompt = attr(attributes, "gen_ai.prompt");
    let raw_completion = attr(attributes, "gen_ai.completion");
    let prompt = serde_json::from_str::<Payload>(raw_prompt).unwrap_or_default();
    let completion = serde_json::from_str::<Payload>(raw_completion).unwrap_or_default();
    if kind == ObservationType::Llm
        && serde_json::from_str::<Value>(raw_completion).is_ok_and(|value| value.is_object())
    {
        let input = prompt
            .messages
            .as_ref()
            .map_or_else(String::new, |messages| {
                normalized_messages(messages.first_batch())
            });
        let generation = completion
            .generations
            .as_ref()
            .and_then(|batches| batches.first())
            .and_then(|batch| batch.first())
            .and_then(|generation| generation.message.as_ref())
            .and_then(|message| message.kwargs.as_ref());
        if let Some(generation) = generation {
            let id = generation
                .response_metadata
                .as_ref()
                .and_then(|metadata| metadata.id.as_deref())
                .unwrap_or_default()
                .to_owned();
            return SpanIo {
                input,
                output: encode(&generation.normalized()),
                request_id: id,
            };
        }
        return SpanIo {
            input,
            output: raw_completion.to_owned(),
            request_id: String::new(),
        };
    }
    if kind == ObservationType::Tool {
        return SpanIo {
            input: raw_prompt.to_owned(),
            output: tool_output(completion, raw_completion),
            request_id: String::new(),
        };
    }
    if kind == ObservationType::Agent {
        let input = prompt
            .messages
            .as_ref()
            .filter(|messages| !messages.agent_messages().is_empty())
            .map_or_else(
                || raw_prompt.to_owned(),
                |messages| normalized_messages(messages.agent_messages()),
            );
        let output = completion
            .messages
            .as_ref()
            .and_then(|messages| messages.agent_messages().last())
            .map_or_else(
                || raw_completion.to_owned(),
                |message| encode(&message.normalized()),
            );
        return SpanIo {
            input,
            output,
            request_id: String::new(),
        };
    }
    SpanIo {
        input: raw_prompt.to_owned(),
        output: raw_completion.to_owned(),
        request_id: String::new(),
    }
}

impl SpanNormalizer for LangSmithNormalizer {
    fn matches(&self, scope_name: &str, attributes: &BTreeMap<String, String>) -> bool {
        scope_name == "langsmith" || attributes.contains_key("langsmith.span.kind")
    }

    fn normalize(
        &self,
        name: &str,
        parent_span_id: &str,
        attributes: &BTreeMap<String, String>,
    ) -> NormalizedSpan {
        let observation_type = span_type(name, parent_span_id, attributes);
        let io = span_io(observation_type, attributes);
        NormalizedSpan {
            observation_type,
            agent_name: attr(attributes, "langsmith.metadata.lc_agent_name").to_owned(),
            litellm_request_id: io.request_id,
            model: attr(attributes, "gen_ai.request.model").to_owned(),
            input_tokens: 0,
            output_tokens: 0,
            input: io.input,
            output: io.output,
        }
    }
}
