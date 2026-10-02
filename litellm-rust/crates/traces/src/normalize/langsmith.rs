use std::{collections::BTreeMap, io};

use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize, de::DeserializeOwned};
use serde_json::{Value, ser::Formatter};

use super::{NormalizedSpan, ObservationType, SpanNormalizer, attr, usage_tokens};
use crate::{Error, otlp::DecodedEvent};

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
#[serde(transparent)]
struct RawToolCall(IndexMap<String, Value>);

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

enum MessageBatch {
    Flat(Vec<RawMessage>),
    Nested(Vec<Vec<RawMessage>>),
}

impl<'de> Deserialize<'de> for MessageBatch {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let Value::Array(items) = value else {
            return Err(serde::de::Error::custom("messages must be an array"));
        };
        let parse = |items: Vec<Value>| {
            items
                .into_iter()
                .filter_map(|item| serde_json::from_value(item).ok())
                .collect()
        };
        Ok(if items.first().is_some_and(Value::is_array) {
            Self::Nested(
                items
                    .into_iter()
                    .filter_map(|item| item.as_array().cloned())
                    .map(parse)
                    .collect(),
            )
        } else {
            Self::Flat(parse(items))
        })
    }
}

fn lenient<'de, D: Deserializer<'de>, T: DeserializeOwned>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    Ok(serde_json::from_value(value).ok())
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
    #[serde(default, deserialize_with = "lenient")]
    messages: Option<MessageBatch>,
    #[serde(default, deserialize_with = "lenient")]
    generations: Option<Vec<Vec<Generation>>>,
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
    match attr(attributes, "langsmith.span.kind") {
        "llm" => ObservationType::Llm,
        "tool" => ObservationType::Tool,
        _ if parent_span_id.is_empty()
            || name == attr(attributes, "langsmith.metadata.lc_agent_name") =>
        {
            ObservationType::Agent
        }
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

fn tool_output(raw_completion: &str) -> String {
    let completion = serde_json::from_str::<Value>(raw_completion).unwrap_or(Value::Null);
    let raw = completion.get("output").cloned().unwrap_or(completion);
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
        let input = prompt.messages.as_ref().map_or_else(
            || "[]".to_owned(),
            |messages| normalized_messages(messages.first_batch()),
        );
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
            output: tool_output(raw_completion),
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

    fn consumed_attributes(&self, _attributes: &BTreeMap<String, String>) -> [&'static str; 2] {
        ["gen_ai.prompt", "gen_ai.completion"]
    }

    fn normalize(
        &self,
        name: &str,
        parent_span_id: &str,
        attributes: &BTreeMap<String, String>,
        _events: &[DecodedEvent],
    ) -> Result<NormalizedSpan, Error> {
        let (input_tokens, output_tokens) = usage_tokens(attributes)?;
        let observation_type = span_type(name, parent_span_id, attributes);
        let io = span_io(observation_type, attributes);
        Ok(NormalizedSpan {
            observation_type,
            agent_name: attr(attributes, "langsmith.metadata.lc_agent_name").to_owned(),
            framework: String::new(),
            litellm_request_id: io.request_id,
            model: attr(attributes, "gen_ai.request.model").to_owned(),
            input_tokens,
            output_tokens,
            input: io.input,
            output: io.output,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::Value;

    use super::{ObservationType, span_io};

    #[rstest]
    fn malformed_messages_preserve_valid_input_and_response_id() {
        let attributes = BTreeMap::from([
            (
                "gen_ai.prompt".to_owned(),
                r#"{"messages":[[{"kwargs":{"type":"human","content":"hello"}},null]]}"#.to_owned(),
            ),
            (
                "gen_ai.completion".to_owned(),
                r#"{"messages":"unexpected","generations":[[{"message":{"kwargs":{"type":"ai","content":"hi","response_metadata":{"id":"response-1"}}}}]]}"#.to_owned(),
            ),
        ]);
        let io = span_io(ObservationType::Llm, &attributes);
        let input: Value = serde_json::from_str(&io.input).expect("normalized input");
        assert_eq!(input.as_array().expect("messages").len(), 1);
        assert_eq!(input[0]["content"], "hello");
        assert_eq!(io.request_id, "response-1");
    }

    #[rstest]
    fn explicit_null_tool_output_is_preserved() {
        let attributes = BTreeMap::from([(
            "gen_ai.completion".to_owned(),
            r#"{"output":null}"#.to_owned(),
        )]);
        let io = span_io(ObservationType::Tool, &attributes);
        assert_eq!(io.output, "null");
    }

    #[rstest]
    fn absent_llm_messages_render_as_an_empty_list() {
        let attributes = BTreeMap::from([("gen_ai.completion".to_owned(), "{}".to_owned())]);
        let io = span_io(ObservationType::Llm, &attributes);
        assert_eq!(io.input, "[]");
    }
}
