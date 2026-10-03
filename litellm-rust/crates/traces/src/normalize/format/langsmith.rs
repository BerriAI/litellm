use std::collections::BTreeMap;

use serde::{
    Deserialize, Deserializer,
    de::{DeserializeOwned, IgnoredAny},
};
use serde_json::Value;

use super::{Extraction, Format, SpanFacts, genai::GenAi};
use crate::{
    Error,
    normalize::{
        CallEvidence, ObservationType, RoleEvidence, SpanContext, attr,
        messages::{RawMessage, encode, langchain_result},
    },
};

/// LangSmith's OpenTelemetry exporter: spans carry `langsmith.span.kind`.
pub(crate) struct LangSmith;

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
}

#[derive(Default, Deserialize)]
struct Payload {
    #[serde(default, deserialize_with = "lenient")]
    messages: Option<MessageBatch>,
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

#[derive(Deserialize)]
struct WrappedOutput {
    output: Value,
    #[serde(flatten)]
    _other: BTreeMap<String, IgnoredAny>,
}

struct SpanIo {
    input: String,
    output: String,
    calls: CallEvidence,
}

fn normalized_messages(messages: &[RawMessage]) -> String {
    encode(
        &messages
            .iter()
            .map(RawMessage::normalized)
            .collect::<Vec<_>>(),
    )
}

fn tool_output(raw_completion: &str) -> String {
    let completion = serde_json::from_str::<Value>(raw_completion).unwrap_or(Value::Null);
    let raw = WrappedOutput::deserialize(&completion)
        .map(|wrapped| wrapped.output)
        .unwrap_or(completion);
    let selected = Command::deserialize(&raw)
        .ok()
        .and_then(|command| command.update.messages.into_iter().last())
        .unwrap_or(raw);
    let output = ContentValue::deserialize(&selected)
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
    if kind == ObservationType::Llm
        && serde_json::from_str::<Value>(raw_completion).is_ok_and(|value| value.is_object())
    {
        let input = prompt.messages.as_ref().map_or_else(
            || "[]".to_owned(),
            |messages| normalized_messages(messages.first_batch()),
        );
        let result = serde_json::from_str::<Value>(raw_completion)
            .ok()
            .and_then(|value| langchain_result(&value));
        return match result {
            Some(result) if result.first.is_some() => SpanIo {
                input,
                output: result.first.as_ref().map(encode).unwrap_or_default(),
                calls: result.calls,
            },
            _ => SpanIo {
                input,
                output: raw_completion.to_owned(),
                calls: result.map_or(CallEvidence::Unknown, |result| result.calls),
            },
        };
    }
    if kind == ObservationType::Tool {
        return SpanIo {
            input: raw_prompt.to_owned(),
            output: tool_output(raw_completion),
            calls: CallEvidence::Unknown,
        };
    }

    SpanIo {
        input: raw_prompt.to_owned(),
        output: raw_completion.to_owned(),
        calls: CallEvidence::Unknown,
    }
}

impl Format for LangSmith {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        context.scope == "langsmith" || context.attributes.contains_key("langsmith.span.kind")
    }

    fn extract(&self, context: &SpanContext<'_>) -> Result<Extraction, Error> {
        let attributes = context.attributes;
        let base = GenAi.extract(context)?;
        let observation_type = ObservationType::try_from(attr(attributes, "langsmith.span.kind"))
            .unwrap_or(ObservationType::Chain);
        let io = span_io(observation_type, attributes);
        Ok(Extraction {
            facts: SpanFacts {
                role: Some(RoleEvidence::Declared(observation_type)),
                input: if attr(attributes, "gen_ai.prompt").is_empty() {
                    String::new()
                } else {
                    io.input
                },
                output: if attr(attributes, "gen_ai.completion").is_empty() {
                    String::new()
                } else {
                    io.output
                },
                calls: io.calls,
                ..SpanFacts::default()
            }
            .or(base.facts),
            display_name: None,
            consumed_attributes: base.consumed_attributes,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{CallEvidence, ObservationType, span_io};
    use crate::normalize::CallKey;

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
        assert_eq!(
            io.calls,
            CallEvidence::complete(CallKey::ProviderResponse("response-1".to_owned()))
        );
    }

    #[rstest]
    #[case::null(r#"{"output":null}"#, Value::Null)]
    #[case::string(r#""answer""#, json!("answer"))]
    #[case::wrapped_string(r#"{"output":"answer","other":7}"#, json!("answer"))]
    #[case::repeated_output(r#"{"output":"first","output":"last"}"#, json!("last"))]
    #[case::wrapped_content(r#"{"output":{"content":"answer"}}"#, json!("answer"))]
    #[case::last_command_message(r#"{"output":{"update":{"messages":[{"content":"first"},{"content":"last"}]}}}"#, json!("last"))]
    #[case::direct_command(r#"{"update":{"messages":[{"content":"answer"}]}}"#, json!("answer"))]
    #[case::empty_command(r#"{"update":{"messages":[]}}"#, json!({"update":{"messages":[]}}))]
    #[case::arbitrary_object(r#"{"result":7}"#, json!({"result":7}))]
    #[case::arbitrary_array(r#"[1,2]"#, json!([1,2]))]
    #[case::malformed("not-json", Value::Null)]
    fn tool_outputs_preserve_content_and_fallbacks(
        #[case] completion: &str,
        #[case] expected: Value,
    ) {
        let attributes = BTreeMap::from([("gen_ai.completion".to_owned(), completion.to_owned())]);
        let io = span_io(ObservationType::Tool, &attributes);
        match expected {
            Value::String(text) => assert_eq!(io.output, text),
            value => assert_eq!(serde_json::from_str::<Value>(&io.output).unwrap(), value),
        }
    }

    #[rstest]
    fn absent_llm_messages_render_as_an_empty_list() {
        let attributes = BTreeMap::from([("gen_ai.completion".to_owned(), "{}".to_owned())]);
        let io = span_io(ObservationType::Llm, &attributes);
        assert_eq!(io.input, "[]");
    }
}
