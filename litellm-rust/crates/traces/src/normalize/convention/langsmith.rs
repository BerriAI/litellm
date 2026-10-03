use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, de::DeserializeOwned};
use serde_json::Value;

use super::{Convention, Extraction, SpanFacts};
use crate::{
    Error,
    normalize::{
        CallEvidence, ObservationType, RoleEvidence, SpanContext, attr,
        messages::{RawMessage, encode, langchain_result},
        present, usage_tokens,
    },
};

/// LangSmith's OpenTelemetry exporter: spans carry `langsmith.span.kind`.
pub(crate) struct LangSmith;

const MIDDLEWARE_SUFFIXES: [&str; 6] = [
    ".wrap_model_call",
    ".wrap_tool_call",
    ".before_agent",
    ".after_agent",
    ".before_model",
    ".after_model",
];

/// LangChain agent middleware hooks, which run around the agent's steps rather than being one.
pub(crate) fn is_langchain_middleware(name: &str) -> bool {
    MIDDLEWARE_SUFFIXES
        .iter()
        .any(|suffix| name.ends_with(suffix))
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

fn span_type(
    name: &str,
    parent_span_id: &str,
    attributes: &BTreeMap<String, String>,
) -> ObservationType {
    match ObservationType::try_from(attr(attributes, "langsmith.span.kind")) {
        Ok(kind) if kind != ObservationType::Chain => kind,
        _ if parent_span_id.is_empty()
            || name == attr(attributes, "langsmith.metadata.lc_agent_name") =>
        {
            ObservationType::Agent
        }
        _ if is_langchain_middleware(name) => ObservationType::Framework,
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
            calls: CallEvidence::Unknown,
        };
    }
    SpanIo {
        input: raw_prompt.to_owned(),
        output: raw_completion.to_owned(),
        calls: CallEvidence::Unknown,
    }
}

impl Convention for LangSmith {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        context.scope == "langsmith" || context.attributes.contains_key("langsmith.span.kind")
    }

    fn extract(&self, context: &SpanContext<'_>) -> Result<Extraction, Error> {
        let attributes = context.attributes;
        let (input_tokens, output_tokens) = usage_tokens(attributes)?;
        let observation_type = span_type(context.name, context.parent_span_id, attributes);
        let io = span_io(observation_type, attributes);
        Ok(Extraction {
            facts: SpanFacts {
                role: Some(RoleEvidence::Declared(observation_type)),
                agent_name: present(attributes, &["langsmith.metadata.lc_agent_name"]),
                model: present(attributes, &["gen_ai.request.model"]),
                input_tokens,
                output_tokens,
                input: io.input,
                output: io.output,
                calls: io.calls,
                ..SpanFacts::default()
            },
            display_name: None,
            consumed_attributes: ["gen_ai.prompt", "gen_ai.completion"]
                .into_iter()
                .filter(|key| attributes.contains_key(*key))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;
    use serde_json::Value;

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
