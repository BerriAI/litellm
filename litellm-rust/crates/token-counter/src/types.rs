use std::fmt;

use indexmap::IndexMap;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Number;

use super::Error;

/// The parts of a request body the host's budget counter reads. Chat and
/// Anthropic Messages bodies carry `messages`; completions carry `prompt`;
/// Responses and embeddings carry `input`; rerank carries `query` and
/// `documents`. The host checks key presence, not nullness, so an explicit
/// `null` is kept distinct from an absent key. Anything outside this shape is
/// declined so the host can fall back to its own counter instead of silently
/// miscounting.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CountableRequest {
    pub(crate) model: Option<String>,
    #[serde(default, deserialize_with = "present_messages")]
    pub(crate) messages: Option<Vec<Message>>,
    pub(crate) tools: Option<Vec<ToolDefinition>>,
    pub(crate) tool_choice: Option<ToolChoice>,
    #[serde(default, deserialize_with = "present_text")]
    pub(crate) prompt: Option<TextValue>,
    #[serde(default, deserialize_with = "present_text")]
    pub(crate) input: Option<TextValue>,
    #[serde(default, deserialize_with = "present_text")]
    pub(crate) query: Option<TextValue>,
    #[serde(default, deserialize_with = "present_text")]
    pub(crate) documents: Option<TextValue>,
}

impl CountableRequest {
    pub fn parse(body: &[u8]) -> Result<Self, Error> {
        serde_json::from_slice(body).map_err(Error::RequestParse)
    }
}

fn present_messages<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<Message>>, D::Error> {
    Option::<Vec<Message>>::deserialize(deserializer)
        .map(|messages| Some(messages.unwrap_or_default()))
}

fn present_text<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<TextValue>, D::Error> {
    TextValue::deserialize(deserializer).map(Some)
}

/// Free-form JSON the host counts as text: strings and integers via `str()`,
/// objects via `json.dumps()`, lists flattened. Objects keep document order so
/// the dumped text matches Python byte for byte.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum TextValue {
    Null,
    Bool(bool),
    Number(Number),
    Text(String),
    List(Vec<TextValue>),
    Object(IndexMap<String, TextValue>),
}

impl<'de> Deserialize<'de> for TextValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(TextValueVisitor)
    }
}

struct TextValueVisitor;

impl<'de> Visitor<'de> for TextValueVisitor {
    type Value = TextValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(TextValue::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(TextValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(TextValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(TextValue::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(TextValue::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Number::from_f64(value)
            .map(TextValue::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(TextValue::Text(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(TextValue::Text(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut items = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
        while let Some(item) = sequence.next_element()? {
            items.push(item);
        }
        Ok(TextValue::List(items))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut entries = IndexMap::with_capacity(map.size_hint().unwrap_or(0));
        while let Some((key, value)) = map.next_entry()? {
            entries.insert(key, value);
        }
        Ok(TextValue::Object(entries))
    }
}

/// The parts of an assistant tool call Python counts: only the `arguments`
/// string contributes (`_count_function_call_tokens` in
/// `litellm_core_utils/token_counter.py`). An absent arguments key counts as the
/// empty string, while an explicit null counts as the string "None" the way
/// Python's `str(None)` does; any other non-string `arguments` declines so
/// Python handles the fallback instead of miscounting.
///
/// `deny_unknown_fields` is deliberately absent from this shape, and from
/// `ToolCallFunction` and `LegacyFunctionCall`: Python reads only
/// `function.arguments` off a tool call, so the extra `id`, `type` and `name`
/// keys the OpenAI shape carries are ignored rather than declined.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct ToolCall {
    pub(crate) function: ToolCallFunction,
}

/// Python runs `str(function.get("arguments", ""))`: a missing key counts as
/// empty, but an explicit null counts as the string "None". `serde(default)`
/// keeps a missing key at `None`, and the deserializer turns an explicit null
/// into "None" so the counter sees what Python counts.
fn explicit_null_arguments_as_none<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    match Option::<String>::deserialize(deserializer)? {
        Some(arguments) => Ok(Some(arguments)),
        None => Ok(Some(String::from("None"))),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct ToolCallFunction {
    #[serde(default, deserialize_with = "explicit_null_arguments_as_none")]
    pub(crate) arguments: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct LegacyFunctionCall {
    #[serde(default, deserialize_with = "explicit_null_arguments_as_none")]
    pub(crate) arguments: Option<String>,
}

/// Python counts every string-valued key of a message, so any key beyond these
/// makes the shape unsupported rather than silently uncounted.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Message {
    pub(crate) role: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) content: Option<MessageContent>,
    pub(crate) tool_call_id: Option<String>,
    pub(crate) tool_calls: Option<Vec<ToolCall>>,
    pub(crate) function_call: Option<LegacyFunctionCall>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum MessageContent {
    Text(String),
    Blocks(Vec<ContentItem>),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum ContentItem {
    Text(String),
    Block(ContentBlock),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub(crate) enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(rename = "tool_reference")]
    ToolReference { tool_name: Option<String> },
    /// Images, documents, files and tool use/result blocks price through
    /// Python-only helpers, so they stay on the Python counter.
    #[serde(other)]
    Unsupported,
}

/// Either the OpenAI `{"type": "function", "function": {...}}` shape or the
/// Anthropic `{"name", "description", "input_schema"}` shape.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct ToolDefinition {
    pub(crate) function: Option<FunctionDefinition>,
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) input_schema: Option<Schema>,
    pub(crate) parameters: Option<Schema>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct FunctionDefinition {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) parameters: Option<Schema>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct Schema {
    #[serde(rename = "type")]
    pub(crate) schema_type: Option<SchemaType>,
    pub(crate) description: Option<String>,
    #[serde(rename = "enum")]
    pub(crate) enum_values: Option<Vec<EnumValue>>,
    pub(crate) items: Option<Box<Schema>>,
    pub(crate) properties: Option<IndexMap<String, Schema>>,
    pub(crate) required: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum SchemaType {
    Name(String),
    Union(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum EnumValue {
    Text(String),
    Integer(i64),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum ToolChoice {
    Mode(String),
    Named(NamedToolChoice),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct NamedToolChoice {
    pub(crate) function: NamedFunction,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct NamedFunction {
    pub(crate) name: String,
}
