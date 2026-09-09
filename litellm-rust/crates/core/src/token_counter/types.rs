use indexmap::IndexMap;
use serde::{Deserialize, Deserializer};

use super::TokenCountError;

/// The parts of a request body the host's budget counter reads. Chat and
/// Anthropic Messages bodies carry `messages`; completions carry `prompt`;
/// Responses and embeddings carry `input`; rerank carries `query` and
/// `documents`. The host checks key presence, not nullness, so an explicit
/// `null` is kept distinct from an absent key. Anything outside this shape is
/// declined so the host can fall back to its own counter instead of silently
/// miscounting.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CountableRequest {
    pub model: Option<String>,
    #[serde(default, deserialize_with = "present_messages")]
    pub messages: Option<Vec<Message>>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, deserialize_with = "present_text")]
    pub prompt: Option<TextValue>,
    #[serde(default, deserialize_with = "present_text")]
    pub input: Option<TextValue>,
    #[serde(default, deserialize_with = "present_text")]
    pub query: Option<TextValue>,
    #[serde(default, deserialize_with = "present_text")]
    pub documents: Option<TextValue>,
}

impl CountableRequest {
    pub fn parse(body: &[u8]) -> Result<Self, TokenCountError> {
        serde_json::from_slice(body)
            .map_err(|error| TokenCountError::Unsupported(error.to_string()))
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
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TextValue {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    Text(String),
    List(Vec<TextValue>),
    Object(IndexMap<String, TextValue>),
}

/// Python counts every string-valued key of a message, so any key beyond these
/// makes the shape unsupported rather than silently uncounted.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: Option<String>,
    pub name: Option<String>,
    pub content: Option<MessageContent>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentItem>),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ContentItem {
    Text(String),
    Block(ContentBlock),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentBlock {
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
pub struct ToolDefinition {
    pub function: Option<FunctionDefinition>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub input_schema: Option<Schema>,
    pub parameters: Option<Schema>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct FunctionDefinition {
    pub name: Option<String>,
    pub description: Option<String>,
    pub parameters: Option<Schema>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Schema {
    #[serde(rename = "type")]
    pub schema_type: Option<SchemaType>,
    pub description: Option<String>,
    #[serde(rename = "enum")]
    pub enum_values: Option<Vec<EnumValue>>,
    pub items: Option<Box<Schema>>,
    pub properties: Option<IndexMap<String, Schema>>,
    pub required: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum SchemaType {
    Name(String),
    Union(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EnumValue {
    Text(String),
    Integer(i64),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Named(NamedToolChoice),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NamedToolChoice {
    pub function: NamedFunction,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NamedFunction {
    pub name: String,
}
