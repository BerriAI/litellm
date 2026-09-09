use indexmap::IndexMap;
use serde::Deserialize;

use super::TokenCountError;

/// The parts of a request body `litellm.token_counter` reads when a host counts
/// input tokens for budget checks. Anything outside this shape is declined so
/// the host can fall back to its own counter instead of silently miscounting.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CountableRequest {
    pub model: String,
    pub messages: Option<Vec<Message>>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<ToolChoice>,
}

impl CountableRequest {
    pub fn parse(body: &[u8]) -> Result<Self, TokenCountError> {
        serde_json::from_slice(body)
            .map_err(|error| TokenCountError::Unsupported(error.to_string()))
    }
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
