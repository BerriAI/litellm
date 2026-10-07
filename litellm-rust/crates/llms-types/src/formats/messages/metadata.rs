use serde_json::{Map, Value};

use crate::json_schema::JsonSchema;

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MessagesMetadata {
    pub user_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct OutputFormat {
    #[serde(rename = "type")]
    pub format_type: OutputFormatType,
    pub schema: Option<JsonSchema>,
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormatType {
    JsonSchema,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct MessagesCompaction {
    #[serde(rename = "type")]
    pub compaction_type: CompactionType,
    pub instructions: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum CompactionType {
    Summarize,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MessagesContainer {
    pub id: Option<String>,
    pub expires_at: Option<String>,
    pub skills: Option<Vec<ContainerSkill>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct ContainerSkill {
    #[serde(rename = "type")]
    pub skill_type: SkillType,
    pub skill_id: Option<String>,
    pub version: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum SkillType {
    Anthropic,
    Custom,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct McpServer {
    #[serde(rename = "type")]
    pub server_type: McpServerType,
    pub url: Option<String>,
    pub name: Option<String>,
    pub authorization_token: Option<String>,
    pub tool_configuration: Option<McpToolConfiguration>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum McpServerType {
    Url,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct McpToolConfiguration {
    pub allowed_tools: Option<Vec<String>>,
    pub enabled: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct StopDetails {
    #[serde(rename = "type")]
    pub detail_type: StopDetailsType,
    pub category: Option<String>,
    pub explanation: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum StopDetailsType {
    Refusal,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ContextManagementResponse {
    pub applied_edits: Option<Vec<AppliedEdit>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct AppliedEdit {
    #[serde(rename = "type")]
    pub edit_type: Option<String>,
    pub cleared_input_tokens: Option<u64>,
    pub cleared_tool_uses: Option<u64>,
    pub cleared_thinking_turns: Option<u64>,
    pub summary_input_tokens: Option<u64>,
    pub summary_output_tokens: Option<u64>,
    pub error: Option<String>,
    pub warnings: Option<Vec<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct Safeguard {
    #[serde(rename = "type")]
    pub safeguard_type: String,
    pub classifier_context: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    strum::AsRefStr,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
#[strum(serialize_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    #[strum(default, transparent)]
    Other(String),
}

impl MessageRole {
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl From<String> for MessageRole {
    fn from(value: String) -> Self {
        value.parse().unwrap_or_else(|never| match never {})
    }
}

impl From<MessageRole> for String {
    fn from(value: MessageRole) -> Self {
        value.to_string()
    }
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    strum::AsRefStr,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
#[strum(serialize_all = "snake_case")]
pub enum MessageType {
    Message,
    #[strum(default, transparent)]
    Other(String),
}

impl MessageType {
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl From<String> for MessageType {
    fn from(value: String) -> Self {
        value.parse().unwrap_or_else(|never| match never {})
    }
}

impl From<MessageType> for String {
    fn from(value: MessageType) -> Self {
        value.to_string()
    }
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    strum::AsRefStr,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
#[strum(serialize_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    Refusal,
    Compaction,
    PauseTurn,
    ModelContextWindowExceeded,
    #[strum(default, transparent)]
    Other(String),
}

impl StopReason {
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl From<String> for StopReason {
    fn from(value: String) -> Self {
        value.parse().unwrap_or_else(|never| match never {})
    }
}

impl From<StopReason> for String {
    fn from(value: StopReason) -> Self {
        value.to_string()
    }
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum ContainerReference {
    Id(String),
    Parameters(Box<MessagesContainer>),
}
