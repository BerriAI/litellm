use serde_json::{Map, Value};

use super::CacheControl;
use crate::json_schema::JsonSchema;

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ToolDefinition {
    pub name: Option<String>,
    pub description: Option<String>,
    pub input_schema: Option<JsonSchema>,
    pub strict: Option<bool>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub allowed_callers: Option<Vec<String>>,
    pub input_examples: Option<Vec<Map<String, Value>>>,
    pub eager_input_streaming: Option<bool>,
    pub display_width_px: Option<u64>,
    pub display_height_px: Option<u64>,
    pub display_number: Option<u64>,
    pub max_uses: Option<u64>,
    pub max_tokens: Option<u64>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub citations: Option<super::CitationsConfig>,
    pub user_location: Option<WebSearchUserLocation>,
    pub model: Option<String>,
    pub caching: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct WebSearchUserLocation {
    #[serde(rename = "type")]
    pub location_type: UserLocationType,
    pub city: Option<String>,
    pub country: Option<String>,
    pub region: Option<String>,
    pub timezone: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum UserLocationType {
    Approximate,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct ToolChoice {
    #[serde(rename = "type")]
    pub choice_type: ToolChoiceType,
    pub name: Option<String>,
    pub disable_parallel_tool_use: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceType {
    Auto,
    Any,
    Tool,
    None,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct CustomTool {
    #[serde(rename = "type")]
    pub tool_type: Option<CustomToolType>,
    pub name: String,
    pub input_schema: JsonSchema,
    pub description: Option<String>,
    pub strict: Option<bool>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub allowed_callers: Option<Vec<String>>,
    pub input_examples: Option<Vec<Map<String, Value>>>,
    pub eager_input_streaming: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum CustomToolType {
    Custom,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type")]
pub enum BuiltinMessagesTool {
    #[serde(rename = "advisor_20260301")]
    Advisor(ToolDefinition),
    #[serde(rename = "tool_search_tool_regex_20251119")]
    ToolSearchRegex(ToolDefinition),
    #[serde(rename = "tool_search_tool_bm25_20251119")]
    ToolSearchBm25(ToolDefinition),
    #[serde(rename = "web_search_20250305")]
    WebSearch(ToolDefinition),
    #[serde(rename = "computer_20250124")]
    Computer(ToolDefinition),
    #[serde(rename = "bash_20250124")]
    Bash(ToolDefinition),
    #[serde(rename = "text_editor_20250728")]
    TextEditor(ToolDefinition),
    #[serde(rename = "code_execution_20250825")]
    CodeExecution(ToolDefinition),
    #[serde(rename = "web_search_20260209")]
    WebSearch20260209(ToolDefinition),
    #[serde(rename = "computer_20241022")]
    Computer20241022(ToolDefinition),
    #[serde(rename = "bash_20241022")]
    Bash20241022(ToolDefinition),
    #[serde(rename = "text_editor_20241022")]
    TextEditor20241022(ToolDefinition),
    #[serde(rename = "text_editor_20250124")]
    TextEditor20250124(ToolDefinition),
    #[serde(rename = "code_execution_20250522")]
    CodeExecution20250522(ToolDefinition),
    #[serde(rename = "memory_20250818")]
    Memory(ToolDefinition),
    #[serde(rename = "web_fetch_20250910")]
    WebFetch(ToolDefinition),
    #[serde(rename = "web_fetch_20260209")]
    WebFetch20260209(ToolDefinition),
    #[serde(rename = "web_fetch_20260309")]
    WebFetch20260309(ToolDefinition),
    #[serde(rename = "web_fetch_20260318")]
    WebFetch20260318(ToolDefinition),
    #[serde(rename = "web_search_20260318")]
    WebSearch20260318(ToolDefinition),
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120(ToolDefinition),
    #[serde(rename = "code_execution_20260521")]
    CodeExecution20260521(ToolDefinition),
    #[serde(rename = "computer_20251124")]
    Computer20251124(ToolDefinition),
    #[serde(rename = "text_editor_20250429")]
    TextEditor20250429(ToolDefinition),
    #[serde(rename = "tool_search_tool_regex")]
    ToolSearchRegexLatest(ToolDefinition),
    #[serde(rename = "tool_search_tool_bm25")]
    ToolSearchBm25Latest(ToolDefinition),
    #[serde(rename = "browser_toolset_20260801")]
    BrowserToolset(ToolDefinition),
    #[serde(rename = "computer_toolset_20260801")]
    ComputerToolset(ToolDefinition),
    #[serde(rename = "mcp_toolset")]
    McpToolset(ToolDefinition),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum MessagesToolParam {
    Builtin(Box<BuiltinMessagesTool>),
    Custom(Box<CustomTool>),
}
