use indexmap::IndexMap;
use serde_json::{Map, Value};

use super::{CacheControl, CitationsConfig, McpListedTool};
use crate::json_schema::JsonSchema;

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
pub enum AllowedCaller {
    #[serde(rename = "direct")]
    Direct,
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825,
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120,
    #[serde(rename = "code_execution_20260521")]
    CodeExecution20260521,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum UserLocationType {
    Approximate,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ToolChoice {
    #[serde(rename = "type")]
    pub choice_type: ToolChoiceType,
    pub name: Option<String>,
    pub disable_parallel_tool_use: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceType {
    Auto,
    Any,
    Tool,
    None,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct CustomTool {
    #[serde(rename = "type")]
    pub tool_type: Option<CustomToolType>,
    pub name: String,
    pub input_schema: JsonSchema,
    pub description: Option<String>,
    pub strict: Option<bool>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub input_examples: Option<Vec<Map<String, Value>>>,
    pub eager_input_streaming: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum CustomToolType {
    Custom,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BashToolName {
    Bash,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StrReplaceEditorName {
    StrReplaceEditor,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StrReplaceBasedEditToolName {
    StrReplaceBasedEditTool,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryToolName {
    Memory,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComputerToolName {
    Computer,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CodeExecutionToolName {
    CodeExecution,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolSearchRegexToolName {
    ToolSearchToolRegex,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolSearchBm25ToolName {
    ToolSearchToolBm25,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchToolName {
    WebSearch,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebFetchToolName {
    WebFetch,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdvisorToolName {
    Advisor,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ServerTool<N> {
    pub name: N,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ClientTool<N> {
    pub name: N,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub input_examples: Option<Vec<Map<String, Value>>>,
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct TextEditorTool20250728 {
    pub name: StrReplaceBasedEditToolName,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub input_examples: Option<Vec<Map<String, Value>>>,
    pub max_characters: Option<u64>,
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ComputerTool {
    pub name: ComputerToolName,
    pub display_width_px: u64,
    pub display_height_px: u64,
    pub display_number: Option<u64>,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub input_examples: Option<Vec<Map<String, Value>>>,
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ComputerTool20251124 {
    pub name: ComputerToolName,
    pub display_width_px: u64,
    pub display_height_px: u64,
    pub display_number: Option<u64>,
    pub enable_zoom: Option<bool>,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub input_examples: Option<Vec<Map<String, Value>>>,
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseInclusion {
    Full,
    Excluded,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebSearchTool {
    pub name: WebSearchToolName,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub max_uses: Option<u64>,
    pub strict: Option<bool>,
    pub user_location: Option<WebSearchUserLocation>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebSearchTool20260318 {
    pub name: WebSearchToolName,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub cache_control: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub max_uses: Option<u64>,
    pub response_inclusion: Option<ResponseInclusion>,
    pub strict: Option<bool>,
    pub user_location: Option<WebSearchUserLocation>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UrlSourceToolReference {
    ToolReference {
        name: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultUrlSource {
    All {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    None {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Only {
        tools: Vec<UrlSourceToolReference>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Except {
        tools: Vec<UrlSourceToolReference>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserInputUrlSource {
    All {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    None {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebFetchUrlSources {
    pub client_tool_results: Option<ToolResultUrlSource>,
    pub server_tool_results: Option<ToolResultUrlSource>,
    pub user_input: Option<UserInputUrlSource>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebFetchTool {
    pub name: WebFetchToolName,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub cache_control: Option<CacheControl>,
    pub citations: Option<CitationsConfig>,
    pub defer_loading: Option<bool>,
    pub max_content_tokens: Option<u64>,
    pub max_uses: Option<u64>,
    pub strict: Option<bool>,
    pub url_sources: Option<WebFetchUrlSources>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebFetchTool20260309 {
    pub name: WebFetchToolName,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub cache_control: Option<CacheControl>,
    pub citations: Option<CitationsConfig>,
    pub defer_loading: Option<bool>,
    pub max_content_tokens: Option<u64>,
    pub max_uses: Option<u64>,
    pub strict: Option<bool>,
    pub url_sources: Option<WebFetchUrlSources>,
    pub use_cache: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebFetchTool20260318 {
    pub name: WebFetchToolName,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub cache_control: Option<CacheControl>,
    pub citations: Option<CitationsConfig>,
    pub defer_loading: Option<bool>,
    pub max_content_tokens: Option<u64>,
    pub max_uses: Option<u64>,
    pub response_inclusion: Option<ResponseInclusion>,
    pub strict: Option<bool>,
    pub url_sources: Option<WebFetchUrlSources>,
    pub use_cache: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct AdvisorTool {
    pub name: AdvisorToolName,
    pub model: String,
    pub allowed_callers: Option<Vec<AllowedCaller>>,
    pub cache_control: Option<CacheControl>,
    pub caching: Option<CacheControl>,
    pub defer_loading: Option<bool>,
    pub max_tokens: Option<u64>,
    pub max_uses: Option<u64>,
    pub strict: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ToolsetToolConfig {
    pub defer_loading: Option<bool>,
    pub enabled: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct McpToolset {
    pub mcp_server_name: String,
    pub cache_control: Option<CacheControl>,
    pub configs: Option<IndexMap<String, ToolsetToolConfig>>,
    pub default_config: Option<ToolsetToolConfig>,
    pub tools: Option<Vec<McpListedTool>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct BrowserToolsetConfigs {
    #[serde(rename = "type")]
    pub type_text: Option<ToolsetToolConfig>,
    pub close_tab: Option<ToolsetToolConfig>,
    pub double_click: Option<ToolsetToolConfig>,
    pub file_upload: Option<ToolsetToolConfig>,
    pub find: Option<ToolsetToolConfig>,
    pub form_input: Option<ToolsetToolConfig>,
    pub get_page_text: Option<ToolsetToolConfig>,
    pub hold_key: Option<ToolsetToolConfig>,
    pub hover: Option<ToolsetToolConfig>,
    pub javascript_exec: Option<ToolsetToolConfig>,
    pub key: Option<ToolsetToolConfig>,
    pub left_click: Option<ToolsetToolConfig>,
    pub left_click_drag: Option<ToolsetToolConfig>,
    pub left_mouse_down: Option<ToolsetToolConfig>,
    pub left_mouse_up: Option<ToolsetToolConfig>,
    pub list_tabs: Option<ToolsetToolConfig>,
    pub middle_click: Option<ToolsetToolConfig>,
    pub mouse_move: Option<ToolsetToolConfig>,
    pub navigate: Option<ToolsetToolConfig>,
    pub new_tab: Option<ToolsetToolConfig>,
    pub read_console: Option<ToolsetToolConfig>,
    pub read_network: Option<ToolsetToolConfig>,
    pub read_page: Option<ToolsetToolConfig>,
    pub right_click: Option<ToolsetToolConfig>,
    pub screenshot: Option<ToolsetToolConfig>,
    pub scroll: Option<ToolsetToolConfig>,
    pub scroll_to: Option<ToolsetToolConfig>,
    pub switch_tab: Option<ToolsetToolConfig>,
    pub triple_click: Option<ToolsetToolConfig>,
    pub wait: Option<ToolsetToolConfig>,
    pub zoom: Option<ToolsetToolConfig>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ComputerToolsetConfigs {
    #[serde(rename = "type")]
    pub type_text: Option<ToolsetToolConfig>,
    pub cursor_position: Option<ToolsetToolConfig>,
    pub double_click: Option<ToolsetToolConfig>,
    pub hold_key: Option<ToolsetToolConfig>,
    pub key: Option<ToolsetToolConfig>,
    pub left_click: Option<ToolsetToolConfig>,
    pub left_click_drag: Option<ToolsetToolConfig>,
    pub left_mouse_down: Option<ToolsetToolConfig>,
    pub left_mouse_up: Option<ToolsetToolConfig>,
    pub middle_click: Option<ToolsetToolConfig>,
    pub mouse_move: Option<ToolsetToolConfig>,
    pub right_click: Option<ToolsetToolConfig>,
    pub screenshot: Option<ToolsetToolConfig>,
    pub scroll: Option<ToolsetToolConfig>,
    pub triple_click: Option<ToolsetToolConfig>,
    pub wait: Option<ToolsetToolConfig>,
    pub zoom: Option<ToolsetToolConfig>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct Toolset<C> {
    pub cache_control: Option<CacheControl>,
    pub configs: Option<Box<C>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type")]
pub enum BuiltinMessagesTool {
    #[serde(rename = "bash_20241022")]
    Bash20241022(ClientTool<BashToolName>),
    #[serde(rename = "bash_20250124")]
    Bash20250124(ClientTool<BashToolName>),
    #[serde(rename = "text_editor_20241022")]
    TextEditor20241022(ClientTool<StrReplaceEditorName>),
    #[serde(rename = "text_editor_20250124")]
    TextEditor20250124(ClientTool<StrReplaceEditorName>),
    #[serde(rename = "text_editor_20250429")]
    TextEditor20250429(ClientTool<StrReplaceBasedEditToolName>),
    #[serde(rename = "text_editor_20250728")]
    TextEditor20250728(TextEditorTool20250728),
    #[serde(rename = "memory_20250818")]
    Memory20250818(ClientTool<MemoryToolName>),
    #[serde(rename = "computer_20241022")]
    Computer20241022(ComputerTool),
    #[serde(rename = "computer_20250124")]
    Computer20250124(ComputerTool),
    #[serde(rename = "computer_20251124")]
    Computer20251124(ComputerTool20251124),
    #[serde(rename = "code_execution_20250522")]
    CodeExecution20250522(ServerTool<CodeExecutionToolName>),
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825(ServerTool<CodeExecutionToolName>),
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120(ServerTool<CodeExecutionToolName>),
    #[serde(rename = "code_execution_20260521")]
    CodeExecution20260521(ServerTool<CodeExecutionToolName>),
    #[serde(rename = "tool_search_tool_regex_20251119")]
    ToolSearchRegex20251119(ServerTool<ToolSearchRegexToolName>),
    #[serde(rename = "tool_search_tool_regex")]
    ToolSearchRegex(ServerTool<ToolSearchRegexToolName>),
    #[serde(rename = "tool_search_tool_bm25_20251119")]
    ToolSearchBm2520251119(ServerTool<ToolSearchBm25ToolName>),
    #[serde(rename = "tool_search_tool_bm25")]
    ToolSearchBm25(ServerTool<ToolSearchBm25ToolName>),
    #[serde(rename = "web_search_20250305")]
    WebSearch20250305(WebSearchTool),
    #[serde(rename = "web_search_20260209")]
    WebSearch20260209(WebSearchTool),
    #[serde(rename = "web_search_20260318")]
    WebSearch20260318(WebSearchTool20260318),
    #[serde(rename = "web_fetch_20250910")]
    WebFetch20250910(WebFetchTool),
    #[serde(rename = "web_fetch_20260209")]
    WebFetch20260209(WebFetchTool),
    #[serde(rename = "web_fetch_20260309")]
    WebFetch20260309(WebFetchTool20260309),
    #[serde(rename = "web_fetch_20260318")]
    WebFetch20260318(WebFetchTool20260318),
    #[serde(rename = "advisor_20260301")]
    Advisor20260301(AdvisorTool),
    #[serde(rename = "browser_toolset_20260801")]
    BrowserToolset20260801(Toolset<BrowserToolsetConfigs>),
    #[serde(rename = "computer_toolset_20260801")]
    ComputerToolset20260801(Toolset<ComputerToolsetConfigs>),
    #[serde(rename = "mcp_toolset")]
    McpToolset(McpToolset),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum MessagesToolParam {
    Builtin(Box<BuiltinMessagesTool>),
    Custom(Box<CustomTool>),
}
