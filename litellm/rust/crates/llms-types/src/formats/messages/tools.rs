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

#[cfg(test)]
mod tests {
    use crate::formats::messages::{
        AdvisorTool, AdvisorToolName, CacheControl, UrlSourceToolReference,
    };
    use serde_json::Map;

    use crate::formats::messages::AllowedCaller;

    use crate::formats::messages::BashToolName;

    use crate::formats::messages::{BrowserToolsetConfigs, BuiltinMessagesTool};

    use crate::formats::messages::{CitationsConfig, ClientTool};

    use crate::formats::messages::CodeExecutionToolName;

    use crate::formats::messages::{
        ComputerTool, ComputerTool20251124, ComputerToolName, ComputerToolsetConfigs,
    };

    use crate::formats::messages::{CustomTool, CustomToolType};

    use crate::formats::messages::McpListedTool;

    use crate::formats::messages::{McpToolset, MemoryToolName};

    use crate::formats::messages::MessagesToolParam;

    use crate::formats::messages::ResponseInclusion;

    use crate::formats::messages::ServerTool;

    use crate::formats::messages::{StrReplaceBasedEditToolName, StrReplaceEditorName};

    use crate::formats::messages::TextEditorTool20250728;

    use crate::formats::messages::{
        ToolChoice, ToolChoiceType, ToolResultUrlSource, ToolSearchBm25ToolName,
    };

    use crate::formats::messages::ToolSearchRegexToolName;

    use crate::formats::messages::{Toolset, ToolsetToolConfig};

    use crate::formats::messages::{UserInputUrlSource, UserLocationType};

    use crate::formats::messages::{
        WebFetchTool, WebFetchTool20260309, WebFetchTool20260318, WebFetchToolName,
    };

    use crate::formats::messages::{
        WebFetchUrlSources, WebSearchTool, WebSearchTool20260318, WebSearchToolName,
    };

    use crate::{
        formats::messages::WebSearchUserLocation,
        json_schema::{JsonSchema, JsonSchemaObject, JsonSchemaType},
    };

    use crate::test_support::*;
    use indexmap::IndexMap;
    use rstest::rstest;

    use serde_json::{Value, json};

    #[rstest]
    fn tool_choice_round_trips() {
        let choice = round_trip::<ToolChoice>(
            json!({"type":"tool","name":"search","disable_parallel_tool_use":true}),
        );
        assert_eq!(choice.choice_type, ToolChoiceType::Tool);
        assert_eq!(choice.name.as_deref(), Some("search"));
        assert_eq!(choice.disable_parallel_tool_use, Some(true));
        assert!(choice.extra.is_empty());
    }

    fn client_tool<N>(name: N) -> ClientTool<N> {
        ClientTool {
            name,
            allowed_callers: None,
            cache_control: None,
            defer_loading: None,
            input_examples: None,
            strict: None,
            extra: Map::new(),
        }
    }

    fn server_tool<N>(name: N) -> ServerTool<N> {
        ServerTool {
            name,
            allowed_callers: None,
            cache_control: None,
            defer_loading: None,
            strict: None,
            extra: Map::new(),
        }
    }

    fn computer_tool(display_width_px: u64, display_height_px: u64) -> ComputerTool {
        ComputerTool {
            name: ComputerToolName::Computer,
            display_width_px,
            display_height_px,
            display_number: None,
            allowed_callers: None,
            cache_control: None,
            defer_loading: None,
            input_examples: None,
            strict: None,
            extra: Map::new(),
        }
    }

    fn web_search_tool() -> WebSearchTool {
        WebSearchTool {
            name: WebSearchToolName::WebSearch,
            allowed_callers: None,
            allowed_domains: None,
            blocked_domains: None,
            cache_control: None,
            defer_loading: None,
            max_uses: None,
            strict: None,
            user_location: None,
            extra: Map::new(),
        }
    }

    fn web_fetch_tool() -> WebFetchTool {
        WebFetchTool {
            name: WebFetchToolName::WebFetch,
            allowed_callers: None,
            allowed_domains: None,
            blocked_domains: None,
            cache_control: None,
            citations: None,
            defer_loading: None,
            max_content_tokens: None,
            max_uses: None,
            strict: None,
            url_sources: None,
            extra: Map::new(),
        }
    }

    fn web_fetch_url_sources() -> WebFetchUrlSources {
        WebFetchUrlSources {
            client_tool_results: None,
            server_tool_results: None,
            user_input: None,
            extra: Map::new(),
        }
    }

    fn toolset_config(enabled: Option<bool>, defer_loading: Option<bool>) -> ToolsetToolConfig {
        ToolsetToolConfig {
            defer_loading,
            enabled,
            extra: Map::new(),
        }
    }

    fn browser_toolset_configs() -> BrowserToolsetConfigs {
        BrowserToolsetConfigs {
            type_text: None,
            close_tab: None,
            double_click: None,
            file_upload: None,
            find: None,
            form_input: None,
            get_page_text: None,
            hold_key: None,
            hover: None,
            javascript_exec: None,
            key: None,
            left_click: None,
            left_click_drag: None,
            left_mouse_down: None,
            left_mouse_up: None,
            list_tabs: None,
            middle_click: None,
            mouse_move: None,
            navigate: None,
            new_tab: None,
            read_console: None,
            read_network: None,
            read_page: None,
            right_click: None,
            screenshot: None,
            scroll: None,
            scroll_to: None,
            switch_tab: None,
            triple_click: None,
            wait: None,
            zoom: None,
            extra: Map::new(),
        }
    }

    fn computer_toolset_configs() -> ComputerToolsetConfigs {
        ComputerToolsetConfigs {
            type_text: None,
            cursor_position: None,
            double_click: None,
            hold_key: None,
            key: None,
            left_click: None,
            left_click_drag: None,
            left_mouse_down: None,
            left_mouse_up: None,
            middle_click: None,
            mouse_move: None,
            right_click: None,
            screenshot: None,
            scroll: None,
            triple_click: None,
            wait: None,
            zoom: None,
            extra: Map::new(),
        }
    }

    #[rstest]
    #[case::bash_20241022(
        json!({"type":"bash_20241022","name":"bash","input_examples":[{"command":"ls"}]}),
        BuiltinMessagesTool::Bash20241022(ClientTool {
            input_examples: Some(vec![Map::from_iter([(String::from("command"), json!("ls"))])]),
            ..client_tool(BashToolName::Bash)
        })
    )]
    #[case::bash_20250124(
        json!({"type":"bash_20250124","name":"bash","allowed_callers":["direct","code_execution_20260521"]}),
        BuiltinMessagesTool::Bash20250124(ClientTool {
            allowed_callers: Some(vec![AllowedCaller::Direct, AllowedCaller::CodeExecution20260521]),
            ..client_tool(BashToolName::Bash)
        })
    )]
    #[case::text_editor_20241022(
        json!({"type":"text_editor_20241022","name":"str_replace_editor","defer_loading":true}),
        BuiltinMessagesTool::TextEditor20241022(ClientTool {
            defer_loading: Some(true),
            ..client_tool(StrReplaceEditorName::StrReplaceEditor)
        })
    )]
    #[case::text_editor_20250124(
        json!({"type":"text_editor_20250124","name":"str_replace_editor","strict":true}),
        BuiltinMessagesTool::TextEditor20250124(ClientTool {
            strict: Some(true),
            ..client_tool(StrReplaceEditorName::StrReplaceEditor)
        })
    )]
    #[case::text_editor_20250429(
        json!({"type":"text_editor_20250429","name":"str_replace_based_edit_tool","cache_control":{"type":"ephemeral","ttl":"1h"}}),
        BuiltinMessagesTool::TextEditor20250429(ClientTool {
            cache_control: Some(cache_control("ephemeral", "1h")),
            ..client_tool(StrReplaceBasedEditToolName::StrReplaceBasedEditTool)
        })
    )]
    #[case::text_editor_20250728(
        json!({"type":"text_editor_20250728","name":"str_replace_based_edit_tool","max_characters":10000}),
        BuiltinMessagesTool::TextEditor20250728(TextEditorTool20250728 {
            name: StrReplaceBasedEditToolName::StrReplaceBasedEditTool,
            allowed_callers: None,
            cache_control: None,
            defer_loading: None,
            input_examples: None,
            max_characters: Some(10000),
            strict: None,
            extra: Map::new(),
        })
    )]
    #[case::memory_20250818(
        json!({"type":"memory_20250818","name":"memory","allowed_callers":["code_execution_20250825"]}),
        BuiltinMessagesTool::Memory20250818(ClientTool {
            allowed_callers: Some(vec![AllowedCaller::CodeExecution20250825]),
            ..client_tool(MemoryToolName::Memory)
        })
    )]
    #[case::computer_20241022(
        json!({"type":"computer_20241022","name":"computer","display_width_px":1024,"display_height_px":768,"display_number":1}),
        BuiltinMessagesTool::Computer20241022(ComputerTool {
            display_number: Some(1),
            ..computer_tool(1024, 768)
        })
    )]
    #[case::computer_20250124(
        json!({"type":"computer_20250124","name":"computer","display_width_px":1280,"display_height_px":800}),
        BuiltinMessagesTool::Computer20250124(computer_tool(1280, 800))
    )]
    #[case::computer_20251124(
        json!({"type":"computer_20251124","name":"computer","display_width_px":1920,"display_height_px":1080,"enable_zoom":true}),
        BuiltinMessagesTool::Computer20251124(ComputerTool20251124 {
            name: ComputerToolName::Computer,
            display_width_px: 1920,
            display_height_px: 1080,
            display_number: None,
            enable_zoom: Some(true),
            allowed_callers: None,
            cache_control: None,
            defer_loading: None,
            input_examples: None,
            strict: None,
            extra: Map::new(),
        })
    )]
    #[case::code_execution_20250522(
        json!({"type":"code_execution_20250522","name":"code_execution","strict":false}),
        BuiltinMessagesTool::CodeExecution20250522(ServerTool {
            strict: Some(false),
            ..server_tool(CodeExecutionToolName::CodeExecution)
        })
    )]
    #[case::code_execution_20250825(
        json!({"type":"code_execution_20250825","name":"code_execution","defer_loading":false}),
        BuiltinMessagesTool::CodeExecution20250825(ServerTool {
            defer_loading: Some(false),
            ..server_tool(CodeExecutionToolName::CodeExecution)
        })
    )]
    #[case::code_execution_20260120(
        json!({"type":"code_execution_20260120","name":"code_execution","allowed_callers":["direct"]}),
        BuiltinMessagesTool::CodeExecution20260120(ServerTool {
            allowed_callers: Some(vec![AllowedCaller::Direct]),
            ..server_tool(CodeExecutionToolName::CodeExecution)
        })
    )]
    #[case::code_execution_20260521(
        json!({"type":"code_execution_20260521","name":"code_execution","allowed_callers":["code_execution_20260120"]}),
        BuiltinMessagesTool::CodeExecution20260521(ServerTool {
            allowed_callers: Some(vec![AllowedCaller::CodeExecution20260120]),
            ..server_tool(CodeExecutionToolName::CodeExecution)
        })
    )]
    #[case::tool_search_regex_20251119(
        json!({"type":"tool_search_tool_regex_20251119","name":"tool_search_tool_regex","defer_loading":true}),
        BuiltinMessagesTool::ToolSearchRegex20251119(ServerTool {
            defer_loading: Some(true),
            ..server_tool(ToolSearchRegexToolName::ToolSearchToolRegex)
        })
    )]
    #[case::tool_search_regex(
        json!({"type":"tool_search_tool_regex","name":"tool_search_tool_regex","strict":true}),
        BuiltinMessagesTool::ToolSearchRegex(ServerTool {
            strict: Some(true),
            ..server_tool(ToolSearchRegexToolName::ToolSearchToolRegex)
        })
    )]
    #[case::tool_search_bm25_20251119(
        json!({"type":"tool_search_tool_bm25_20251119","name":"tool_search_tool_bm25","defer_loading":false}),
        BuiltinMessagesTool::ToolSearchBm2520251119(ServerTool {
            defer_loading: Some(false),
            ..server_tool(ToolSearchBm25ToolName::ToolSearchToolBm25)
        })
    )]
    #[case::tool_search_bm25(
        json!({"type":"tool_search_tool_bm25","name":"tool_search_tool_bm25","strict":false}),
        BuiltinMessagesTool::ToolSearchBm25(ServerTool {
            strict: Some(false),
            ..server_tool(ToolSearchBm25ToolName::ToolSearchToolBm25)
        })
    )]
    #[case::web_search_20250305(
        json!({"type":"web_search_20250305","name":"web_search","max_uses":3,"allowed_domains":["example.test"],
            "user_location":{"type":"approximate","city":"San Francisco","country":"US","timezone":"America/Los_Angeles"}}),
        BuiltinMessagesTool::WebSearch20250305(WebSearchTool {
            max_uses: Some(3),
            allowed_domains: Some(vec![String::from("example.test")]),
            user_location: Some(WebSearchUserLocation {
                location_type: UserLocationType::Approximate,
                city: Some(String::from("San Francisco")),
                country: Some(String::from("US")),
                region: None,
                timezone: Some(String::from("America/Los_Angeles")),
                extra: Map::new(),
            }),
            ..web_search_tool()
        })
    )]
    #[case::web_search_20260209(
        json!({"type":"web_search_20260209","name":"web_search","blocked_domains":["blocked.test"]}),
        BuiltinMessagesTool::WebSearch20260209(WebSearchTool {
            blocked_domains: Some(vec![String::from("blocked.test")]),
            ..web_search_tool()
        })
    )]
    #[case::web_search_20260318(
        json!({"type":"web_search_20260318","name":"web_search","response_inclusion":"excluded"}),
        BuiltinMessagesTool::WebSearch20260318(WebSearchTool20260318 {
            name: WebSearchToolName::WebSearch,
            allowed_callers: None,
            allowed_domains: None,
            blocked_domains: None,
            cache_control: None,
            defer_loading: None,
            max_uses: None,
            response_inclusion: Some(ResponseInclusion::Excluded),
            strict: None,
            user_location: None,
            extra: Map::new(),
        })
    )]
    #[case::web_fetch_20250910(
        json!({"type":"web_fetch_20250910","name":"web_fetch","max_content_tokens":5000,"citations":{"enabled":true},
            "url_sources":{"client_tool_results":{"type":"only","tools":[{"type":"tool_reference","name":"lookup"}]},
                "server_tool_results":{"type":"all"},"user_input":{"type":"none"}}}),
        BuiltinMessagesTool::WebFetch20250910(WebFetchTool {
            max_content_tokens: Some(5000),
            citations: Some(CitationsConfig {
                enabled: Some(true),
                extra: Map::new(),
            }),
            url_sources: Some(WebFetchUrlSources {
                client_tool_results: Some(ToolResultUrlSource::Only {
                    tools: vec![tool_reference("lookup")],
                    extra: Map::new(),
                }),
                server_tool_results: Some(ToolResultUrlSource::All { extra: Map::new() }),
                user_input: Some(UserInputUrlSource::None { extra: Map::new() }),
                extra: Map::new(),
            }),
            ..web_fetch_tool()
        })
    )]
    #[case::web_fetch_20260209(
        json!({"type":"web_fetch_20260209","name":"web_fetch","url_sources":{"server_tool_results":{"type":"except","tools":[{"type":"tool_reference","name":"web_search"}]}}}),
        BuiltinMessagesTool::WebFetch20260209(WebFetchTool {
            url_sources: Some(WebFetchUrlSources {
                server_tool_results: Some(ToolResultUrlSource::Except {
                    tools: vec![tool_reference("web_search")],
                    extra: Map::new(),
                }),
                ..web_fetch_url_sources()
            }),
            ..web_fetch_tool()
        })
    )]
    #[case::web_fetch_20260309(
        json!({"type":"web_fetch_20260309","name":"web_fetch","use_cache":false}),
        BuiltinMessagesTool::WebFetch20260309(WebFetchTool20260309 {
            name: WebFetchToolName::WebFetch,
            allowed_callers: None,
            allowed_domains: None,
            blocked_domains: None,
            cache_control: None,
            citations: None,
            defer_loading: None,
            max_content_tokens: None,
            max_uses: None,
            strict: None,
            url_sources: None,
            use_cache: Some(false),
            extra: Map::new(),
        })
    )]
    #[case::web_fetch_20260318(
        json!({"type":"web_fetch_20260318","name":"web_fetch","use_cache":true,"response_inclusion":"full"}),
        BuiltinMessagesTool::WebFetch20260318(WebFetchTool20260318 {
            name: WebFetchToolName::WebFetch,
            allowed_callers: None,
            allowed_domains: None,
            blocked_domains: None,
            cache_control: None,
            citations: None,
            defer_loading: None,
            max_content_tokens: None,
            max_uses: None,
            response_inclusion: Some(ResponseInclusion::Full),
            strict: None,
            url_sources: None,
            use_cache: Some(true),
            extra: Map::new(),
        })
    )]
    #[case::advisor_20260301(
        json!({"type":"advisor_20260301","name":"advisor","model":"claude-opus-5-5","max_tokens":2048,"caching":{"type":"ephemeral","ttl":"5m"}}),
        BuiltinMessagesTool::Advisor20260301(AdvisorTool {
            name: AdvisorToolName::Advisor,
            model: String::from("claude-opus-5-5"),
            allowed_callers: None,
            cache_control: None,
            caching: Some(cache_control("ephemeral", "5m")),
            defer_loading: None,
            max_tokens: Some(2048),
            max_uses: None,
            strict: None,
            extra: Map::new(),
        })
    )]
    #[case::browser_toolset_20260801(
        json!({"type":"browser_toolset_20260801","configs":{"type":{"enabled":false},"javascript_exec":{"defer_loading":true}}}),
        BuiltinMessagesTool::BrowserToolset20260801(Toolset {
            cache_control: None,
            configs: Some(Box::new(BrowserToolsetConfigs {
                type_text: Some(toolset_config(Some(false), None)),
                javascript_exec: Some(toolset_config(None, Some(true))),
                ..browser_toolset_configs()
            })),
            extra: Map::new(),
        })
    )]
    #[case::computer_toolset_20260801(
        json!({"type":"computer_toolset_20260801","configs":{"zoom":{"enabled":false},"cursor_position":{"enabled":true}}}),
        BuiltinMessagesTool::ComputerToolset20260801(Toolset {
            cache_control: None,
            configs: Some(Box::new(ComputerToolsetConfigs {
                zoom: Some(toolset_config(Some(false), None)),
                cursor_position: Some(toolset_config(Some(true), None)),
                ..computer_toolset_configs()
            })),
            extra: Map::new(),
        })
    )]
    #[case::mcp_toolset(
        json!({"type":"mcp_toolset","mcp_server_name":"kb","default_config":{"enabled":false},
            "configs":{"search":{"enabled":true,"defer_loading":true}},
            "tools":[{"name":"search","input_schema":{"type":"object"},"description":"Search"}]}),
        BuiltinMessagesTool::McpToolset(McpToolset {
            mcp_server_name: String::from("kb"),
            cache_control: None,
            configs: Some(IndexMap::from([(String::from("search"), toolset_config(Some(true), Some(true)))])),
            default_config: Some(toolset_config(Some(false), None)),
            tools: Some(vec![McpListedTool {
                name: String::from("search"),
                description: Some(String::from("Search")),
                input_schema: JsonSchema::Object(Box::new(JsonSchemaObject {
                    schema_type: Some(JsonSchemaType::Name(String::from("object"))),
                    ..JsonSchemaObject::default()
                })),
                extra: Map::new(),
            }]),
            extra: Map::new(),
        })
    )]
    fn builtin_tools_decode_typed_definitions(
        #[case] wire: Value,
        #[case] expected: BuiltinMessagesTool,
    ) {
        assert_eq!(round_trip::<BuiltinMessagesTool>(wire), expected);
    }

    #[rstest]
    fn builtin_tools_preserve_unknown_fields() {
        let tool = round_trip::<BuiltinMessagesTool>(
            json!({"type":"bash_20250124","name":"bash","extension":[1,null]}),
        );
        let BuiltinMessagesTool::Bash20250124(bash) = tool else {
            panic!("expected bash tool");
        };
        assert_eq!(bash.extra.get("extension"), Some(&json!([1, null])));
    }

    #[rstest]
    #[case::missing_discriminator(json!({"name":"bash"}))]
    #[case::unknown_discriminator(json!({"type":"future_tool","name":"bash"}))]
    #[case::null_discriminator(json!({"type":null,"name":"bash"}))]
    #[case::bash_without_name(json!({"type":"bash_20250124"}))]
    #[case::bash_wrong_name(json!({"type":"bash_20241022","name":"shell"}))]
    #[case::legacy_editor_with_new_name(json!({"type":"text_editor_20250124","name":"str_replace_based_edit_tool"}))]
    #[case::new_editor_with_legacy_name(json!({"type":"text_editor_20250728","name":"str_replace_editor"}))]
    #[case::editor_20250429_with_legacy_name(json!({"type":"text_editor_20250429","name":"str_replace_editor"}))]
    #[case::memory_without_name(json!({"type":"memory_20250818"}))]
    #[case::computer_without_width(json!({"type":"computer_20250124","name":"computer","display_height_px":768}))]
    #[case::computer_without_height(json!({"type":"computer_20241022","name":"computer","display_width_px":1024}))]
    #[case::zoom_computer_without_display(json!({"type":"computer_20251124","name":"computer"}))]
    #[case::code_execution_without_name(json!({"type":"code_execution_20250825"}))]
    #[case::regex_search_with_bm25_name(json!({"type":"tool_search_tool_regex","name":"tool_search_tool_bm25"}))]
    #[case::bm25_search_with_regex_name(json!({"type":"tool_search_tool_bm25_20251119","name":"tool_search_tool_regex"}))]
    #[case::web_search_with_fetch_name(json!({"type":"web_search_20250305","name":"web_fetch"}))]
    #[case::web_fetch_without_name(json!({"type":"web_fetch_20260318"}))]
    #[case::advisor_without_model(json!({"type":"advisor_20260301","name":"advisor"}))]
    #[case::advisor_without_name(json!({"type":"advisor_20260301","model":"claude-opus-5-5"}))]
    #[case::mcp_toolset_without_server(json!({"type":"mcp_toolset"}))]
    #[case::unknown_allowed_caller(json!({"type":"code_execution_20250825","name":"code_execution","allowed_callers":["code_execution_20250522"]}))]
    #[case::unknown_response_inclusion(json!({"type":"web_search_20260318","name":"web_search","response_inclusion":"partial"}))]
    #[case::user_input_only_filter(json!({"type":"web_fetch_20250910","name":"web_fetch","url_sources":{"user_input":{"type":"only","tools":[]}}}))]
    #[case::negative_limit(json!({"type":"web_search_20250305","name":"web_search","max_uses":-1}))]
    #[case::wrong_mcp_config(json!({"type":"mcp_toolset","mcp_server_name":"kb","configs":{"search":{"enabled":"yes"}}}))]
    #[case::wrong_browser_config(json!({"type":"browser_toolset_20260801","configs":{"navigate":{"enabled":1}}}))]
    fn builtin_tools_reject_malformed_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<BuiltinMessagesTool>(wire.clone()).is_err());
        assert!(serde_json::from_value::<MessagesToolParam>(wire).is_err());
    }

    #[rstest]
    fn custom_tool_exposes_schema_and_preserves_extensions() {
        let tool = round_trip::<CustomTool>(json!({
            "type":"custom",
            "name":"lookup",
            "input_schema":{"type":"object","properties":{"query":{"type":"string"}}},
            "strict":false,
            "defer_loading":true,
            "allowed_callers":["direct","code_execution_20260120"],
            "extension":{"nested":[1,null]}
        }));
        assert_eq!(
            tool.allowed_callers.as_deref(),
            Some([AllowedCaller::Direct, AllowedCaller::CodeExecution20260120].as_slice())
        );
        assert_eq!(tool.tool_type, Some(CustomToolType::Custom));
        assert_eq!(tool.name, "lookup");
        assert_eq!((tool.strict, tool.defer_loading), (Some(false), Some(true)));
        let JsonSchema::Object(schema) = &tool.input_schema else {
            panic!("expected an object schema");
        };
        assert!(schema.properties.as_ref().unwrap().contains_key("query"));
        assert_eq!(tool.extra["extension"], json!({"nested":[1,null]}));
    }

    #[rstest]
    fn custom_tool_omits_null_discriminator() {
        let tool: CustomTool =
            serde_json::from_value(json!({"type":null,"name":"lookup","input_schema":true}))
                .unwrap();
        assert!(tool.tool_type.is_none());
        assert!(tool.description.is_none());
        assert_eq!(
            serde_json::to_value(tool).unwrap(),
            json!({"name":"lookup","input_schema":true})
        );
    }

    #[rstest]
    #[case::missing_name(json!({"input_schema":{}}))]
    #[case::missing_schema(json!({"name":"lookup"}))]
    #[case::wrong_name_shape(json!({"name":7,"input_schema":{}}))]
    #[case::builtin_tool(json!({"name":"lookup","input_schema":{},"type":"bash_20250124"}))]
    #[case::unknown_discriminator(json!({"name":"lookup","input_schema":{},"type":"future_tool"}))]
    #[case::unknown_allowed_caller(json!({"name":"lookup","input_schema":{},"allowed_callers":["anyone"]}))]
    fn custom_tool_rejects_invalid_shapes(#[case] wire: Value) {
        assert!(serde_json::from_value::<CustomTool>(wire).is_err());
    }

    #[rstest]
    #[case::builtin(json!({"type":"web_search_20250305","name":"web_search","max_uses":2}), true)]
    #[case::builtin_toolset(json!({"type":"mcp_toolset","mcp_server_name":"kb"}), true)]
    #[case::custom(json!({"name":"lookup","input_schema":{"type":"object"}}), false)]
    fn tool_params_dispatch_on_discriminator(#[case] wire: Value, #[case] builtin: bool) {
        let tool = round_trip::<MessagesToolParam>(wire);
        assert_eq!(matches!(tool, MessagesToolParam::Builtin(_)), builtin);
    }

    #[rstest]
    fn tool_params_reject_unknown_tool_types() {
        assert!(
            serde_json::from_value::<MessagesToolParam>(
                json!({"type":"future_tool","name":"lookup","input_schema":{}})
            )
            .is_err()
        );
    }

    fn tool_reference(name: &str) -> UrlSourceToolReference {
        UrlSourceToolReference::ToolReference {
            name: String::from(name),
            extra: Map::new(),
        }
    }

    fn cache_control(cache_type: &str, ttl: &str) -> CacheControl {
        CacheControl {
            cache_type: Some(String::from(cache_type)),
            ttl: Some(String::from(ttl)),
            ..CacheControl::default()
        }
    }
}
