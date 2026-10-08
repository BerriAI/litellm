use serde_json::{Map, Value};

use super::ContentBlock;

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentSource {
    Base64 {
        media_type: String,
        data: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Url {
        url: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    File {
        file_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Text {
        media_type: String,
        data: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Content {
        content: BlockContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum BlockContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
    SearchError(WebSearchResultError),
    Block(Box<ContentBlock>),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCaller {
    Direct {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    #[serde(rename = "code_execution_20250825")]
    CodeExecution {
        tool_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct CitationsConfig {
    pub enabled: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct PageCitation {
    pub cited_text: Option<String>,
    pub document_index: Option<u64>,
    pub document_title: Option<String>,
    pub start_page_number: Option<u64>,
    pub end_page_number: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct CharCitation {
    pub cited_text: Option<String>,
    pub document_index: Option<u64>,
    pub document_title: Option<String>,
    pub start_char_index: Option<u64>,
    pub end_char_index: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct WebSearchCitation {
    pub cited_text: Option<String>,
    pub url: Option<String>,
    pub title: Option<String>,
    pub encrypted_index: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Citation {
    PageLocation(PageCitation),
    CharLocation(CharCitation),
    WebSearchResultLocation(WebSearchCitation),
    ContentBlockLocation(ContentBlockCitation),
    SearchResultLocation(SearchResultCitation),
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum Citations {
    Config(CitationsConfig),
    Results(Vec<Citation>),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct WebSearchResultError {
    #[serde(rename = "type")]
    pub error_type: WebSearchResultErrorType,
    pub error_code: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchResultErrorType {
    WebSearchToolResultError,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct PromptCacheBreakpoint {
    pub mode: Option<PromptCacheMode>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheMode {
    Explicit,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ContentBlockCitation {
    pub cited_text: Option<String>,
    pub document_index: Option<u64>,
    pub document_title: Option<String>,
    pub start_block_index: Option<u64>,
    pub end_block_index: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct SearchResultCitation {
    pub cited_text: Option<String>,
    pub search_result_index: Option<u64>,
    pub title: Option<String>,
    pub source: Option<String>,
    pub start_block_index: Option<u64>,
    pub end_block_index: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ContentBlockPayload {
    pub tool: Option<Box<ContentBlock>>,
    pub text: Option<String>,
    pub thinking: Option<String>,
    pub signature: Option<String>,
    pub data: Option<String>,
    pub id: Option<String>,
    pub name: Option<String>,
    pub input: Option<Map<String, Value>>,
    pub content: Option<BlockContent>,
    pub provider_specific_fields: Option<Map<String, Value>>,
    pub source: Option<ContentSource>,
    pub citations: Option<Citations>,
    pub caller: Option<ToolCaller>,
    pub is_error: Option<bool>,
    pub file_id: Option<String>,
    pub title: Option<String>,
    pub context: Option<String>,
    pub tool_name: Option<String>,
    pub url: Option<String>,
    pub page_age: Option<String>,
    pub encrypted_content: Option<String>,
    pub snippet: Option<String>,
    pub prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub return_code: Option<i64>,
    pub encrypted_stdout: Option<String>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub retrieved_at: Option<String>,
    pub server_name: Option<String>,
    pub tool_references: Option<Vec<super::ContentBlock>>,
    pub file_type: Option<String>,
    pub num_lines: Option<u64>,
    pub start_line: Option<u64>,
    pub total_lines: Option<u64>,
    pub is_file_update: Option<bool>,
    pub lines: Option<Vec<String>>,
    pub new_lines: Option<u64>,
    pub new_start: Option<u64>,
    pub old_lines: Option<u64>,
    pub old_start: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
