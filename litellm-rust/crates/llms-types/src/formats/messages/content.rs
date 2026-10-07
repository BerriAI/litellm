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
