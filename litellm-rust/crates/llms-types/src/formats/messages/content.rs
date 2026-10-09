use serde_json::{Map, Value};

use super::{CacheControl, MessagesToolParam};
use crate::json_schema::JsonSchema;

#[macro_rules_attribute::apply(crate::wire_type)]
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

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum BlockContent {
    Text(String),
    Blocks(Vec<MessagesContentPart>),
}

#[macro_rules_attribute::apply(crate::wire_type)]
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
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120 {
        tool_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct CitationsConfig {
    pub enabled: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct PageCitation {
    pub cited_text: String,
    pub document_index: u64,
    pub document_title: Option<String>,
    pub start_page_number: u64,
    pub end_page_number: u64,
    pub file_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct CharCitation {
    pub cited_text: String,
    pub document_index: u64,
    pub document_title: Option<String>,
    pub start_char_index: u64,
    pub end_char_index: u64,
    pub file_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ContentBlockCitation {
    pub cited_text: String,
    pub document_index: u64,
    pub document_title: Option<String>,
    pub start_block_index: u64,
    pub end_block_index: u64,
    pub file_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebSearchCitation {
    pub cited_text: String,
    pub url: String,
    pub encrypted_index: String,
    pub title: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct SearchResultCitation {
    pub cited_text: String,
    pub search_result_index: u64,
    pub source: String,
    pub title: Option<String>,
    pub start_block_index: u64,
    pub end_block_index: u64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Citation {
    PageLocation(PageCitation),
    CharLocation(CharCitation),
    WebSearchResultLocation(WebSearchCitation),
    ContentBlockLocation(ContentBlockCitation),
    SearchResultLocation(SearchResultCitation),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagesContentPart {
    Text(TextBlock),
    Image(ImageBlock),
    Document(Box<DocumentBlock>),
    SearchResult(SearchResultBlock),
    Thinking(ThinkingBlock),
    RedactedThinking(RedactedThinkingBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    ToolReference(ToolReferenceBlock),
    BrowserState(BrowserStateBlock),
    ServerToolUse(ServerToolUseBlock),
    WebSearchToolResult(ServerToolResultBlock<WebSearchToolResultContent>),
    WebFetchToolResult(ServerToolResultBlock<WebFetchToolResultContent>),
    CodeExecutionToolResult(ServerToolResultBlock<CodeExecutionToolResultContent>),
    BashCodeExecutionToolResult(ServerToolResultBlock<BashCodeExecutionToolResultContent>),
    TextEditorCodeExecutionToolResult(
        ServerToolResultBlock<TextEditorCodeExecutionToolResultContent>,
    ),
    ToolSearchToolResult(ServerToolResultBlock<ToolSearchToolResultContent>),
    AdvisorToolResult(ServerToolResultBlock<AdvisorToolResultContent>),
    ContainerUpload(ContainerUploadBlock),
    McpToolUse(McpToolUseBlock),
    McpToolResult(McpToolResultBlock),
    McpToolListing(McpToolListingBlock),
    Compaction(CompactionBlock),
    ToolAddition(ToolChangeBlock),
    ToolRemoval(ToolChangeBlock),
    Fallback(FallbackBlock),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct TextBlock {
    pub text: String,
    pub citations: Option<Vec<Citation>>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ImageBlock {
    pub source: ContentSource,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct DocumentBlock {
    pub source: ContentSource,
    pub title: Option<String>,
    pub context: Option<String>,
    pub citations: Option<CitationsConfig>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct SearchResultBlock {
    pub source: String,
    pub title: String,
    pub content: Vec<MessagesContentPart>,
    pub citations: Option<CitationsConfig>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ThinkingBlock {
    pub thinking: String,
    pub signature: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct RedactedThinkingBlock {
    pub data: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: Map<String, Value>,
    pub caller: Option<ToolCaller>,
    pub toolset_name: Option<String>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: Option<BlockContent>,
    pub is_error: Option<bool>,
    pub toolset_name: Option<String>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ToolReferenceBlock {
    pub tool_name: String,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct BrowserStateBlock {
    pub tabs: Vec<BrowserTab>,
    pub state_changes: Option<Vec<BrowserStateChange>>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct BrowserTab {
    pub tab_id: String,
    pub title: String,
    pub url: String,
    pub active: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BrowserStateChange {
    TabOpened {
        tab_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    DownloadStarted {
        download_id: String,
        url: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    DownloadCompleted {
        download_id: String,
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        size_bytes: Option<u64>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    DownloadFailed {
        download_id: String,
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ServerToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: Map<String, Value>,
    pub caller: Option<ToolCaller>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ServerToolResultBlock<C> {
    pub tool_use_id: String,
    pub content: C,
    pub caller: Option<ToolCaller>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ServerToolError {
    pub error_code: String,
    pub error_message: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum WebSearchToolResultContent {
    Error(WebSearchResultError),
    Results(Vec<WebSearchResult>),
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebSearchResultError {
    #[serde(rename = "type")]
    pub error_type: WebSearchResultErrorType,
    pub error_code: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchResultErrorType {
    WebSearchToolResultError,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebSearchResult {
    #[serde(rename = "type")]
    pub result_type: WebSearchResultType,
    pub url: String,
    pub title: String,
    pub encrypted_content: String,
    pub page_age: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchResultType {
    WebSearchResult,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebFetchToolResultContent {
    WebFetchToolResultError(ServerToolError),
    WebFetchResult(WebFetchResult),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct WebFetchResult {
    pub url: String,
    pub content: WebFetchDocument,
    pub retrieved_at: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebFetchDocument {
    Document(Box<DocumentBlock>),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodeExecutionToolResultContent {
    CodeExecutionToolResultError(ServerToolError),
    CodeExecutionResult(CodeExecutionResult),
    EncryptedCodeExecutionResult(EncryptedCodeExecutionResult),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BashCodeExecutionToolResultContent {
    BashCodeExecutionToolResultError(ServerToolError),
    BashCodeExecutionResult(CodeExecutionResult<BashCodeExecutionOutput>),
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct CodeExecutionResult<O = CodeExecutionOutput> {
    pub stdout: String,
    pub stderr: String,
    pub return_code: i64,
    pub content: Vec<O>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct EncryptedCodeExecutionResult {
    pub encrypted_stdout: String,
    pub stderr: String,
    pub return_code: i64,
    pub content: Vec<CodeExecutionOutput>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodeExecutionOutput {
    CodeExecutionOutput {
        file_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BashCodeExecutionOutput {
    BashCodeExecutionOutput {
        file_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextEditorCodeExecutionToolResultContent {
    TextEditorCodeExecutionToolResultError(ServerToolError),
    TextEditorCodeExecutionViewResult(TextEditorViewResult),
    TextEditorCodeExecutionCreateResult(TextEditorCreateResult),
    TextEditorCodeExecutionStrReplaceResult(TextEditorStrReplaceResult),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct TextEditorViewResult {
    pub content: String,
    pub file_type: TextEditorFileType,
    pub num_lines: Option<u64>,
    pub start_line: Option<u64>,
    pub total_lines: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum TextEditorFileType {
    Text,
    Image,
    Pdf,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct TextEditorCreateResult {
    pub is_file_update: bool,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct TextEditorStrReplaceResult {
    pub lines: Option<Vec<String>>,
    pub new_lines: Option<u64>,
    pub new_start: Option<u64>,
    pub old_lines: Option<u64>,
    pub old_start: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolSearchToolResultContent {
    ToolSearchToolResultError(ServerToolError),
    ToolSearchToolSearchResult(ToolSearchResult),
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ToolSearchResult {
    pub tool_references: Vec<ToolSearchReference>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolSearchReference {
    ToolReference(ToolReferenceBlock),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AdvisorToolResultContent {
    AdvisorToolResultError(ServerToolError),
    AdvisorResult {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    AdvisorRedactedResult {
        encrypted_content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ContainerUploadBlock {
    pub file_id: String,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct McpToolUseBlock {
    pub id: String,
    pub name: String,
    pub server_name: String,
    pub input: Map<String, Value>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct McpToolResultBlock {
    pub tool_use_id: String,
    pub content: Option<McpToolResultContent>,
    pub is_error: Option<bool>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum McpToolResultContent {
    Text(String),
    Blocks(Vec<McpToolResultText>),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpToolResultText {
    Text(TextBlock),
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct McpToolListingBlock {
    pub mcp_server_name: String,
    pub tools: Vec<McpListedTool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct McpListedTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: JsonSchema,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct CompactionBlock {
    pub content: Option<String>,
    pub encrypted_content: Option<String>,
    pub signature: Option<String>,
    pub tool_changes: Option<Vec<ToolChange>>,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChange {
    ToolAddition(ToolChangeBlock),
    ToolRemoval(ToolChangeBlock),
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ToolChangeBlock {
    pub tool: ToolChangeTarget,
    pub cache_control: Option<CacheControl>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChangeTarget {
    ToolReference {
        name: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    McpToolReference {
        server_name: String,
        name: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    McpToolsetReference {
        server_name: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ToolDefinition {
        definition: Box<MessagesToolParam>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct FallbackBlock {
    pub from: FallbackModel,
    pub to: FallbackModel,
    pub trigger: Option<FallbackTrigger>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct FallbackModel {
    pub model: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FallbackTrigger {
    Refusal {
        #[serde(skip_serializing_if = "Option::is_none")]
        category: Option<String>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}
