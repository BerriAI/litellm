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

#[cfg(test)]
mod tests {

    use crate::formats::messages::AdvisorToolResultContent;

    use crate::formats::messages::{BashCodeExecutionOutput, BashCodeExecutionToolResultContent};

    use crate::formats::messages::{BlockContent, BrowserStateChange};

    use crate::formats::messages::Citation;

    use crate::formats::messages::CodeExecutionOutput;

    use crate::formats::messages::CodeExecutionToolResultContent;

    use crate::formats::messages::ContentSource;

    use crate::formats::messages::FallbackTrigger;

    use crate::formats::messages::{McpToolResultContent, McpToolResultText};

    use crate::formats::messages::MessagesContentPart;

    use crate::formats::messages::MessagesToolParam;

    use crate::formats::messages::{TextEditorCodeExecutionToolResultContent, TextEditorFileType};

    use crate::formats::messages::{ToolCaller, ToolChange, ToolChangeTarget};

    use crate::formats::messages::ToolSearchReference;

    use crate::formats::messages::ToolSearchToolResultContent;

    use crate::formats::messages::WebFetchDocument;

    use crate::formats::messages::WebFetchToolResultContent;

    use crate::formats::messages::WebSearchToolResultContent;

    use crate::test_support::*;

    use rstest::rstest;

    use serde_json::{Value, json};

    #[rstest]
    #[case::base64(json!({"type":"base64","media_type":"image/png","data":"AA=="}))]
    #[case::url(json!({"type":"url","url":"https://example.test/image"}))]
    #[case::file(json!({"type":"file","file_id":"file_1"}))]
    #[case::text(json!({"type":"text","media_type":"text/plain","data":"document"}))]
    #[case::content(json!({"type":"content","content":"nested"}))]
    fn content_sources_round_trip(#[case] wire: Value) {
        round_trip::<ContentSource>(wire);
    }

    fn part(wire: Value) -> MessagesContentPart {
        round_trip::<MessagesContentPart>(wire)
    }

    #[rstest]
    fn text_blocks_expose_every_citation_location() {
        let block = part(json!({
            "type":"text",
            "text":"cited",
            "citations":[
                {"type":"char_location","cited_text":"a","document_index":0,"start_char_index":1,"end_char_index":6,"file_id":"file_1"},
                {"type":"page_location","cited_text":"b","document_index":1,"start_page_number":1,"end_page_number":2},
                {"type":"content_block_location","cited_text":"c","document_index":2,"start_block_index":0,"end_block_index":1},
                {"type":"web_search_result_location","cited_text":"d","url":"https://example.test","encrypted_index":"opaque","title":"result"},
                {"type":"search_result_location","cited_text":"e","search_result_index":3,"source":"kb","start_block_index":0,"end_block_index":2}
            ],
            "cache_control":{"type":"ephemeral"}
        }));
        let MessagesContentPart::Text(text) = &block else {
            panic!("expected text block");
        };
        assert_eq!(text.text, "cited");
        let [
            Citation::CharLocation(chars),
            Citation::PageLocation(page),
            Citation::ContentBlockLocation(blocks),
            Citation::WebSearchResultLocation(search),
            Citation::SearchResultLocation(result),
        ] = text.citations.as_deref().unwrap()
        else {
            panic!("expected one citation of each location type");
        };
        assert_eq!((chars.start_char_index, chars.end_char_index), (1, 6));
        assert_eq!(chars.file_id.as_deref(), Some("file_1"));
        assert_eq!((page.start_page_number, page.end_page_number), (1, 2));
        assert_eq!(blocks.document_index, 2);
        assert_eq!(search.encrypted_index, "opaque");
        assert_eq!(result.search_result_index, 3);
        assert_eq!(result.source, "kb");
    }

    #[rstest]
    fn text_block_null_optionals_are_omitted() {
        let block: MessagesContentPart = serde_json::from_value(
            json!({"type":"text","text":"hi","citations":null,"cache_control":null,"future":null}),
        )
        .unwrap();
        let MessagesContentPart::Text(text) = &block else {
            panic!("expected text block");
        };
        assert!(text.citations.is_none());
        assert!(text.cache_control.is_none());
        assert_eq!(text.extra.get("future"), Some(&Value::Null));
        assert_eq!(
            serde_json::to_value(block).unwrap(),
            json!({"type":"text","text":"hi","future":null})
        );
    }

    #[rstest]
    fn request_media_blocks_expose_sources() {
        let MessagesContentPart::Image(image) = part(json!({
            "type":"image",
            "source":{"type":"base64","media_type":"image/png","data":"AA=="},
            "transformations":{"oversized_image":"error"}
        })) else {
            panic!("expected image block");
        };
        assert!(
            matches!(&image.source, ContentSource::Base64 { media_type, .. } if media_type == "image/png")
        );
        assert_eq!(image.extra["transformations"]["oversized_image"], "error");
        let MessagesContentPart::Document(document) = part(json!({
            "type":"document",
            "source":{"type":"content","content":[
                {"type":"text","text":"Section 1"},
                {"type":"image","source":{"type":"url","url":"https://example.test/chart.png"}}
            ]},
            "title":"Q3 report",
            "context":"quarterly",
            "citations":{"enabled":true}
        })) else {
            panic!("expected document block");
        };
        assert_eq!(document.title.as_deref(), Some("Q3 report"));
        assert_eq!(document.citations.as_ref().unwrap().enabled, Some(true));
        let ContentSource::Content {
            content: BlockContent::Blocks(blocks),
            ..
        } = &document.source
        else {
            panic!("expected content-block source");
        };
        assert!(matches!(
            blocks.as_slice(),
            [MessagesContentPart::Text(_), MessagesContentPart::Image(_)]
        ));
        let MessagesContentPart::SearchResult(search) = part(json!({
            "type":"search_result",
            "source":"https://example.test/result",
            "title":"result",
            "content":[{"type":"text","text":"found"}],
            "citations":{"enabled":false}
        })) else {
            panic!("expected search result block");
        };
        assert_eq!(search.source, "https://example.test/result");
        assert!(
            matches!(search.content.as_slice(), [MessagesContentPart::Text(text)] if text.text == "found")
        );
    }

    #[rstest]
    fn reasoning_and_tool_blocks_expose_required_fields() {
        let MessagesContentPart::Thinking(thinking) =
            part(json!({"type":"thinking","thinking":"plan","signature":"sig"}))
        else {
            panic!("expected thinking block");
        };
        assert_eq!(
            (thinking.thinking.as_str(), thinking.signature.as_str()),
            ("plan", "sig")
        );
        let MessagesContentPart::RedactedThinking(redacted) =
            part(json!({"type":"redacted_thinking","data":"opaque"}))
        else {
            panic!("expected redacted thinking block");
        };
        assert_eq!(redacted.data, "opaque");
        let MessagesContentPart::ToolUse(tool_use) = part(json!({
            "type":"tool_use",
            "id":"toolu_1",
            "name":"lookup",
            "input":{"query":[1,null]},
            "caller":{"type":"code_execution_20260120","tool_id":"srvtoolu_1"},
            "toolset_name":"browser"
        })) else {
            panic!("expected tool use block");
        };
        assert_eq!(tool_use.input["query"], json!([1, null]));
        assert!(
            matches!(&tool_use.caller, Some(ToolCaller::CodeExecution20260120 { tool_id, .. }) if tool_id == "srvtoolu_1")
        );
        assert_eq!(tool_use.toolset_name.as_deref(), Some("browser"));
        let MessagesContentPart::ToolResult(result) = part(json!({
            "type":"tool_result",
            "tool_use_id":"toolu_1",
            "is_error":false,
            "content":[
                {"type":"tool_reference","tool_name":"lookup"},
                {"type":"browser_state","tabs":[{"tab_id":"1","title":"","url":"","active":true}],
                 "state_changes":[{"type":"download_completed","download_id":"d1","url":"https://example.test/f","size_bytes":3}]}
            ]
        })) else {
            panic!("expected tool result block");
        };
        let Some(BlockContent::Blocks(blocks)) = &result.content else {
            panic!("expected nested result blocks");
        };
        let [
            MessagesContentPart::ToolReference(reference),
            MessagesContentPart::BrowserState(browser),
        ] = blocks.as_slice()
        else {
            panic!("expected tool reference and browser state");
        };
        assert_eq!(reference.tool_name, "lookup");
        assert_eq!(browser.tabs[0].active, Some(true));
        assert!(matches!(
            browser.state_changes.as_deref(),
            Some([BrowserStateChange::DownloadCompleted {
                size_bytes: Some(3),
                path: None,
                ..
            }])
        ));
        let MessagesContentPart::ToolResult(text_result) =
            part(json!({"type":"tool_result","tool_use_id":"toolu_2","content":"done"}))
        else {
            panic!("expected tool result block");
        };
        assert_eq!(text_result.content, Some(BlockContent::Text("done".into())));
        assert!(text_result.is_error.is_none());
    }

    #[rstest]
    fn server_tool_results_expose_nested_result_unions() {
        let MessagesContentPart::ServerToolUse(server) = part(
            json!({"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"rust"}}),
        ) else {
            panic!("expected server tool use");
        };
        assert_eq!(server.name, "web_search");
        let MessagesContentPart::WebSearchToolResult(search) = part(json!({
            "type":"web_search_tool_result",
            "tool_use_id":"srvtoolu_1",
            "content":[{"type":"web_search_result","url":"https://example.test","title":"t","encrypted_content":"e","page_age":"1d"}]
        })) else {
            panic!("expected web search result");
        };
        let WebSearchToolResultContent::Results(results) = &search.content else {
            panic!("expected search results");
        };
        assert_eq!(results[0].page_age.as_deref(), Some("1d"));
        let MessagesContentPart::WebSearchToolResult(search_error) = part(json!({
            "type":"web_search_tool_result",
            "tool_use_id":"srvtoolu_1",
            "content":{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}
        })) else {
            panic!("expected web search error");
        };
        assert!(
            matches!(&search_error.content, WebSearchToolResultContent::Error(error) if error.error_code == "max_uses_exceeded")
        );
        let MessagesContentPart::WebFetchToolResult(fetch) = part(json!({
            "type":"web_fetch_tool_result",
            "tool_use_id":"srvtoolu_2",
            "content":{"type":"web_fetch_result","url":"https://example.test","retrieved_at":"2026-01-01T00:00:00Z",
                "content":{"type":"document","source":{"type":"text","media_type":"text/plain","data":"page"}}}
        })) else {
            panic!("expected web fetch result");
        };
        let WebFetchToolResultContent::WebFetchResult(fetched) = &fetch.content else {
            panic!("expected fetched page");
        };
        let WebFetchDocument::Document(document) = &fetched.content;
        assert!(matches!(&document.source, ContentSource::Text { data, .. } if data == "page"));
        assert!(fetched.extra.is_empty());
        let MessagesContentPart::CodeExecutionToolResult(code) = part(json!({
            "type":"code_execution_tool_result",
            "tool_use_id":"srvtoolu_3",
            "content":{"type":"encrypted_code_execution_result","encrypted_stdout":"opaque","stderr":"","return_code":-1,
                "content":[{"type":"code_execution_output","file_id":"file_1"}]}
        })) else {
            panic!("expected code execution result");
        };
        let CodeExecutionToolResultContent::EncryptedCodeExecutionResult(encrypted) = &code.content
        else {
            panic!("expected encrypted execution result");
        };
        assert_eq!(encrypted.return_code, -1);
        assert!(
            matches!(encrypted.content.as_slice(), [CodeExecutionOutput::CodeExecutionOutput { file_id, .. }] if file_id == "file_1")
        );
        let MessagesContentPart::BashCodeExecutionToolResult(bash) = part(json!({
            "type":"bash_code_execution_tool_result",
            "tool_use_id":"srvtoolu_7",
            "content":{"type":"bash_code_execution_result","stdout":"ok","stderr":"","return_code":0,
                "content":[{"type":"bash_code_execution_output","file_id":"file_2"}]}
        })) else {
            panic!("expected bash code execution result");
        };
        let BashCodeExecutionToolResultContent::BashCodeExecutionResult(ran) = &bash.content else {
            panic!("expected bash execution result");
        };
        assert_eq!((ran.stdout.as_str(), ran.return_code), ("ok", 0));
        assert!(
            matches!(ran.content.as_slice(), [BashCodeExecutionOutput::BashCodeExecutionOutput { file_id, .. }] if file_id == "file_2")
        );
        assert!(ran.extra.is_empty());
        let MessagesContentPart::TextEditorCodeExecutionToolResult(editor) = part(json!({
            "type":"text_editor_code_execution_tool_result",
            "tool_use_id":"srvtoolu_4",
            "content":{"type":"text_editor_code_execution_view_result","content":"fn main() {}","file_type":"text","num_lines":1}
        })) else {
            panic!("expected text editor result");
        };
        let TextEditorCodeExecutionToolResultContent::TextEditorCodeExecutionViewResult(view) =
            &editor.content
        else {
            panic!("expected view result");
        };
        assert_eq!(view.file_type, TextEditorFileType::Text);
        assert_eq!(view.num_lines, Some(1));
        assert!(view.start_line.is_none());
        let MessagesContentPart::ToolSearchToolResult(tool_search) = part(json!({
            "type":"tool_search_tool_result",
            "tool_use_id":"srvtoolu_5",
            "content":{"type":"tool_search_tool_result_error","error_code":"unavailable","error_message":"down"}
        })) else {
            panic!("expected tool search result");
        };
        assert!(
            matches!(&tool_search.content, ToolSearchToolResultContent::ToolSearchToolResultError(error) if error.error_message.as_deref() == Some("down"))
        );
        let MessagesContentPart::ToolSearchToolResult(found) = part(json!({
            "type":"tool_search_tool_result",
            "tool_use_id":"srvtoolu_8",
            "content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"lookup"}]}
        })) else {
            panic!("expected tool search result");
        };
        let ToolSearchToolResultContent::ToolSearchToolSearchResult(result) = &found.content else {
            panic!("expected tool search hits");
        };
        let [ToolSearchReference::ToolReference(reference)] = result.tool_references.as_slice()
        else {
            panic!("expected one tool reference");
        };
        assert_eq!(reference.tool_name, "lookup");
        assert!(reference.extra.is_empty() && result.extra.is_empty());
        let MessagesContentPart::AdvisorToolResult(advisor) = part(json!({
            "type":"advisor_tool_result",
            "tool_use_id":"srvtoolu_6",
            "content":{"type":"advisor_result","text":"advice"}
        })) else {
            panic!("expected advisor result");
        };
        assert!(
            matches!(&advisor.content, AdvisorToolResultContent::AdvisorResult { text, stop_reason: None, .. } if text == "advice")
        );
    }

    #[rstest]
    fn beta_blocks_expose_mcp_compaction_and_fallback_fields() {
        let MessagesContentPart::McpToolUse(mcp) = part(
            json!({"type":"mcp_tool_use","id":"mcptoolu_1","name":"search","server_name":"kb","input":{}}),
        ) else {
            panic!("expected MCP tool use");
        };
        assert_eq!(mcp.server_name, "kb");
        let MessagesContentPart::McpToolResult(mcp_result) = part(
            json!({"type":"mcp_tool_result","tool_use_id":"mcptoolu_1","is_error":true,"content":"failed"}),
        ) else {
            panic!("expected MCP tool result");
        };
        assert_eq!(mcp_result.is_error, Some(true));
        assert_eq!(
            mcp_result.content,
            Some(McpToolResultContent::Text("failed".into()))
        );
        let MessagesContentPart::McpToolResult(text_result) = part(json!({
            "type":"mcp_tool_result",
            "tool_use_id":"mcptoolu_3",
            "content":[{"type":"text","text":"found"}]
        })) else {
            panic!("expected MCP tool result");
        };
        let Some(McpToolResultContent::Blocks(blocks)) = &text_result.content else {
            panic!("expected MCP text blocks");
        };
        let [McpToolResultText::Text(text)] = blocks.as_slice() else {
            panic!("expected one text block");
        };
        assert_eq!(text.text, "found");
        let MessagesContentPart::McpToolResult(empty_result) =
            part(json!({"type":"mcp_tool_result","tool_use_id":"mcptoolu_2"}))
        else {
            panic!("expected MCP tool result");
        };
        assert!(empty_result.content.is_none());
        assert!(empty_result.extra.is_empty());
        let MessagesContentPart::McpToolListing(listing) = part(json!({
            "type":"mcp_tool_listing",
            "mcp_server_name":"kb",
            "tools":[{"name":"search","input_schema":{"type":"object"}}]
        })) else {
            panic!("expected MCP tool listing");
        };
        assert!(listing.tools[0].description.is_none());
        let MessagesContentPart::Compaction(compaction) = part(json!({
            "type":"compaction",
            "content":"summary",
            "encrypted_content":"opaque",
            "tool_changes":[
                {"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"lookup","input_schema":{"type":"object"}}}},
                {"type":"tool_removal","tool":{"type":"mcp_tool_reference","server_name":"kb","name":"search"}}
            ]
        })) else {
            panic!("expected compaction block");
        };
        let [
            ToolChange::ToolAddition(addition),
            ToolChange::ToolRemoval(removal),
        ] = compaction.tool_changes.as_deref().unwrap()
        else {
            panic!("expected tool addition and removal");
        };
        let ToolChangeTarget::ToolDefinition { definition, .. } = &addition.tool else {
            panic!("expected inline tool definition");
        };
        assert!(
            matches!(definition.as_ref(), MessagesToolParam::Custom(tool) if tool.name == "lookup")
        );
        assert!(
            matches!(&removal.tool, ToolChangeTarget::McpToolReference { server_name, .. } if server_name == "kb")
        );
        let failed: MessagesContentPart =
            serde_json::from_value(json!({"type":"compaction","content":null})).unwrap();
        assert_eq!(failed, MessagesContentPart::Compaction(Default::default()));
        let MessagesContentPart::Fallback(fallback) = part(json!({
            "type":"fallback",
            "from":{"model":"claude-opus-5-5"},
            "to":{"model":"claude-sonnet-5-5"},
            "trigger":{"type":"refusal","category":"cyber"}
        })) else {
            panic!("expected fallback block");
        };
        assert_eq!(fallback.from.model, "claude-opus-5-5");
        assert_eq!(fallback.to.model, "claude-sonnet-5-5");
        let Some(FallbackTrigger::Refusal { category, extra }) = &fallback.trigger else {
            panic!("expected refusal trigger");
        };
        assert_eq!(category.as_deref(), Some("cyber"));
        assert!(extra.is_empty() && fallback.extra.is_empty());
        let MessagesContentPart::Fallback(untriggered) = part(json!({
            "type":"fallback",
            "from":{"model":"claude-opus-5-5"},
            "to":{"model":"claude-sonnet-5-5"}
        })) else {
            panic!("expected fallback block");
        };
        assert!(untriggered.trigger.is_none());
    }

    #[rstest]
    #[case::missing_tag(json!({"text":"hello"}))]
    #[case::unknown_tag(json!({"type":"future_block","text":"hello"}))]
    #[case::text_without_text(json!({"type":"text"}))]
    #[case::wrong_text(json!({"type":"text","text":7}))]
    #[case::citation_missing_location(json!({"type":"text","text":"a","citations":[{"type":"char_location","cited_text":"a","document_index":0}]}))]
    #[case::unknown_citation(json!({"type":"text","text":"a","citations":[{"type":"future_location"}]}))]
    #[case::thinking_without_signature(json!({"type":"thinking","thinking":"plan"}))]
    #[case::tool_use_without_id(json!({"type":"tool_use","name":"lookup","input":{}}))]
    #[case::tool_use_array_input(json!({"type":"tool_use","id":"t","name":"lookup","input":[]}))]
    #[case::tool_result_without_id(json!({"type":"tool_result","content":"done"}))]
    #[case::malformed_nested_block(json!({"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":7}]}))]
    #[case::malformed_source(json!({"type":"document","source":{"type":"url","url":7}}))]
    #[case::image_without_source(json!({"type":"image"}))]
    #[case::search_result_without_title(json!({"type":"search_result","source":"s","content":[]}))]
    #[case::web_search_result_without_url(json!({"type":"web_search_tool_result","tool_use_id":"t","content":[{"type":"web_search_result","title":"t","encrypted_content":"e"}]}))]
    #[case::unknown_fetch_result(json!({"type":"web_fetch_tool_result","tool_use_id":"t","content":{"type":"future"}}))]
    #[case::fractional_return_code(json!({"type":"bash_code_execution_tool_result","tool_use_id":"t","content":{"type":"bash_code_execution_result","stdout":"","stderr":"","return_code":0.5,"content":[]}}))]
    #[case::unknown_file_type(json!({"type":"text_editor_code_execution_tool_result","tool_use_id":"t","content":{"type":"text_editor_code_execution_view_result","content":"","file_type":"video"}}))]
    #[case::negative_line_count(json!({"type":"text_editor_code_execution_tool_result","tool_use_id":"t","content":{"type":"text_editor_code_execution_str_replace_result","new_lines":-1}}))]
    #[case::unknown_tool_change(json!({"type":"compaction","tool_changes":[{"type":"tool_addition","tool":{"type":"future"}}]}))]
    #[case::tool_search_non_reference(json!({"type":"tool_search_tool_result","tool_use_id":"t","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"text","text":"lookup"}]}}))]
    #[case::web_fetch_text_content(json!({"type":"web_fetch_tool_result","tool_use_id":"t","content":{"type":"web_fetch_result","url":"https://example.test","content":{"type":"text","text":"page"}}}))]
    #[case::bash_result_with_code_output(json!({"type":"bash_code_execution_tool_result","tool_use_id":"t","content":{"type":"bash_code_execution_result","stdout":"","stderr":"","return_code":0,"content":[{"type":"code_execution_output","file_id":"f"}]}}))]
    #[case::code_result_with_bash_output(json!({"type":"code_execution_tool_result","tool_use_id":"t","content":{"type":"code_execution_result","stdout":"","stderr":"","return_code":0,"content":[{"type":"bash_code_execution_output","file_id":"f"}]}}))]
    #[case::fallback_unknown_trigger(json!({"type":"fallback","from":{"model":"a"},"to":{"model":"b"},"trigger":{"type":"overload"}}))]
    #[case::unknown_state_change(json!({"type":"browser_state","tabs":[],"state_changes":[{"type":"tab_closed","tab_id":"1"}]}))]
    fn content_parts_reject_malformed_known_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<MessagesContentPart>(wire).is_err());
    }

    #[rstest]
    #[case::image(json!([{"type":"image","source":{"type":"url","url":"https://example.test/a.png"}}]))]
    #[case::tool_reference(json!([{"type":"tool_reference","tool_name":"lookup"}]))]
    #[case::untagged(json!([{"text":"found"}]))]
    fn mcp_tool_result_content_rejects_non_text_blocks(#[case] content: Value) {
        assert!(
            serde_json::from_value::<MessagesContentPart>(
                json!({"type":"mcp_tool_result","tool_use_id":"mcptoolu_1","content":content})
            )
            .is_err()
        );
    }
}
