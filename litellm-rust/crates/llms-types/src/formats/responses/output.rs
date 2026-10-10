use crate::formats::chat_completions::PromptCacheBreakpoint;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message(ResponsesMessage),
    FunctionCall(ResponsesFunctionCall),
    CustomToolCall(ResponsesCustomToolCall),
    Reasoning(ResponsesReasoning),
    WebSearchCall(ResponsesWebSearchCall),
    FileSearchCall(ResponsesFileSearchCall),
    ImageGenerationCall(ResponsesImageGenerationCall),
    CodeInterpreterCall(ResponsesCodeInterpreterCall),
    McpCall(ResponsesMcpCall),
    McpListTools(ResponsesMcpListTools),
    FunctionCallOutput(ResponsesFunctionCallOutput),
    ComputerCall(ResponsesComputerCall),
    ComputerCallOutput(ResponsesComputerCallOutput),
    Program(ResponsesProgram),
    ProgramOutput(ResponsesProgramOutput),
    ToolSearchCall(ResponsesToolSearchCall),
    ToolSearchOutput(ResponsesToolSearchOutput),
    AdditionalTools(ResponsesAdditionalTools),
    Compaction(ResponsesCompaction),
    LocalShellCall(ResponsesLocalShellCall),
    LocalShellCallOutput(ResponsesLocalShellCallOutput),
    ShellCall(ResponsesShellCall),
    ShellCallOutput(ResponsesShellCallOutput),
    ApplyPatchCall(ResponsesApplyPatchCall),
    ApplyPatchCallOutput(ResponsesApplyPatchCallOutput),
    McpApprovalRequest(ResponsesMcpApprovalRequest),
    McpApprovalResponse(ResponsesMcpApprovalResponse),
    CustomToolCallOutput(ResponsesCustomToolCallOutput),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<ResponsesAnnotation>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<crate::formats::chat_completions::ChatTokenLogprob>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Refusal {
        refusal: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    SummaryText {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ReasoningText {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesAnnotation {
    UrlCitation(ResponsesUrlCitation),
    FileCitation(ResponsesFileCitation),
    FilePath(ResponsesFileCitation),
    ContainerFileCitation(ResponsesFileCitation),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesCodeOutput {
    Logs {
        logs: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Image {
        url: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesMessage {
    pub id: Option<String>,
    pub status: Option<String>,
    pub role: Option<String>,
    pub phase: Option<String>,
    pub content: Option<Vec<ResponsesContentPart>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesFunctionCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
    pub namespace: Option<String>,
    pub r#async: Option<bool>,
    pub caller: Option<ResponsesToolCaller>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesCustomToolCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub name: Option<String>,
    pub input: Option<String>,
    pub namespace: Option<String>,
    pub r#async: Option<bool>,
    pub caller: Option<ResponsesToolCaller>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesReasoning {
    pub id: Option<String>,
    pub status: Option<String>,
    pub summary: Option<Vec<ResponsesContentPart>>,
    pub content: Option<Vec<ResponsesContentPart>>,
    pub encrypted_content: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesWebSearchCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub action: Option<ResponsesWebSearchAction>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesFileSearchCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub queries: Option<Vec<String>>,
    pub results: Option<Vec<ResponsesFileSearchResult>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesFileSearchResult {
    pub file_id: Option<String>,
    pub filename: Option<String>,
    pub score: Option<serde_json::Number>,
    pub text: Option<String>,
    pub attributes: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesImageGenerationCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub result: Option<String>,
    pub action: Option<String>,
    pub background: Option<String>,
    pub output_format: Option<String>,
    pub quality: Option<String>,
    pub revised_prompt: Option<String>,
    pub size: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesCodeInterpreterCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub code: Option<String>,
    pub container_id: Option<String>,
    pub outputs: Option<Vec<ResponsesCodeOutput>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesMcpCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub name: Option<String>,
    pub server_label: Option<String>,
    pub arguments: Option<String>,
    pub output: Option<String>,
    pub error: Option<ResponsesMcpError>,
    pub approval_request_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesUrlCitation {
    pub url: Option<String>,
    pub title: Option<String>,
    pub start_index: Option<u64>,
    pub end_index: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesFileCitation {
    pub file_id: Option<String>,
    pub filename: Option<String>,
    pub index: Option<u64>,
    pub start_index: Option<u64>,
    pub end_index: Option<u64>,
    pub container_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesWebSearchAction {
    Search {
        #[serde(skip_serializing_if = "Option::is_none")]
        query: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        queries: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        sources: Option<Vec<ResponsesWebSearchSource>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    OpenPage {
        #[serde(skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    #[serde(alias = "find")]
    FindInPage {
        url: String,
        pattern: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesWebSearchSource {
    #[serde(rename = "type")]
    pub source_type: Option<String>,
    pub url: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesMcpListTools {
    pub id: Option<String>,
    pub server_label: Option<String>,
    pub tools: Option<Vec<ResponsesMcpTool>>,
    pub error: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesMcpTool {
    pub name: Option<String>,
    pub description: Option<String>,
    pub input_schema: Option<crate::json_schema::JsonSchema>,
    pub annotations: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum ResponsesMcpError {
    Message(String),
    Detail(ResponsesMcpErrorDetail),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesMcpErrorDetail {
    McpProtocolError {
        code: i64,
        message: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    HttpError {
        code: i64,
        message: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    McpToolExecutionError {
        content: Value,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesToolCaller {
    Direct {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Program {
        caller_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum ResponsesToolOutput {
    Text(String),
    Content(Vec<ResponsesInputContent>),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesInputContent {
    InputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    InputImage {
        detail: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesFunctionCallOutput {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub output: Option<ResponsesToolOutput>,
    pub caller: Option<ResponsesToolCaller>,
    pub created_by: Option<String>,
    pub name: Option<String>,
    pub namespace: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesCustomToolCallOutput {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub output: Option<ResponsesToolOutput>,
    pub caller: Option<ResponsesToolCaller>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesSafetyCheck {
    pub id: Option<String>,
    pub code: Option<String>,
    pub message: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesComputerCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub pending_safety_checks: Option<Vec<ResponsesSafetyCheck>>,
    pub action: Option<ResponsesComputerAction>,
    pub actions: Option<Vec<ResponsesComputerAction>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesComputerAction {
    Click {
        button: String,
        x: i64,
        y: i64,
        #[serde(skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    DoubleClick {
        x: i64,
        y: i64,
        #[serde(skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Drag {
        path: Vec<ResponsesCoordinate>,
        #[serde(skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Keypress {
        keys: Vec<String>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Move {
        x: i64,
        y: i64,
        #[serde(skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Screenshot {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Scroll {
        scroll_x: i64,
        scroll_y: i64,
        x: i64,
        y: i64,
        #[serde(skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Type {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Wait {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ResponsesCoordinate {
    pub x: i64,
    pub y: i64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesComputerCallOutput {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub output: Option<ResponsesComputerOutput>,
    pub acknowledged_safety_checks: Option<Vec<ResponsesSafetyCheck>>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesComputerOutput {
    ComputerScreenshot {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesProgram {
    pub id: Option<String>,
    pub call_id: Option<String>,
    pub code: Option<String>,
    pub fingerprint: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesProgramOutput {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub result: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesToolSearchCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub arguments: Option<Value>,
    pub execution: Option<String>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesToolSearchOutput {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub execution: Option<String>,
    pub tools: Option<Vec<Map<String, Value>>>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesAdditionalTools {
    pub id: Option<String>,
    pub role: Option<String>,
    pub tools: Option<Vec<Map<String, Value>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesCompaction {
    pub id: Option<String>,
    pub encrypted_content: Option<String>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesLocalShellCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub action: Option<ResponsesLocalShellAction>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesLocalShellAction {
    Exec {
        command: Vec<String>,
        env: BTreeMap<String, String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        working_directory: Option<String>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesLocalShellCallOutput {
    pub id: Option<String>,
    pub status: Option<String>,
    pub output: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesShellCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub action: Option<ResponsesShellAction>,
    pub environment: Option<ResponsesShellEnvironment>,
    pub caller: Option<ResponsesToolCaller>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesShellAction {
    pub commands: Option<Vec<String>>,
    pub max_output_length: Option<u64>,
    pub timeout_ms: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesShellEnvironment {
    Local {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ContainerReference {
        container_id: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesShellCallOutput {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub max_output_length: Option<u64>,
    pub output: Option<Vec<ResponsesShellOutputChunk>>,
    pub caller: Option<ResponsesToolCaller>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesShellOutputChunk {
    pub outcome: Option<ResponsesShellOutcome>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesShellOutcome {
    Timeout {
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Exit {
        exit_code: i64,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesApplyPatchCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub operation: Option<ResponsesApplyPatchOperation>,
    pub caller: Option<ResponsesToolCaller>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesApplyPatchOperation {
    CreateFile {
        path: String,
        diff: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    DeleteFile {
        path: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    UpdateFile {
        path: String,
        diff: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesApplyPatchCallOutput {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub output: Option<String>,
    pub caller: Option<ResponsesToolCaller>,
    pub created_by: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesMcpApprovalRequest {
    pub id: Option<String>,
    pub server_label: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ResponsesMcpApprovalResponse {
    pub id: Option<String>,
    pub approval_request_id: Option<String>,
    pub approve: Option<bool>,
    pub reason: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
