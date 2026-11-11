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

#[cfg(test)]
mod tests {
    use crate::formats::{
        chat_completions::{PromptCacheBreakpoint, PromptCacheMode},
        responses::{
            ResponsesAdditionalTools, ResponsesApplyPatchCall, ResponsesApplyPatchCallOutput,
            ResponsesApplyPatchOperation, ResponsesCodeOutput, ResponsesCompaction,
            ResponsesComputerAction, ResponsesComputerCall, ResponsesComputerCallOutput,
            ResponsesComputerOutput, ResponsesContentPart, ResponsesCoordinate,
            ResponsesCustomToolCall, ResponsesCustomToolCallOutput, ResponsesFunctionCall,
            ResponsesFunctionCallOutput, ResponsesImageGenerationCall, ResponsesInputContent,
            ResponsesLocalShellAction, ResponsesLocalShellCall, ResponsesLocalShellCallOutput,
            ResponsesMcpApprovalRequest, ResponsesMcpApprovalResponse, ResponsesMcpError,
            ResponsesMcpErrorDetail, ResponsesOutputItem, ResponsesProgram, ResponsesProgramOutput,
            ResponsesSafetyCheck, ResponsesShellAction, ResponsesShellCall,
            ResponsesShellCallOutput, ResponsesShellEnvironment, ResponsesShellOutcome,
            ResponsesShellOutputChunk, ResponsesToolCaller, ResponsesToolOutput,
            ResponsesToolSearchCall, ResponsesToolSearchOutput, ResponsesWebSearchAction,
        },
    };

    use rstest::rstest;

    use serde_json::{Value, json};

    #[rstest]
    #[case::message(json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[{"type":"url_citation","url":"https://example.test","start_index":0,"end_index":6}]}]}))]
    #[case::function_call(json!({"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"query\":\"q\"}"}))]
    #[case::custom_tool(json!({"type":"custom_tool_call","name":"lookup","input":"q"}))]
    #[case::reasoning(json!({"type":"reasoning","summary":[{"type":"summary_text","text":"summary"}],"encrypted_content":"opaque"}))]
    #[case::web_search(json!({"type":"web_search_call","action":{"type":"search","queries":["q"],"sources":[{"type":"url","url":"https://example.test"}]}}))]
    #[case::file_search(json!({"type":"file_search_call","queries":["q"],"results":[{"file_id":"file_1","score":1,"attributes":{"custom":[1,null]}}]}))]
    #[case::code(json!({"type":"code_interpreter_call","outputs":[{"type":"logs","logs":"done"},{"type":"image","url":"https://example.test"}]}))]
    #[case::image(json!({"type":"image_generation_call","result":"generated"}))]
    #[case::mcp(json!({"type":"mcp_call","server_label":"server","name":"lookup","arguments":"{}","output":"done"}))]
    fn output_items_round_trip(#[case] wire: Value) {
        let item: ResponsesOutputItem = serde_json::from_value(wire.clone()).unwrap();
        match &item {
            ResponsesOutputItem::Message(message) => {
                assert_eq!(message.role.as_deref(), Some("assistant"));
                let Some(content) = &message.content else {
                    panic!("expected content")
                };
                let [
                    ResponsesContentPart::OutputText {
                        text,
                        annotations: Some(annotations),
                        ..
                    },
                ] = content.as_slice()
                else {
                    panic!("expected output text and annotations")
                };
                assert_eq!(text, "answer");
                let [crate::formats::responses::ResponsesAnnotation::UrlCitation(citation)] =
                    annotations.as_slice()
                else {
                    panic!("expected URL citation")
                };
                assert_eq!(citation.url.as_deref(), Some("https://example.test"));
                assert_eq!(citation.start_index, Some(0));
            }
            ResponsesOutputItem::FunctionCall(call) => {
                assert_eq!(call.call_id.as_deref(), Some("call_1"));
                assert_eq!(call.name.as_deref(), Some("lookup"));
                assert_eq!(call.arguments.as_deref(), Some("{\"query\":\"q\"}"));
            }
            ResponsesOutputItem::CustomToolCall(call) => {
                assert_eq!(call.name.as_deref(), Some("lookup"));
                assert_eq!(call.input.as_deref(), Some("q"));
            }
            ResponsesOutputItem::Reasoning(reasoning) => {
                assert_eq!(reasoning.encrypted_content.as_deref(), Some("opaque"));
                let Some(summary) = &reasoning.summary else {
                    panic!("expected summary")
                };
                let [ResponsesContentPart::SummaryText { text, .. }] = summary.as_slice() else {
                    panic!("expected summary text")
                };
                assert_eq!(text, "summary");
            }
            ResponsesOutputItem::WebSearchCall(call) => {
                let Some(ResponsesWebSearchAction::Search {
                    queries: Some(queries),
                    sources: Some(sources),
                    ..
                }) = &call.action
                else {
                    panic!("expected search action")
                };
                assert_eq!(queries, &["q"]);
                assert_eq!(sources[0].url.as_deref(), Some("https://example.test"));
            }
            ResponsesOutputItem::FileSearchCall(call) => {
                assert_eq!(call.queries.as_deref(), Some(["q".to_owned()].as_slice()));
                let Some(results) = &call.results else {
                    panic!("expected search results")
                };
                assert_eq!(results[0].file_id.as_deref(), Some("file_1"));
                assert_eq!(results[0].score, Some(1.into()));
                assert_eq!(
                    results[0].attributes.as_ref().unwrap()["custom"],
                    json!([1, null])
                );
            }
            ResponsesOutputItem::CodeInterpreterCall(call) => {
                let Some(outputs) = &call.outputs else {
                    panic!("expected code outputs")
                };
                let [
                    ResponsesCodeOutput::Logs { logs, .. },
                    ResponsesCodeOutput::Image { url, .. },
                ] = outputs.as_slice()
                else {
                    panic!("expected logs and image")
                };
                assert_eq!(logs, "done");
                assert_eq!(url, "https://example.test");
            }
            ResponsesOutputItem::ImageGenerationCall(call) => {
                assert_eq!(call.result.as_deref(), Some("generated"))
            }
            ResponsesOutputItem::McpCall(call) => {
                assert_eq!(call.server_label.as_deref(), Some("server"));
                assert_eq!(call.name.as_deref(), Some("lookup"));
                assert_eq!(call.arguments.as_deref(), Some("{}"));
                assert_eq!(call.output.as_deref(), Some("done"));
            }
            other => panic!("unexpected variant {other:?}"),
        }
        assert_eq!(serde_json::to_value(item).unwrap(), wire);
    }

    #[rstest]
    #[case::find_in_page(
        json!({"type":"find_in_page","url":"https://example.test","pattern":"needle"}),
        json!({"type":"find_in_page","url":"https://example.test","pattern":"needle"})
    )]
    #[case::legacy_find(
        json!({"type":"find","url":"https://example.test","pattern":"needle"}),
        json!({"type":"find_in_page","url":"https://example.test","pattern":"needle"})
    )]
    fn web_search_find_action_exposes_url_and_pattern(
        #[case] wire: Value,
        #[case] serialized: Value,
    ) {
        let action: ResponsesWebSearchAction = serde_json::from_value(wire).unwrap();
        let ResponsesWebSearchAction::FindInPage {
            url,
            pattern,
            extra,
        } = &action
        else {
            panic!("expected find_in_page action");
        };
        assert_eq!(url, "https://example.test");
        assert_eq!(pattern, "needle");
        assert!(extra.is_empty());
        assert_eq!(serde_json::to_value(action).unwrap(), serialized);
    }

    #[rstest]
    #[case::with_url(json!({"type":"open_page","url":"https://example.test"}), Some("https://example.test"))]
    #[case::without_url(json!({"type":"open_page"}), None)]
    fn web_search_open_page_url_is_optional(#[case] wire: Value, #[case] expected: Option<&str>) {
        let action: ResponsesWebSearchAction = serde_json::from_value(wire.clone()).unwrap();
        let ResponsesWebSearchAction::OpenPage { url, .. } = &action else {
            panic!("expected open_page action");
        };
        assert_eq!(url.as_deref(), expected);
        assert_eq!(serde_json::to_value(action).unwrap(), wire);
    }

    #[rstest]
    #[case::find_pattern(json!({"type":"find_in_page","url":"u","pattern":false}))]
    #[case::find_missing_url(json!({"type":"find_in_page","pattern":"p"}))]
    #[case::open_page_url(json!({"type":"open_page","url":7}))]
    fn web_search_action_rejects_malformed_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<ResponsesWebSearchAction>(wire).is_err());
    }

    #[rstest]
    #[case::refusal(json!({"type":"refusal","refusal":"refused","future":null}), ResponsesContentPart::Refusal { refusal:"refused".into(), extra:serde_json::Map::from_iter([("future".into(), Value::Null)]) })]
    #[case::reasoning(json!({"type":"reasoning_text","text":"reason"}), ResponsesContentPart::ReasoningText { text:"reason".into(), extra:Default::default() })]
    fn content_parts_expose_typed_variants(
        #[case] wire: Value,
        #[case] expected: ResponsesContentPart,
    ) {
        let content: ResponsesContentPart = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(content, expected);
        assert_eq!(serde_json::to_value(content).unwrap(), wire);
    }

    #[rstest]
    fn mcp_discovery_exposes_tools_and_nested_schemas() {
        let wire = json!({
            "type":"mcp_list_tools","id":"item_1","server_label":"tools",
            "tools":[{"name":"lookup","description":"Look up a value","input_schema":{"type":"object","properties":{"query":{"type":"string"}}},"annotations":{"read_only":false},"future":null}],
            "extension":[1,null]
        });
        let item: ResponsesOutputItem = serde_json::from_value(wire.clone()).unwrap();
        let ResponsesOutputItem::McpListTools(discovery) = &item else {
            panic!("expected discovery")
        };
        assert_eq!(discovery.server_label.as_deref(), Some("tools"));
        let tool = &discovery.tools.as_ref().unwrap()[0];
        assert_eq!(tool.name.as_deref(), Some("lookup"));
        let Some(crate::json_schema::JsonSchema::Object(schema)) = &tool.input_schema else {
            panic!("expected tool schema")
        };
        assert!(schema.properties.as_ref().unwrap().contains_key("query"));
        assert_eq!(tool.annotations, Some(json!({"read_only":false})));
        assert_eq!(tool.extra["future"], Value::Null);
        assert_eq!(serde_json::to_value(item).unwrap(), wire);
    }

    #[rstest]
    #[case::partial(json!({"server_label":"tools","tools":[]}))]
    #[case::failure(json!({"server_label":"tools","error":"unavailable"}))]
    fn mcp_discovery_accepts_partial_payloads(#[case] wire: Value) {
        let discovery: crate::formats::responses::ResponsesMcpListTools =
            serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(discovery.server_label.as_deref(), Some("tools"));
        assert!(discovery.id.is_none());
        assert_eq!(serde_json::to_value(discovery).unwrap(), wire);
    }

    #[rstest]
    #[case::message(json!("unavailable"), ResponsesMcpError::Message("unavailable".into()))]
    #[case::protocol(json!({"type":"mcp_protocol_error","code":-32600,"message":"invalid","extension":null}), ResponsesMcpError::Detail(ResponsesMcpErrorDetail::McpProtocolError {code:-32600,message:"invalid".into(),extra:serde_json::Map::from_iter([("extension".into(), Value::Null)])}))]
    #[case::http(json!({"type":"http_error","code":503,"message":"unavailable"}), ResponsesMcpError::Detail(ResponsesMcpErrorDetail::HttpError {code:503,message:"unavailable".into(),extra:Default::default()}))]
    #[case::tool(json!({"type":"mcp_tool_execution_error","content":{"arbitrary":[1,null]}}), ResponsesMcpError::Detail(ResponsesMcpErrorDetail::McpToolExecutionError {content:json!({"arbitrary":[1,null]}),extra:Default::default()}))]
    fn mcp_call_errors_expose_typed_variants(
        #[case] wire: Value,
        #[case] expected: ResponsesMcpError,
    ) {
        let error: ResponsesMcpError = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(error, expected);
        assert_eq!(serde_json::to_value(&error).unwrap(), wire);
        let call: ResponsesOutputItem =
            serde_json::from_value(json!({"type":"mcp_call","error":wire})).unwrap();
        let ResponsesOutputItem::McpCall(call) = call else {
            panic!("expected call")
        };
        assert_eq!(call.error, Some(error));
    }

    #[rstest]
    #[case::tools_not_array(json!({"type":"mcp_list_tools","tools":{}}))]
    #[case::invalid_name(json!({"type":"mcp_list_tools","tools":[{"name":7}]}))]
    #[case::invalid_schema(json!({"type":"mcp_list_tools","tools":[{"input_schema":{"properties":{"query":7}}}]}))]
    #[case::invalid_error_code(json!({"type":"mcp_call","error":{"type":"http_error","code":"503","message":"unavailable"}}))]
    #[case::missing_error_content(json!({"type":"mcp_call","error":{"type":"mcp_tool_execution_error"}}))]
    #[case::unknown_error_tag(json!({"type":"mcp_call","error":{"type":"future"}}))]
    fn mcp_output_rejects_malformed_known_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<ResponsesOutputItem>(wire).is_err());
    }

    fn program_caller() -> Option<ResponsesToolCaller> {
        Some(ResponsesToolCaller::Program {
            caller_id: "prog_1".into(),
            extra: Default::default(),
        })
    }

    fn direct_caller() -> Option<ResponsesToolCaller> {
        Some(ResponsesToolCaller::Direct {
            extra: Default::default(),
        })
    }

    fn safety_check() -> ResponsesSafetyCheck {
        ResponsesSafetyCheck {
            id: text("sc_1"),
            code: text("malicious_instructions"),
            message: text("check"),
            ..Default::default()
        }
    }

    #[rstest]
    #[case::function_call(
        json!({"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"lookup","arguments":"{}","namespace":"ns","async":true,"caller":{"type":"program","caller_id":"prog_1"}}),
        ResponsesOutputItem::FunctionCall(ResponsesFunctionCall {
            id: text("fc_1"), status: text("completed"), call_id: text("call_1"), name: text("lookup"),
            arguments: text("{}"), namespace: text("ns"), r#async: Some(true), caller: program_caller(), ..Default::default()
        })
    )]
    #[case::custom_tool_call(
        json!({"type":"custom_tool_call","call_id":"call_1","name":"lookup","input":"q","namespace":"ns","async":false,"caller":{"type":"direct"}}),
        ResponsesOutputItem::CustomToolCall(ResponsesCustomToolCall {
            call_id: text("call_1"), name: text("lookup"), input: text("q"), namespace: text("ns"),
            r#async: Some(false), caller: direct_caller(), ..Default::default()
        })
    )]
    #[case::image_generation_call(
        json!({"type":"image_generation_call","id":"ig_1","status":"completed","result":"b64","action":"edit","background":"opaque","output_format":"webp","quality":"high","revised_prompt":"a cat","size":"1536x864"}),
        ResponsesOutputItem::ImageGenerationCall(ResponsesImageGenerationCall {
            id: text("ig_1"), status: text("completed"), result: text("b64"), action: text("edit"), background: text("opaque"),
            output_format: text("webp"), quality: text("high"), revised_prompt: text("a cat"), size: text("1536x864"), ..Default::default()
        })
    )]
    #[case::function_call_output_text(
        json!({"type":"function_call_output","id":"fco_1","status":"completed","call_id":"call_1","output":"done","caller":{"type":"direct"},"created_by":"user_1","name":"lookup","namespace":"ns"}),
        ResponsesOutputItem::FunctionCallOutput(ResponsesFunctionCallOutput {
            id: text("fco_1"), status: text("completed"), call_id: text("call_1"), output: Some(ResponsesToolOutput::Text("done".into())),
            caller: direct_caller(), created_by: text("user_1"), name: text("lookup"), namespace: text("ns"), ..Default::default()
        })
    )]
    #[case::function_call_output_content(
        json!({"type":"function_call_output","call_id":"call_1","output":[
            {"type":"input_text","text":"t","prompt_cache_breakpoint":{"mode":"explicit"}},
            {"type":"input_image","detail":"low","file_id":"file_1","image_url":"https://example.test/i.png"},
            {"type":"input_file","detail":"high","file_data":"data","file_id":"file_2","file_url":"https://example.test/f","filename":"f.pdf"}
        ]}),
        ResponsesOutputItem::FunctionCallOutput(ResponsesFunctionCallOutput {
            call_id: text("call_1"),
            output: Some(ResponsesToolOutput::Content(vec![
                ResponsesInputContent::InputText {
                    text: "t".into(),
                    prompt_cache_breakpoint: Some(PromptCacheBreakpoint { mode: PromptCacheMode::Explicit, extra: Default::default() }),
                    extra: Default::default(),
                },
                ResponsesInputContent::InputImage {
                    detail: "low".into(), file_id: text("file_1"), image_url: text("https://example.test/i.png"),
                    prompt_cache_breakpoint: None, extra: Default::default(),
                },
                ResponsesInputContent::InputFile {
                    detail: text("high"), file_data: text("data"), file_id: text("file_2"), file_url: text("https://example.test/f"),
                    filename: text("f.pdf"), prompt_cache_breakpoint: None, extra: Default::default(),
                },
            ])),
            ..Default::default()
        })
    )]
    #[case::custom_tool_call_output(
        json!({"type":"custom_tool_call_output","id":"cto_1","status":"completed","call_id":"call_1","output":[{"type":"input_text","text":"t"}],"caller":{"type":"program","caller_id":"prog_1"},"created_by":"user_1"}),
        ResponsesOutputItem::CustomToolCallOutput(ResponsesCustomToolCallOutput {
            id: text("cto_1"), status: text("completed"), call_id: text("call_1"),
            output: Some(ResponsesToolOutput::Content(vec![ResponsesInputContent::InputText {
                text: "t".into(), prompt_cache_breakpoint: None, extra: Default::default(),
            }])),
            caller: program_caller(), created_by: text("user_1"), ..Default::default()
        })
    )]
    #[case::computer_call(
        json!({"type":"computer_call","id":"cu_1","status":"completed","call_id":"call_1",
            "pending_safety_checks":[{"id":"sc_1","code":"malicious_instructions","message":"check"}],
            "action":{"type":"click","button":"left","x":1,"y":2,"keys":["shift"]},
            "actions":[
                {"type":"double_click","x":3,"y":4},
                {"type":"drag","path":[{"x":5,"y":6},{"x":7,"y":8}],"keys":["ctrl"]},
                {"type":"keypress","keys":["enter"]},
                {"type":"move","x":-1,"y":9},
                {"type":"screenshot"},
                {"type":"scroll","scroll_x":0,"scroll_y":-10,"x":11,"y":12},
                {"type":"type","text":"hello"},
                {"type":"wait"}
            ]}),
        ResponsesOutputItem::ComputerCall(ResponsesComputerCall {
            id: text("cu_1"), status: text("completed"), call_id: text("call_1"), pending_safety_checks: Some(vec![safety_check()]),
            action: Some(ResponsesComputerAction::Click { button: "left".into(), x: 1, y: 2, keys: Some(vec!["shift".into()]), extra: Default::default() }),
            actions: Some(vec![
                ResponsesComputerAction::DoubleClick { x: 3, y: 4, keys: None, extra: Default::default() },
                ResponsesComputerAction::Drag {
                    path: vec![
                        ResponsesCoordinate { x: 5, y: 6, extra: Default::default() },
                        ResponsesCoordinate { x: 7, y: 8, extra: Default::default() },
                    ],
                    keys: Some(vec!["ctrl".into()]),
                    extra: Default::default(),
                },
                ResponsesComputerAction::Keypress { keys: vec!["enter".into()], extra: Default::default() },
                ResponsesComputerAction::Move { x: -1, y: 9, keys: None, extra: Default::default() },
                ResponsesComputerAction::Screenshot { extra: Default::default() },
                ResponsesComputerAction::Scroll { scroll_x: 0, scroll_y: -10, x: 11, y: 12, keys: None, extra: Default::default() },
                ResponsesComputerAction::Type { text: "hello".into(), extra: Default::default() },
                ResponsesComputerAction::Wait { extra: Default::default() },
            ]),
            ..Default::default()
        })
    )]
    #[case::computer_call_output(
        json!({"type":"computer_call_output","id":"cuo_1","status":"completed","call_id":"call_1","output":{"type":"computer_screenshot","file_id":"file_1","image_url":"https://example.test/s.png"},"acknowledged_safety_checks":[{"id":"sc_1","code":"malicious_instructions","message":"check"}],"created_by":"user_1"}),
        ResponsesOutputItem::ComputerCallOutput(ResponsesComputerCallOutput {
            id: text("cuo_1"), status: text("completed"), call_id: text("call_1"),
            output: Some(ResponsesComputerOutput::ComputerScreenshot { file_id: text("file_1"), image_url: text("https://example.test/s.png"), extra: Default::default() }),
            acknowledged_safety_checks: Some(vec![safety_check()]), created_by: text("user_1"), ..Default::default()
        })
    )]
    #[case::program(
        json!({"type":"program","id":"prog_item","call_id":"prog_1","code":"run()","fingerprint":"fp"}),
        ResponsesOutputItem::Program(ResponsesProgram {
            id: text("prog_item"), call_id: text("prog_1"), code: text("run()"), fingerprint: text("fp"), ..Default::default()
        })
    )]
    #[case::program_output(
        json!({"type":"program_output","id":"po_1","status":"completed","call_id":"prog_1","result":"42"}),
        ResponsesOutputItem::ProgramOutput(ResponsesProgramOutput {
            id: text("po_1"), status: text("completed"), call_id: text("prog_1"), result: text("42"), ..Default::default()
        })
    )]
    #[case::tool_search_call(
        json!({"type":"tool_search_call","id":"ts_1","status":"completed","call_id":"call_1","arguments":{"query":["weather",null]},"execution":"server","created_by":"user_1"}),
        ResponsesOutputItem::ToolSearchCall(ResponsesToolSearchCall {
            id: text("ts_1"), status: text("completed"), call_id: text("call_1"), arguments: Some(json!({"query":["weather",null]})),
            execution: text("server"), created_by: text("user_1"), ..Default::default()
        })
    )]
    #[case::tool_search_output(
        json!({"type":"tool_search_output","id":"tso_1","status":"completed","call_id":"call_1","execution":"client","tools":[{"type":"function","name":"lookup","parameters":null,"strict":true}],"created_by":"user_1"}),
        ResponsesOutputItem::ToolSearchOutput(ResponsesToolSearchOutput {
            id: text("tso_1"), status: text("completed"), call_id: text("call_1"), execution: text("client"),
            tools: Some(vec![serde_json::Map::from_iter([
                ("type".into(), json!("function")), ("name".into(), json!("lookup")),
                ("parameters".into(), Value::Null), ("strict".into(), json!(true)),
            ])]),
            created_by: text("user_1"), ..Default::default()
        })
    )]
    #[case::additional_tools(
        json!({"type":"additional_tools","id":"at_1","role":"developer","tools":[{"type":"local_shell"}]}),
        ResponsesOutputItem::AdditionalTools(ResponsesAdditionalTools {
            id: text("at_1"), role: text("developer"),
            tools: Some(vec![serde_json::Map::from_iter([("type".into(), json!("local_shell"))])]), ..Default::default()
        })
    )]
    #[case::compaction(
        json!({"type":"compaction","id":"cmp_1","encrypted_content":"opaque","created_by":"user_1"}),
        ResponsesOutputItem::Compaction(ResponsesCompaction {
            id: text("cmp_1"), encrypted_content: text("opaque"), created_by: text("user_1"), ..Default::default()
        })
    )]
    #[case::local_shell_call(
        json!({"type":"local_shell_call","id":"ls_1","status":"completed","call_id":"call_1","action":{"type":"exec","command":["ls","-a"],"env":{"HOME":"/home/u"},"timeout_ms":1000,"user":"u","working_directory":"/tmp"}}),
        ResponsesOutputItem::LocalShellCall(ResponsesLocalShellCall {
            id: text("ls_1"), status: text("completed"), call_id: text("call_1"),
            action: Some(ResponsesLocalShellAction::Exec {
                command: vec!["ls".into(), "-a".into()], env: [("HOME".into(), "/home/u".into())].into(),
                timeout_ms: Some(1000), user: text("u"), working_directory: text("/tmp"), extra: Default::default(),
            }),
            ..Default::default()
        })
    )]
    #[case::local_shell_call_output(
        json!({"type":"local_shell_call_output","id":"lso_1","status":"completed","output":"{\"stdout\":\"x\"}"}),
        ResponsesOutputItem::LocalShellCallOutput(ResponsesLocalShellCallOutput {
            id: text("lso_1"), status: text("completed"), output: text("{\"stdout\":\"x\"}"), ..Default::default()
        })
    )]
    #[case::shell_call_local(
        json!({"type":"shell_call","id":"sh_1","status":"in_progress","call_id":"call_1","action":{"commands":["ls"],"max_output_length":100,"timeout_ms":500},"environment":{"type":"local"},"caller":{"type":"direct"},"created_by":"user_1"}),
        ResponsesOutputItem::ShellCall(ResponsesShellCall {
            id: text("sh_1"), status: text("in_progress"), call_id: text("call_1"),
            action: Some(ResponsesShellAction { commands: Some(vec!["ls".into()]), max_output_length: Some(100), timeout_ms: Some(500), ..Default::default() }),
            environment: Some(ResponsesShellEnvironment::Local { extra: Default::default() }),
            caller: direct_caller(), created_by: text("user_1"), ..Default::default()
        })
    )]
    #[case::shell_call_container(
        json!({"type":"shell_call","call_id":"call_1","environment":{"type":"container_reference","container_id":"cntr_1"}}),
        ResponsesOutputItem::ShellCall(ResponsesShellCall {
            call_id: text("call_1"),
            environment: Some(ResponsesShellEnvironment::ContainerReference { container_id: "cntr_1".into(), extra: Default::default() }),
            ..Default::default()
        })
    )]
    #[case::shell_call_output(
        json!({"type":"shell_call_output","id":"sho_1","status":"completed","call_id":"call_1","max_output_length":100,"output":[
            {"outcome":{"type":"exit","exit_code":2},"stdout":"out","stderr":"err","created_by":"user_1"},
            {"outcome":{"type":"timeout"},"stdout":"","stderr":""}
        ],"caller":{"type":"program","caller_id":"prog_1"},"created_by":"user_1"}),
        ResponsesOutputItem::ShellCallOutput(ResponsesShellCallOutput {
            id: text("sho_1"), status: text("completed"), call_id: text("call_1"), max_output_length: Some(100),
            output: Some(vec![
                ResponsesShellOutputChunk {
                    outcome: Some(ResponsesShellOutcome::Exit { exit_code: 2, extra: Default::default() }),
                    stdout: text("out"), stderr: text("err"), created_by: text("user_1"), ..Default::default()
                },
                ResponsesShellOutputChunk {
                    outcome: Some(ResponsesShellOutcome::Timeout { extra: Default::default() }),
                    stdout: text(""), stderr: text(""), ..Default::default()
                },
            ]),
            caller: program_caller(), created_by: text("user_1"), ..Default::default()
        })
    )]
    #[case::apply_patch_create(
        json!({"type":"apply_patch_call","id":"ap_1","status":"completed","call_id":"call_1","operation":{"type":"create_file","path":"a.txt","diff":"+a"},"caller":{"type":"direct"},"created_by":"user_1"}),
        ResponsesOutputItem::ApplyPatchCall(ResponsesApplyPatchCall {
            id: text("ap_1"), status: text("completed"), call_id: text("call_1"),
            operation: Some(ResponsesApplyPatchOperation::CreateFile { path: "a.txt".into(), diff: "+a".into(), extra: Default::default() }),
            caller: direct_caller(), created_by: text("user_1"), ..Default::default()
        })
    )]
    #[case::apply_patch_delete(
        json!({"type":"apply_patch_call","call_id":"call_1","operation":{"type":"delete_file","path":"a.txt"}}),
        ResponsesOutputItem::ApplyPatchCall(ResponsesApplyPatchCall {
            call_id: text("call_1"),
            operation: Some(ResponsesApplyPatchOperation::DeleteFile { path: "a.txt".into(), extra: Default::default() }),
            ..Default::default()
        })
    )]
    #[case::apply_patch_update(
        json!({"type":"apply_patch_call","call_id":"call_1","operation":{"type":"update_file","path":"a.txt","diff":"-a\n+b"}}),
        ResponsesOutputItem::ApplyPatchCall(ResponsesApplyPatchCall {
            call_id: text("call_1"),
            operation: Some(ResponsesApplyPatchOperation::UpdateFile { path: "a.txt".into(), diff: "-a\n+b".into(), extra: Default::default() }),
            ..Default::default()
        })
    )]
    #[case::apply_patch_call_output(
        json!({"type":"apply_patch_call_output","id":"apo_1","status":"failed","call_id":"call_1","output":"conflict","caller":{"type":"direct"},"created_by":"user_1"}),
        ResponsesOutputItem::ApplyPatchCallOutput(ResponsesApplyPatchCallOutput {
            id: text("apo_1"), status: text("failed"), call_id: text("call_1"), output: text("conflict"),
            caller: direct_caller(), created_by: text("user_1"), ..Default::default()
        })
    )]
    #[case::mcp_approval_request(
        json!({"type":"mcp_approval_request","id":"apr_1","server_label":"s","name":"n","arguments":"{}"}),
        ResponsesOutputItem::McpApprovalRequest(ResponsesMcpApprovalRequest {
            id: text("apr_1"), server_label: text("s"), name: text("n"), arguments: text("{}"), ..Default::default()
        })
    )]
    #[case::mcp_approval_response(
        json!({"type":"mcp_approval_response","id":"aprr_1","approval_request_id":"apr_1","approve":false,"reason":"denied"}),
        ResponsesOutputItem::McpApprovalResponse(ResponsesMcpApprovalResponse {
            id: text("aprr_1"), approval_request_id: text("apr_1"), approve: Some(false), reason: text("denied"), ..Default::default()
        })
    )]
    fn documented_output_items_parse_into_typed_variants(
        #[case] wire: Value,
        #[case] expected: ResponsesOutputItem,
    ) {
        let item: ResponsesOutputItem = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(item, expected);
        assert_eq!(serde_json::to_value(item).unwrap(), wire);
    }

    #[rstest]
    #[case::unknown_caller(json!({"type":"function_call","caller":{"type":"future"}}))]
    #[case::program_caller_missing_id(json!({"type":"shell_call","caller":{"type":"program"}}))]
    #[case::untagged_caller(json!({"type":"custom_tool_call","caller":{}}))]
    #[case::async_not_bool(json!({"type":"function_call","async":"yes"}))]
    #[case::output_not_text_or_list(json!({"type":"function_call_output","output":{"text":"t"}}))]
    #[case::unknown_input_content(json!({"type":"function_call_output","output":[{"type":"input_audio"}]}))]
    #[case::input_text_missing_text(json!({"type":"custom_tool_call_output","output":[{"type":"input_text"}]}))]
    #[case::input_image_missing_detail(json!({"type":"custom_tool_call_output","output":[{"type":"input_image","file_id":"f"}]}))]
    #[case::bad_cache_mode(json!({"type":"custom_tool_call_output","output":[{"type":"input_text","text":"t","prompt_cache_breakpoint":{"mode":"implicit"}}]}))]
    #[case::unknown_computer_action(json!({"type":"computer_call","action":{"type":"teleport"}}))]
    #[case::click_missing_button(json!({"type":"computer_call","action":{"type":"click","x":1,"y":2}}))]
    #[case::click_fractional_x(json!({"type":"computer_call","action":{"type":"click","button":"left","x":1.5,"y":2}}))]
    #[case::drag_point_missing_y(json!({"type":"computer_call","actions":[{"type":"drag","path":[{"x":1}]}]}))]
    #[case::keypress_keys_not_list(json!({"type":"computer_call","actions":[{"type":"keypress","keys":"enter"}]}))]
    #[case::safety_check_not_object(json!({"type":"computer_call","pending_safety_checks":["sc_1"]}))]
    #[case::unknown_screenshot_tag(json!({"type":"computer_call_output","output":{"type":"screenshot"}}))]
    #[case::untagged_screenshot(json!({"type":"computer_call_output","output":{"file_id":"f"}}))]
    #[case::tool_not_object(json!({"type":"tool_search_output","tools":["lookup"]}))]
    #[case::unknown_local_shell_action(json!({"type":"local_shell_call","action":{"type":"spawn","command":[],"env":{}}}))]
    #[case::local_shell_missing_env(json!({"type":"local_shell_call","action":{"type":"exec","command":["ls"]}}))]
    #[case::local_shell_env_value(json!({"type":"local_shell_call","action":{"type":"exec","command":["ls"],"env":{"A":1}}}))]
    #[case::shell_commands_not_list(json!({"type":"shell_call","action":{"commands":"ls"}}))]
    #[case::unknown_shell_environment(json!({"type":"shell_call","environment":{"type":"container_auto"}}))]
    #[case::container_missing_id(json!({"type":"shell_call","environment":{"type":"container_reference"}}))]
    #[case::unknown_shell_outcome(json!({"type":"shell_call_output","output":[{"outcome":{"type":"killed"}}]}))]
    #[case::exit_missing_code(json!({"type":"shell_call_output","output":[{"outcome":{"type":"exit"}}]}))]
    #[case::negative_max_output(json!({"type":"shell_call_output","max_output_length":-1}))]
    #[case::unknown_patch_operation(json!({"type":"apply_patch_call","operation":{"type":"rename_file","path":"a"}}))]
    #[case::update_missing_diff(json!({"type":"apply_patch_call","operation":{"type":"update_file","path":"a"}}))]
    #[case::approve_not_bool(json!({"type":"mcp_approval_response","approve":"true"}))]
    #[case::compaction_content_not_string(json!({"type":"compaction","encrypted_content":7}))]
    #[case::program_code_not_string(json!({"type":"program","code":["run()"]}))]
    fn output_items_reject_malformed_typed_fields(#[case] wire: Value) {
        assert!(serde_json::from_value::<ResponsesOutputItem>(wire).is_err());
    }

    fn text(value: &str) -> Option<String> {
        Some(value.to_owned())
    }
}
