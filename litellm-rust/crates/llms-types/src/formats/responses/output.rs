use serde_json::{Map, Value};

#[macro_rules_attribute::apply(wire_type)]
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
}

#[macro_rules_attribute::apply(wire_type)]
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

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesAnnotation {
    UrlCitation(ResponsesUrlCitation),
    FileCitation(ResponsesFileCitation),
    FilePath(ResponsesFileCitation),
    ContainerFileCitation(ResponsesFileCitation),
}

#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesFunctionCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
    pub phase: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesCustomToolCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub call_id: Option<String>,
    pub name: Option<String>,
    pub input: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesWebSearchCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub action: Option<ResponsesWebSearchAction>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesImageGenerationCall {
    pub id: Option<String>,
    pub status: Option<String>,
    pub result: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
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

#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesWebSearchSource {
    #[serde(rename = "type")]
    pub source_type: Option<String>,
    pub url: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
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
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesMcpTool {
    pub name: Option<String>,
    pub description: Option<String>,
    pub input_schema: Option<crate::json_schema::JsonSchema>,
    pub annotations: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(untagged)]
pub enum ResponsesMcpError {
    Message(String),
    Detail(ResponsesMcpErrorDetail),
}

#[macro_rules_attribute::apply(wire_type)]
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
