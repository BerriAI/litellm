use std::collections::BTreeMap;

use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct ProviderFields(pub BTreeMap<String, Value>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Incomplete,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponsesUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub input_token_details: ProviderFields,
    pub output_token_details: ProviderFields,
    pub extensions: ProviderFields,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponsesError {
    pub code: Option<String>,
    pub message: String,
    pub param: Option<String>,
    pub extensions: ProviderFields,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponseAnnotation {
    pub kind: String,
    pub start_index: Option<u64>,
    pub end_index: Option<u64>,
    pub url: Option<String>,
    pub title: Option<String>,
    pub extensions: ProviderFields,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseContentPart {
    OutputText {
        text: String,
        annotations: Box<[ResponseAnnotation]>,
    },
    Refusal {
        refusal: String,
    },
    SummaryText {
        text: String,
    },
    Extension {
        kind: String,
        fields: ProviderFields,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseOutputItem {
    Message {
        id: String,
        status: ResponseStatus,
        role: String,
        content: Box<[ResponseContentPart]>,
        phase: Option<String>,
        extensions: ProviderFields,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        namespace: Option<String>,
        arguments: String,
        status: ResponseStatus,
        extensions: ProviderFields,
    },
    CustomToolCall {
        id: String,
        call_id: String,
        name: String,
        input: String,
        status: Option<ResponseStatus>,
        extensions: ProviderFields,
    },
    Reasoning {
        id: String,
        summary: Box<[ResponseContentPart]>,
        encrypted_content: Option<String>,
        extensions: ProviderFields,
    },
    Extension {
        kind: String,
        id: Option<String>,
        fields: ProviderFields,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseInputContent {
    Text(String),
    Image {
        image_url: Option<String>,
        file_id: Option<String>,
        detail: Option<String>,
    },
    File {
        file_id: Option<String>,
        file_data: Option<String>,
        filename: Option<String>,
    },
    Extension {
        kind: String,
        fields: ProviderFields,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseInputItem {
    Message {
        role: String,
        content: Box<[ResponseInputContent]>,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
    CustomToolCallOutput {
        call_id: String,
        output: String,
    },
    ItemReference {
        id: String,
    },
    Output(ResponseOutputItem),
    Extension {
        kind: String,
        fields: ProviderFields,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponsesInput {
    Text(String),
    Items(Box<[ResponseInputItem]>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseTool {
    Function {
        name: String,
        description: Option<String>,
        parameters: ProviderFields,
        strict: Option<bool>,
    },
    Custom {
        name: String,
        description: Option<String>,
        format: ProviderFields,
    },
    Namespace {
        name: String,
        description: Option<String>,
        tools: Box<[ResponseTool]>,
    },
    BuiltIn {
        kind: String,
        options: ProviderFields,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    pub input: Option<ResponsesInput>,
    pub previous_response_id: Option<String>,
    pub instructions: Option<String>,
    pub tools: Box<[ResponseTool]>,
    pub parallel_tool_calls: Option<bool>,
    pub max_output_tokens: Option<u64>,
    pub store: Option<bool>,
    pub background: Option<bool>,
    pub stream: Option<bool>,
    pub include: Box<[String]>,
    pub metadata: BTreeMap<String, String>,
    // TODO: Type remaining Python request options before enabling wire decoding; preserve them losslessly meanwhile
    pub extensions: ProviderFields,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponsesResponse {
    pub id: String,
    pub created_at: u64,
    pub model: Option<String>,
    pub status: Option<ResponseStatus>,
    pub output: Box<[ResponseOutputItem]>,
    pub usage: Option<ResponsesUsage>,
    pub error: Option<ResponsesError>,
    pub incomplete_reason: Option<String>,
    pub previous_response_id: Option<String>,
    pub extensions: ProviderFields,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponsesEvent {
    pub sequence_number: Option<u64>,
    pub data: ResponsesEventData,
    pub extensions: ProviderFields,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponsesEventData {
    Created(ResponsesResponse),
    InProgress(ResponsesResponse),
    Completed(ResponsesResponse),
    Failed(ResponsesResponse),
    Incomplete(ResponsesResponse),
    OutputItemAdded {
        output_index: u64,
        item: ResponseOutputItem,
    },
    OutputItemDone {
        output_index: u64,
        item: ResponseOutputItem,
    },
    ContentPartAdded {
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: ResponseContentPart,
    },
    ContentPartDone {
        item_id: String,
        output_index: u64,
        content_index: u64,
        part: ResponseContentPart,
    },
    OutputTextDelta {
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
    },
    OutputTextDone {
        item_id: String,
        output_index: u64,
        content_index: u64,
        text: String,
    },
    AnnotationAdded {
        item_id: String,
        output_index: u64,
        content_index: u64,
        annotation_index: u64,
        annotation: ResponseAnnotation,
    },
    RefusalDelta {
        item_id: String,
        output_index: u64,
        content_index: u64,
        delta: String,
    },
    RefusalDone {
        item_id: String,
        output_index: u64,
        content_index: u64,
        refusal: String,
    },
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: u64,
        delta: String,
    },
    FunctionCallArgumentsDone {
        item_id: String,
        output_index: u64,
        arguments: String,
    },
    CustomToolCallInputDelta {
        item_id: String,
        output_index: u64,
        delta: String,
    },
    CustomToolCallInputDone {
        item_id: String,
        output_index: u64,
        input: String,
    },
    ReasoningSummaryPartAdded {
        item_id: String,
        output_index: u64,
        summary_index: u64,
        part: ResponseContentPart,
    },
    ReasoningSummaryPartDone {
        item_id: String,
        output_index: u64,
        summary_index: u64,
        part: ResponseContentPart,
    },
    ReasoningSummaryTextDelta {
        item_id: String,
        output_index: u64,
        summary_index: u64,
        delta: String,
    },
    ReasoningSummaryTextDone {
        item_id: String,
        output_index: u64,
        summary_index: u64,
        text: String,
    },
    Error(ResponsesError),
    Extension {
        kind: String,
        fields: ProviderFields,
    },
}
