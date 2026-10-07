use serde_json::{Map, Value};

use crate::recognized::Recognized;
use crate::serde_compat::deserialize_present;

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
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    OutputText {
        text: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        annotations: Option<Recognized<Vec<Recognized<ResponsesAnnotation>>>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        logprobs:
            Option<Recognized<Vec<Recognized<crate::formats::chat_completions::ChatTokenLogprob>>>>,
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
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub status: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub role: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub phase: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub content: Option<Recognized<Vec<Recognized<ResponsesContentPart>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesFunctionCall {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub status: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub call_id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub name: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub arguments: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub phase: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesCustomToolCall {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub status: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub call_id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub name: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub input: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesReasoning {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub status: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub summary: Option<Recognized<Vec<Recognized<ResponsesContentPart>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub content: Option<Recognized<Vec<Recognized<ResponsesContentPart>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub encrypted_content: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesWebSearchCall {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub status: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub action: Option<Recognized<ResponsesWebSearchAction>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesFileSearchCall {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub status: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub queries: Option<Recognized<Vec<String>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub results: Option<Recognized<Vec<Recognized<ResponsesFileSearchResult>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesFileSearchResult {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub file_id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub filename: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub score: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub text: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub attributes: Option<Recognized<Map<String, Value>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesImageGenerationCall {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub status: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub result: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesCodeInterpreterCall {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub status: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub code: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub container_id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub outputs: Option<Recognized<Vec<Recognized<ResponsesCodeOutput>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesMcpCall {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub status: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub name: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub server_label: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub arguments: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub output: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub error: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub approval_request_id: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesUrlCitation {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub url: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub title: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub start_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub end_index: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ResponsesFileCitation {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub file_id: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub filename: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub start_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub end_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub container_id: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesWebSearchAction {
    Search {
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        query: Option<Recognized<String>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        queries: Option<Recognized<Vec<String>>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "deserialize_present"
        )]
        sources: Option<Recognized<Vec<Recognized<ResponsesWebSearchSource>>>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    OpenPage {
        url: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Find {
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
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "type")]
    pub source_type: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub url: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
