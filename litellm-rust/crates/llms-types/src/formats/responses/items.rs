use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::recognized::Recognized;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesContent {
    Text(String),
    Parts(Vec<Recognized<ResponsesContentPart>>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    InputText {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    OutputText {
        text: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    InputImage {
        image_url: String,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesItem {
    Message {
        role: String,
        content: ResponsesContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    FunctionCallOutput {
        call_id: String,
        output: ResponsesContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    Reasoning {
        summary: Vec<Recognized<ResponsesContentPart>>,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}
