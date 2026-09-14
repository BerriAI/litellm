use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct DeepSeekOcrParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequences>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum StopSequences {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DeepSeekOcrRequest {
    pub model: String,
    pub messages: Vec<DeepSeekOcrMessage>,
    #[serde(flatten)]
    pub params: DeepSeekOcrParams,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DeepSeekOcrMessage {
    pub role: UserRole,
    pub content: Vec<crate::ocr::types::OcrDocument>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum UserRole {
    User,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct DeepSeekOcrResponse {
    #[serde(default)]
    pub choices: Vec<DeepSeekChoice>,
    pub usage: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct DeepSeekChoice {
    pub message: DeepSeekResponseMessage,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct DeepSeekResponseMessage {
    pub content: Option<DeepSeekContent>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum DeepSeekContent {
    Text(String),
    Object(DeepSeekOcrResult),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct DeepSeekOcrResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<Vec<DeepSeekPage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_info: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_annotation: Option<Value>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DeepSeekPage {
    #[serde(default)]
    pub index: i64,
    #[serde(default)]
    pub markdown: String,
    pub images: Option<Value>,
    pub dimensions: Option<Value>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}
