use crate::ocr::types::{OcrDocument, OcrPageDimensions, OcrPageImage, OcrUsageInfo};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeepSeekOcrParams {
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
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekOcrRequest {
    pub model: String,
    pub messages: Vec<DeepSeekOcrMessage>,
    #[serde(flatten)]
    pub params: DeepSeekOcrParams,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekOcrMessage {
    pub role: UserRole,
    pub content: Vec<OcrDocument>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    User,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DeepSeekOcrResponse {
    #[serde(default)]
    pub choices: Vec<DeepSeekChoice>,
    pub usage: Option<OcrUsageInfo>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct DeepSeekChoice {
    pub message: DeepSeekResponseMessage,
}
#[derive(Clone, Debug, Deserialize)]
pub struct DeepSeekResponseMessage {
    pub content: Option<DeepSeekContent>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum DeepSeekContent {
    Text(String),
    Object(DeepSeekOcrResult),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeepSeekOcrResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<Vec<DeepSeekPage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_info: Option<OcrUsageInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_annotation: Option<Value>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekPage {
    #[serde(default, deserialize_with = "crate::ocr::wire::required_i64")]
    pub index: i64,
    #[serde(default)]
    pub markdown: String,
    pub images: Option<Vec<OcrPageImage>>,
    pub dimensions: Option<OcrPageDimensions>,
}
