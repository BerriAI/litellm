use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::ocr::types::OcrDocument;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct MistralOcrParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_image_base64: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_limit: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_min_size: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox_annotation_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_annotation_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_annotation_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extract_header: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extract_footer: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_scores_granularity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_blocks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MistralOcrRequest {
    pub model: String,
    pub document: OcrDocument,
    #[serde(flatten)]
    pub params: MistralOcrParams,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct MistralOcrResponse {
    #[serde(default)]
    pub pages: Vec<Value>,
    pub model: Option<String>,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<Value>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}
