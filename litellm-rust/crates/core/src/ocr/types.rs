use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OcrDocument {
    DocumentUrl { document_url: String },
    ImageUrl { image_url: String },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MistralOcrOptionalParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_image_base64: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_min_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox_annotation_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_annotation_format: Option<String>,
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
    pub id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrRequestData {
    pub model: String,
    pub document: OcrDocument,
    #[serde(flatten)]
    pub optional_params: MistralOcrOptionalParams,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OcrFieldValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OcrFieldValue>),
    Object(BTreeMap<String, OcrFieldValue>),
}

pub type OcrObject = BTreeMap<String, OcrFieldValue>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MistralOcrResponseData {
    #[serde(default)]
    pub pages: Vec<OcrObject>,
    pub model: Option<String>,
    pub document_annotation: Option<OcrFieldValue>,
    pub usage_info: Option<OcrObject>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrResponseData {
    pub pages: Vec<OcrObject>,
    pub model: String,
    pub document_annotation: Option<OcrFieldValue>,
    pub usage_info: Option<OcrObject>,
    pub object: String,
}
