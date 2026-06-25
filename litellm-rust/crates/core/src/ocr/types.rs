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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrDimensions {
    pub dpi: Option<u32>,
    pub height: Option<u32>,
    pub width: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrImage {
    pub id: Option<String>,
    pub top_left_x: Option<f64>,
    pub top_left_y: Option<f64>,
    pub bottom_right_x: Option<f64>,
    pub bottom_right_y: Option<f64>,
    pub image_base64: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPage {
    pub index: Option<u32>,
    pub markdown: Option<String>,
    #[serde(default)]
    pub images: Vec<OcrImage>,
    pub dimensions: Option<OcrDimensions>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrDocumentAnnotation {
    pub markdown: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrUsageInfo {
    pub pages_processed: Option<u32>,
    pub doc_size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MistralOcrResponseData {
    #[serde(default)]
    pub pages: Vec<OcrPage>,
    pub model: Option<String>,
    pub document_annotation: Option<OcrDocumentAnnotation>,
    pub usage_info: Option<OcrUsageInfo>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrResponseData {
    pub pages: Vec<OcrPage>,
    pub model: String,
    pub document_annotation: Option<OcrDocumentAnnotation>,
    pub usage_info: Option<OcrUsageInfo>,
    pub object: String,
}
