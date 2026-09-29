use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_with::serde_as;

use crate::serde_compat::{FiniteF64, LaxI64};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OcrDocument {
    #[serde(rename = "document_url")]
    DocumentUrl {
        document_url: String,
        #[serde(flatten)]
        extra_fields: BTreeMap<String, Option<String>>,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: String,
        #[serde(flatten)]
        extra_fields: BTreeMap<String, Option<String>>,
    },
}

impl OcrDocument {
    pub fn source(&self) -> &str {
        match self {
            Self::DocumentUrl { document_url, .. } => document_url,
            Self::ImageUrl { image_url, .. } => image_url,
        }
    }

    pub fn is_remote(&self) -> bool {
        let source = self.source();
        source.starts_with("http://") || source.starts_with("https://")
    }

    pub fn with_source(self, source: String) -> Self {
        match self {
            Self::DocumentUrl { extra_fields, .. } => Self::DocumentUrl {
                document_url: source,
                extra_fields,
            },
            Self::ImageUrl { extra_fields, .. } => Self::ImageUrl {
                image_url: source,
                extra_fields,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrResponseFormat {
    #[default]
    Litellm,
    Native,
}

#[serde_as]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPageDimensions {
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub dpi: Option<i64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub height: Option<i64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub width: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPageImage {
    pub image_base64: Option<String>,
    pub bbox: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[serde_as]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPage {
    #[serde_as(deserialize_as = "LaxI64")]
    pub index: i64,
    pub markdown: String,
    pub images: Option<Vec<OcrPageImage>>,
    pub dimensions: Option<OcrPageDimensions>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[serde_as]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrUsageInfo {
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub pages_processed: Option<i64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub pages_processed_annotation: Option<i64>,
    #[serde_as(deserialize_as = "Option<FiniteF64>")]
    pub credits: Option<f64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub doc_size_bytes: Option<i64>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LiteLLMOcrResponse {
    pub pages: Vec<OcrPage>,
    pub model: String,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<OcrUsageInfo>,
    pub content: Option<String>,
    pub tables: Option<Vec<Map<String, Value>>>,
    #[serde(rename = "keyValuePairs")]
    pub key_value_pairs: Option<Vec<Map<String, Value>>>,
    #[serde(default = "ocr_object")]
    pub object: String,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_native_response: Option<Map<String, Value>>,
}

impl LiteLLMOcrResponse {
    pub fn new(model: impl Into<String>, pages: Vec<OcrPage>) -> Self {
        Self {
            pages,
            model: model.into(),
            document_annotation: None,
            usage_info: None,
            content: None,
            tables: None,
            key_value_pairs: None,
            object: ocr_object(),
            extra_fields: Map::new(),
            provider_native_response: None,
        }
    }

    pub fn into_json(self) -> Value {
        serde_json::to_value(self).expect("OCR response fields are JSON-compatible")
    }
}

fn ocr_object() -> String {
    "ocr".into()
}
