use std::collections::BTreeMap;

use serde_json::{Map, Value};
use serde_with::serde_as;

use crate::recognized::Recognized;
use crate::serde_compat::deserialize_present;
use crate::serde_compat::{FiniteF64, LaxI64};

#[macro_rules_attribute::apply(wire_type)]
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

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Default, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OcrResponseFormat {
    #[default]
    Litellm,
    Native,
}

#[serde_as]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OcrPageDimensions {
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub dpi: Option<i64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub height: Option<i64>,
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub width: Option<i64>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OcrPageImage {
    pub image_base64: Option<String>,
    pub bbox: Option<OcrBoundingBox>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[serde_as]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
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
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
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

#[macro_rules_attribute::apply(wire_type)]
pub struct LiteLLMOcrResponse {
    pub pages: Vec<OcrPage>,
    pub model: String,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<OcrUsageInfo>,
    pub content: Option<String>,
    pub tables: Option<Vec<OcrTable>>,
    #[serde(rename = "keyValuePairs")]
    pub key_value_pairs: Option<Vec<OcrKeyValuePair>>,
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

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OcrBoundingBox {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub x: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub y: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub width: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub height: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub top_left_x: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub top_left_y: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub bottom_right_x: Option<Recognized<serde_json::Number>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub bottom_right_y: Option<Recognized<serde_json::Number>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OcrTable {
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "rowCount")]
    pub row_count: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "columnCount")]
    pub column_count: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cells: Option<Recognized<Vec<Recognized<OcrTableCell>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "boundingRegions")]
    pub bounding_regions: Option<Recognized<Vec<Recognized<OcrBoundingRegion>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub spans: Option<Recognized<Vec<Recognized<OcrTextSpan>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub content: Option<Recognized<String>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OcrTableCell {
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "rowIndex")]
    pub row_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "columnIndex")]
    pub column_index: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "rowSpan")]
    pub row_span: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "columnSpan")]
    pub column_span: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub kind: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub content: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "boundingRegions")]
    pub bounding_regions: Option<Recognized<Vec<Recognized<OcrBoundingRegion>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub spans: Option<Recognized<Vec<Recognized<OcrTextSpan>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OcrKeyValuePair {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub key: Option<Recognized<OcrKeyValueElement>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub value: Option<Recognized<OcrKeyValueElement>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub confidence: Option<Recognized<serde_json::Number>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OcrKeyValueElement {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub content: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "boundingRegions")]
    pub bounding_regions: Option<Recognized<Vec<Recognized<OcrBoundingRegion>>>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub spans: Option<Recognized<Vec<Recognized<OcrTextSpan>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OcrBoundingRegion {
    #[serde(default, deserialize_with = "deserialize_present")]
    #[serde(rename = "pageNumber")]
    pub page_number: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub polygon: Option<Recognized<Vec<serde_json::Number>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct OcrTextSpan {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub offset: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub length: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
