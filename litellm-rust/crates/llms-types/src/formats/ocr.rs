use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_with::serde_as;

use litellm_python_compat::serde_compat::{FiniteF64, LaxI64};

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

#[cfg(test)]
mod tests {
    use super::{LiteLLMOcrResponse, OcrDocument, OcrPage};
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    #[rstest]
    #[case::missing_page_fields(json!({"pages": [{}]}))]
    #[case::invalid_markdown(json!({"pages": [{"index": 0, "markdown": false}]}))]
    #[case::invalid_image_bounds(json!({"pages": [{"index": 0, "markdown": "", "images": [{"bbox": []}]}]}))]
    #[case::fractional_page_count(json!({"usage_info": {"pages_processed": 1.5}}))]
    #[case::invalid_table(json!({"tables": [false]}))]
    #[case::invalid_key_value_pair(json!({"keyValuePairs": [[]]}))]
    #[case::invalid_native_response(json!({"provider_native_response": []}))]
    fn normalized_response_rejects_invalid_shared_fields(#[case] fields: Value) {
        let payload: Map<String, Value> = json!({"model": "model", "pages": []})
            .as_object()
            .unwrap()
            .iter()
            .chain(fields.as_object().unwrap())
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        assert!(serde_json::from_value::<LiteLLMOcrResponse>(Value::Object(payload)).is_err());
    }

    #[rstest]
    fn document_rejects_non_string_provider_fields() {
        assert!(
            serde_json::from_value::<OcrDocument>(json!({
                "type": "image_url", "image_url": "https://example.com/image", "detail": 42
            }))
            .is_err()
        );
    }

    #[rstest]
    #[case::large_integer(json!("9007199254740993.0"), 9_007_199_254_740_993)]
    #[case::signed_decimal(json!("+2.000"), 2)]
    #[case::separator(json!("1_000"), 1000)]
    #[case::boolean(json!(true), 1)]
    #[case::integral_float(json!(2.0), 2)]
    fn numeric_coercion_preserves_integer_precision(#[case] value: Value, #[case] expected: i64) {
        let page: OcrPage =
            serde_json::from_value(json!({"index": value, "markdown": ""})).unwrap();
        assert_eq!(page.index, expected);
        assert_eq!(
            serde_json::to_value(page).unwrap()["index"],
            json!(expected)
        );
    }

    #[rstest]
    #[case::exponent(json!("1e2"))]
    #[case::missing_integer(json!(".0"))]
    #[case::missing_fraction(json!("2."))]
    #[case::leading_separator(json!("_2"))]
    #[case::repeated_separator(json!("2__0"))]
    #[case::fractional_float(json!(2.5))]
    #[case::null(json!(null))]
    fn page_index_rejects_invalid_integers(#[case] value: Value) {
        assert!(
            serde_json::from_value::<OcrPage>(json!({"index": value, "markdown": ""})).is_err()
        );
    }

    #[rstest]
    #[case::document_url("document_url", "document_name", "application/pdf")]
    #[case::image_url("image_url", "detail", "image/png")]
    fn document_variants_preserve_provider_fields_when_rewriting_sources(
        #[case] kind: &str,
        #[case] field: &str,
        #[case] mime_type: &str,
        #[values(json!("kept"), Value::Null)] extra: Value,
    ) {
        let original = "https://example.com/input";
        let replacement = format!("data:{mime_type};base64,AA==");
        let document: OcrDocument =
            serde_json::from_value(json!({"type": kind, kind: original, field: extra})).unwrap();
        assert_eq!(document.source(), original);
        assert!(document.is_remote());
        let rewritten = document.with_source(replacement.clone());
        assert!(!rewritten.is_remote());
        assert_eq!(
            serde_json::to_value(rewritten).unwrap(),
            json!({"type": kind, kind: replacement, field: extra})
        );
    }

    #[rstest]
    #[case::absent_native(None)]
    #[case::present_native(Some(Map::from_iter([("native".into(), json!({"nested": [null, 1]}))])))]
    fn response_serialization_preserves_extensions_and_native_presence(
        #[case] native: Option<Map<String, Value>>,
    ) {
        let response = LiteLLMOcrResponse {
            extra_fields: Map::from_iter([("provider_field".into(), json!("kept"))]),
            provider_native_response: native.clone(),
            ..LiteLLMOcrResponse::new("model", vec![])
        };
        let serialized = response.into_json();
        assert_eq!(serialized["provider_field"], "kept");
        assert_eq!(
            serialized.get("provider_native_response").cloned(),
            native.clone().map(Value::Object)
        );
        let decoded: LiteLLMOcrResponse = serde_json::from_value(serialized.clone()).unwrap();
        assert_eq!(decoded.provider_native_response, native);
        assert_eq!(decoded.into_json(), serialized);
    }
}
