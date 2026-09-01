use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OcrPage {
    pub index: u32,
    pub markdown: String,
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrUsage {
    pub pages_processed: Option<u32>,
    pub credits: Option<f64>,
    pub document_size_bytes: Option<u64>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedOcr {
    pub pages: Vec<OcrPage>,
    pub model: String,
    pub document_annotation: Option<Value>,
    pub content: Option<String>,
    pub tables: Option<Vec<Value>>,
    pub key_value_pairs: Option<Vec<Value>>,
    pub usage: OcrUsage,
}

#[derive(Clone, PartialEq)]
pub struct NativeOcrPayload(Value);

impl NativeOcrPayload {
    pub fn new(value: Value) -> Self {
        Self(value)
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }
}

/// ```compile_fail
/// fn assert_serialize<T: serde::Serialize>() {}
/// assert_serialize::<litellm_core::ocr::response::NativeOcrPayload>();
/// assert_serialize::<litellm_core::ocr::response::OcrOutcome>();
/// ```
#[derive(Clone, PartialEq)]
pub struct OcrOutcome {
    pub normalized: NormalizedOcr,
    pub native: NativeOcrPayload,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalized_serialization_cannot_include_native_payload() {
        let outcome = OcrOutcome {
            normalized: NormalizedOcr {
                pages: vec![OcrPage {
                    index: 0,
                    markdown: "portable".to_string(),
                }],
                model: "ocr-model".to_string(),
                document_annotation: None,
                content: None,
                tables: None,
                key_value_pairs: None,
                usage: OcrUsage::default(),
            },
            native: NativeOcrPayload::new(json!({"provider_secret_field": "native"})),
        };

        let public =
            serde_json::to_value(&outcome.normalized).expect("normalized output serializes");

        assert_eq!(public["pages"][0]["markdown"], "portable");
        assert!(public.get("provider_secret_field").is_none());
        assert_eq!(outcome.native.as_value()["provider_secret_field"], "native");
    }
}
