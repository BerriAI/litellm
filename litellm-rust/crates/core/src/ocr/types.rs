use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleRequest};
use crate::callbacks::custom_guardrail::CustomGuardrail;
use crate::callbacks::custom_logger::CustomLogger;
use crate::callbacks::types::RequestMetadata;
use crate::ocr::transformation::OcrProviderConfig;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OcrDocument {
    #[serde(rename = "document_url")]
    DocumentUrl {
        document_url: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: String,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

impl OcrDocument {
    pub fn url_field(&self) -> (&'static str, &str) {
        match self {
            Self::DocumentUrl { document_url, .. } => ("document_url", document_url),
            Self::ImageUrl { image_url, .. } => ("image_url", image_url),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::OcrDocument;
    use serde_json::json;

    #[test]
    fn deserializes_document_url_variant() {
        let document: OcrDocument = serde_json::from_value(json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        }))
        .expect("document URL should deserialize");

        assert_eq!(document.url_field().1, "https://example.com/doc.pdf");
    }

    #[test]
    fn preserves_document_extension_fields() {
        let document: OcrDocument = serde_json::from_value(json!({
            "type": "image_url",
            "image_url": "data:image/png;base64,abcd",
            "guarded": true
        }))
        .expect("image URL should deserialize");

        let serialized = serde_json::to_value(document).expect("document should serialize");
        assert_eq!(serialized["guarded"], true);
    }

    #[test]
    fn rejects_unknown_document_type() {
        let result = serde_json::from_value::<OcrDocument>(json!({
            "type": "file",
            "file": "document.pdf"
        }));

        assert!(result.is_err());
    }
}

pub struct OcrRequest<'a> {
    pub model: &'a str,
    pub document: OcrDocument,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub callbacks: Vec<Arc<dyn CustomLogger>>,
    pub guardrails: Vec<Arc<dyn CustomGuardrail>>,
    pub request_metadata: RequestMetadata,
    pub litellm_call_id: Option<&'a str>,
}

pub(crate) struct PreparedOcrRequest {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
    pub(crate) litellm_call_id: String,
    pub(crate) document: OcrDocument,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) extra_headers: Option<Map<String, Value>>,
    pub(crate) optional_params: Map<String, Value>,
    pub(crate) timeout: Option<Duration>,
}

impl CallLifecycleRequest for PreparedOcrRequest {
    fn lifecycle_context(&self) -> CallLifecycleContext {
        CallLifecycleContext::new(
            "ocr",
            self.model.clone(),
            self.custom_llm_provider.clone(),
            self.litellm_call_id.clone(),
        )
    }
}

pub(crate) struct ProviderOcrRequest {
    pub(crate) model: String,
    pub(crate) config: &'static dyn OcrProviderConfig,
    pub(crate) url: String,
    pub(crate) body: Value,
    pub(crate) upstream_headers: Vec<(String, String)>,
    pub(crate) timeout: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrRequestData {
    pub data: Value,
    pub files: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrResponseData {
    pub pages: Vec<Value>,
    pub model: String,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<Value>,
    pub object: String,
}

impl OcrResponseData {
    pub fn into_json(self) -> Value {
        serde_json::json!({
            "pages": self.pages,
            "model": self.model,
            "document_annotation": self.document_annotation,
            "usage_info": self.usage_info,
            "object": self.object,
        })
    }
}
