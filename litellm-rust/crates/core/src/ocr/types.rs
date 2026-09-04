use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::transformation::OcrProviderConfig;

pub struct OcrRequest<'a> {
    pub model: &'a str,
    pub document: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub max_document_download_bytes: u64,
}

pub(super) struct PreparedOcrRequest {
    pub(super) model: String,
    pub(super) config: &'static dyn OcrProviderConfig,
    pub(super) document: Value,
    pub(super) api_key: Option<String>,
    pub(super) api_base: Option<String>,
    pub(super) extra_headers: Option<Map<String, Value>>,
    pub(super) url_params: Map<String, Value>,
    pub(super) optional_params: Map<String, Value>,
    pub(super) requires_reducto_upload: bool,
    pub(super) timeout: Option<Duration>,
    pub(super) max_document_download_bytes: u64,
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
    pub extra_fields: Map<String, Value>,
    pub provider_native_response: Option<Value>,
}

impl OcrResponseData {
    pub fn into_json(self) -> Value {
        let mut response = serde_json::json!({
            "pages": self.pages,
            "model": self.model,
            "document_annotation": self.document_annotation,
            "usage_info": self.usage_info,
            "object": self.object,
        });
        if let Value::Object(object) = &mut response {
            object.extend(self.extra_fields);
            if let Some(native_response) = self.provider_native_response {
                object.insert("provider_native_response".to_string(), native_response);
            }
        }
        response
    }
}
