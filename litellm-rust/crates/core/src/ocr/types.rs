use crate::http_utils::body::JsonPayload;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrRequestData {
    pub data: JsonPayload,
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

pub struct ProviderOcrRequest {
    pub model: String,
    pub config: &'static dyn super::transformation::OcrProviderConfig,
    pub url: String,
    pub body: JsonPayload,
    pub upstream_headers: Vec<(String, String)>,
    pub timeout: Option<std::time::Duration>,
}
