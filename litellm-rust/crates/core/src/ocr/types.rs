use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Error;
use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleRequest};
use crate::ocr::transformation::OcrProviderConfig;

pub struct OcrRequest<'a> {
    pub model: &'a str,
    pub document: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub litellm_call_id: Option<&'a str>,
}

pub struct PreparedOcrRequest {
    pub config: Result<&'static dyn OcrProviderConfig, Error>,
    pub model: String,
    pub custom_llm_provider: String,
    pub litellm_call_id: String,
    pub document: Value,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
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

pub struct ProviderOcrRequest {
    pub(super) model: String,
    pub(super) custom_llm_provider: String,
    pub(super) config: &'static dyn OcrProviderConfig,
    pub(super) url: String,
    pub(super) body: Value,
    pub(super) upstream_headers: Vec<(String, String)>,
    pub(super) timeout: Option<Duration>,
}

impl ProviderOcrRequest {
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn custom_llm_provider(&self) -> &str {
        &self.custom_llm_provider
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn body(&self) -> &Value {
        &self.body
    }

    pub fn with_body(self, body: Value) -> Self {
        Self { body, ..self }
    }
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
