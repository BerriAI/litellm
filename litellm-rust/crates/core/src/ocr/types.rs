use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::hooks::{NoopOcrHooks, OcrHooks};
use super::registry::{OcrAdapterRequest, decode_adapter_request, resolve_wire_adapter};
use crate::Error;
use crate::constants::OCR_HTTP_TIMEOUT_SECS;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrRequestData {
    pub data: Value,
    pub files: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OcrDocument {
    #[serde(rename = "document_url")]
    DocumentUrl {
        document_url: String,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: String,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrRequestFormat {
    #[default]
    Litellm,
    Native,
}

#[derive(Clone)]
pub struct OcrConnection {
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Vec<(String, String)>,
    pub timeout: Duration,
}

impl Default for OcrConnection {
    fn default() -> Self {
        Self {
            api_key: None,
            api_base: None,
            extra_headers: Vec::new(),
            timeout: Duration::from_secs(OCR_HTTP_TIMEOUT_SECS),
        }
    }
}

pub struct OcrRequest {
    pub model: String,
    pub document: OcrDocument,
    pub connection: OcrConnection,
    pub hooks: Arc<dyn OcrHooks>,
    pub litellm_call_id: Option<String>,
    pub(crate) adapter: OcrAdapterRequest,
}

impl OcrRequest {
    pub fn new(
        model: String,
        document: OcrDocument,
        custom_llm_provider: Option<&str>,
        optional_params: Map<String, Value>,
    ) -> Result<Self, Error> {
        let (model, adapter_kind) = resolve_wire_adapter(&model, custom_llm_provider)?;
        let adapter = decode_adapter_request(adapter_kind, optional_params)?;
        Ok(Self {
            model,
            document,
            connection: OcrConnection::default(),
            hooks: Arc::new(NoopOcrHooks),
            litellm_call_id: None,
            adapter,
        })
    }

    pub fn with_host_hooks(
        self,
        hooks: Arc<dyn OcrHooks>,
        litellm_call_id: Option<String>,
    ) -> Self {
        Self {
            hooks,
            litellm_call_id,
            ..self
        }
    }
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
