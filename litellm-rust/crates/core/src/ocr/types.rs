use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::hooks::{NoopOcrHooks, OcrHooks};
use super::registry::{OcrAdapterKind, resolve_wire_adapter};
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
pub enum OcrResponseFormat {
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

pub struct LiteLLMOcrRequest {
    pub model: String,
    pub document: OcrDocument,
    pub connection: OcrConnection,
    pub hooks: Arc<dyn OcrHooks>,
    pub litellm_call_id: Option<String>,
    pub optional_params: Map<String, Value>,
    pub(crate) adapter: OcrAdapterKind,
}

impl LiteLLMOcrRequest {
    pub fn new(
        model: String,
        document: OcrDocument,
        custom_llm_provider: Option<&str>,
        optional_params: Map<String, Value>,
    ) -> Result<Self, Error> {
        let (model, adapter_kind) = resolve_wire_adapter(&model, custom_llm_provider)?;

        Ok(Self {
            model,
            document,
            connection: OcrConnection::default(),
            hooks: Arc::new(NoopOcrHooks),
            litellm_call_id: None,
            optional_params,
            adapter: adapter_kind,
        })
    }

    pub(crate) fn response_format(
        &self,
    ) -> Result<OcrResponseFormat, super::error::OcrRequestError> {
        self.optional_params
            .get("req_format")
            .map(|value| {
                serde_json::from_value(value.clone())
                    .map_err(|_| super::error::OcrRequestError::RequestFormat)
            })
            .transpose()
            .map(|format| format.unwrap_or_default())
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
pub struct LiteLLMOcrResponse {
    pub pages: Vec<Value>,
    pub model: String,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<Value>,
    pub object: String,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_native_response: Option<Value>,
}

impl LiteLLMOcrResponse {
    pub fn into_json(self) -> Value {
        serde_json::to_value(self).expect("OCR response fields are JSON-compatible")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn response_serialization_flattens_extra_fields_and_omits_absent_native_response() {
        let response = LiteLLMOcrResponse {
            pages: vec![],
            model: "model".into(),
            document_annotation: None,
            usage_info: None,
            object: "ocr".into(),
            extra_fields: json!({"provider_field":"kept"})
                .as_object()
                .unwrap()
                .clone(),
            provider_native_response: None,
        };
        let serialized = response.into_json();
        assert_eq!(serialized["provider_field"], "kept");
        assert!(serialized.get("provider_native_response").is_none());
    }
}
