use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::hooks::{NoopOcrHooks, OcrHooks};
use super::registry::{OcrAdapterKind, resolve_wire_adapter};
use crate::Error;
use crate::auth::InputSource;
use crate::constants::OCR_HTTP_TIMEOUT_SECS;

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

impl OcrDocument {
    pub(crate) fn source(&self) -> &str {
        match self {
            Self::DocumentUrl { document_url, .. } => document_url,
            Self::ImageUrl { image_url, .. } => image_url,
        }
    }

    pub(crate) fn with_source(self, source: String) -> Self {
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

#[derive(Clone)]
pub struct OcrConnection {
    pub api_key: Option<String>,
    pub api_key_source: InputSource,
    pub api_base: Option<String>,
    pub api_base_source: InputSource,
    pub extra_headers: Vec<(String, String)>,
    pub extra_headers_source: InputSource,
    pub timeout: Duration,
    pub max_download_bytes: u64,
    pub poll_timeout: Duration,
}

impl Default for OcrConnection {
    fn default() -> Self {
        Self {
            api_key: None,
            api_key_source: InputSource::Deployment,
            api_base: None,
            api_base_source: InputSource::Deployment,
            extra_headers: Vec::new(),
            extra_headers_source: InputSource::Deployment,
            timeout: Duration::from_secs(OCR_HTTP_TIMEOUT_SECS),
            max_download_bytes: crate::constants::OCR_DOWNLOAD_MAX_BYTES,
            poll_timeout: Duration::from_secs(crate::constants::OCR_POLL_TIMEOUT_SECS),
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
    pub input_sources: BTreeMap<String, InputSource>,
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
            input_sources: BTreeMap::new(),
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
