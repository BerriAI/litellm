use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::hooks::{NoopOcrHooks, OcrHooks};
use super::prepare::OcrProviderRequest;
use crate::auth::azure::AzureAuthInputs;
use crate::constants::{OCR_DOWNLOAD_MAX_BYTES, OCR_HTTP_TIMEOUT_SECS};
use crate::providers::vertex_ai::auth::VertexAuthInputs;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OcrDocument {
    #[serde(rename = "document_url")]
    DocumentUrl { document_url: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: String },
}

impl OcrDocument {
    pub fn source(&self) -> &str {
        match self {
            Self::DocumentUrl { document_url } => document_url,
            Self::ImageUrl { image_url } => image_url,
        }
    }

    pub fn with_source(self, source: String) -> Self {
        match self {
            Self::DocumentUrl { .. } => Self::DocumentUrl {
                document_url: source,
            },
            Self::ImageUrl { .. } => Self::ImageUrl { image_url: source },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrRequestFormat {
    #[default]
    Litellm,
    Native,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPageDimensions {
    #[serde(default, deserialize_with = "super::wire::optional_i64")]
    pub dpi: Option<i64>,
    #[serde(default, deserialize_with = "super::wire::optional_i64")]
    pub height: Option<i64>,
    #[serde(default, deserialize_with = "super::wire::optional_i64")]
    pub width: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrPageImage {
    pub image_base64: Option<String>,
    pub bbox: Option<Map<String, Value>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrPage {
    #[serde(deserialize_with = "super::wire::required_i64")]
    pub index: i64,
    pub markdown: String,
    pub images: Option<Vec<OcrPageImage>>,
    pub dimensions: Option<OcrPageDimensions>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

impl OcrPage {
    pub fn text(index: i64, markdown: String) -> Self {
        Self {
            index,
            markdown,
            images: None,
            dimensions: None,
            extra_fields: Map::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrUsageInfo {
    #[serde(default, deserialize_with = "super::wire::optional_i64")]
    pub pages_processed: Option<i64>,
    #[serde(default, deserialize_with = "super::wire::optional_i64")]
    pub pages_processed_annotation: Option<i64>,
    #[serde(default, deserialize_with = "super::wire::optional_f64")]
    pub credits: Option<f64>,
    #[serde(default, deserialize_with = "super::wire::optional_i64")]
    pub doc_size_bytes: Option<i64>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrResponseData {
    pub pages: Vec<OcrPage>,
    pub model: String,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<OcrUsageInfo>,
    pub content: Option<String>,
    pub tables: Option<Vec<Map<String, Value>>>,
    #[serde(rename = "keyValuePairs")]
    pub key_value_pairs: Option<Vec<Map<String, Value>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_native_response: Option<Map<String, Value>>,
    pub object: OcrObject,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OcrObject {
    #[serde(rename = "ocr")]
    Ocr,
}

impl OcrResponseData {
    pub fn new(model: String, pages: Vec<OcrPage>) -> Self {
        Self {
            model,
            pages,
            document_annotation: None,
            usage_info: None,
            content: None,
            tables: None,
            key_value_pairs: None,
            extra_fields: Map::new(),
            provider_native_response: None,
            object: OcrObject::Ocr,
        }
    }

    pub fn into_json(self) -> Value {
        serde_json::json!(self)
    }
}

#[derive(Clone, Default)]
pub struct VertexOcrSettings {
    pub project: Option<String>,
    pub location: Option<String>,
}

#[derive(Clone)]
pub struct OcrConnection {
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Vec<(String, String)>,
    pub azure_auth: Option<AzureAuthInputs>,
    pub vertex_auth: VertexAuthInputs,
    pub vertex: VertexOcrSettings,
    pub timeout: Duration,
    pub poll_timeout: Duration,
    pub max_download_bytes: u64,
}

impl Default for OcrConnection {
    fn default() -> Self {
        Self {
            api_key: None,
            api_base: None,
            extra_headers: Vec::new(),
            azure_auth: None,
            vertex_auth: VertexAuthInputs::default(),
            vertex: VertexOcrSettings::default(),
            timeout: Duration::from_secs(OCR_HTTP_TIMEOUT_SECS),
            poll_timeout: Duration::from_secs(crate::constants::OCR_POLL_TIMEOUT_SECS),
            max_download_bytes: OCR_DOWNLOAD_MAX_BYTES,
        }
    }
}

pub struct OcrRequest {
    pub model: String,
    pub document: OcrDocument,
    pub provider: OcrProviderRequest,
    pub connection: OcrConnection,
    pub hooks: Arc<dyn OcrHooks>,
    pub litellm_call_id: Option<String>,
}

impl OcrRequest {
    pub fn new(model: String, document: OcrDocument, provider: OcrProviderRequest) -> Self {
        Self {
            model,
            document,
            provider,
            connection: OcrConnection::default(),
            hooks: Arc::new(NoopOcrHooks),
            litellm_call_id: None,
        }
    }
}
