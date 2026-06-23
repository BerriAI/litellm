//! Base OCR transformation: pure serde types and the `BaseOcrConfig` trait.
//!
//! Mirrors `litellm/llms/base_llm/ocr/transformation.py`. No PyO3, no I/O.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Document passed to a provider. Always a dict with `type="document_url"`
/// or `type="image_url"` by the time it reaches a provider transform.
///
/// We model it as a free-form JSON object so extra keys are preserved and
/// the document is passed through to the upstream body verbatim.
pub type DocumentType = Value;

/// HTTP header pairs (name, value).
pub type Headers = Vec<(String, String)>;

/// Page dimensions from an OCR response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OcrPageDimensions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dpi: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<i64>,
}

/// Image extracted from an OCR page. Extra keys are preserved (`extra = "allow"`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OcrPageImage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_base64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Value>,

    /// Preserve any additional fields returned by the provider.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A single page from an OCR response. Extra keys are preserved.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OcrPage {
    pub index: i64,
    pub markdown: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<OcrPageImage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<OcrPageDimensions>,

    /// Preserve any additional fields returned by the provider.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Usage information from an OCR response. Extra keys are preserved.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OcrUsageInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pages_processed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_size_bytes: Option<i64>,

    /// Preserve any additional fields returned by the provider.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Standard OCR response format (standardized to the Mistral OCR format).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OcrResponse {
    pub pages: Vec<OcrPage>,
    pub model: String,
    #[serde(default)]
    pub document_annotation: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_info: Option<OcrUsageInfo>,
    #[serde(default = "default_object")]
    pub object: String,
}

fn default_object() -> String {
    "ocr".to_string()
}

/// Inputs for a single OCR call, assembled by the caller (Python or HTTP server).
#[derive(Debug, Clone)]
pub struct OcrRequest {
    /// Provider-stripped model id, used directly as body `"model"`.
    pub model: String,
    /// Pre-normalized document dict, passed through verbatim into the body.
    pub document: DocumentType,
    /// Optional explicit API key. If `None`, providers fall back to env.
    pub api_key: Option<String>,
    /// Optional API base override.
    pub api_base: Option<String>,
    /// Raw OCR params; providers filter to their supported list.
    pub optional_params: Map<String, Value>,
}

/// Provider-agnostic OCR transform contract. Pure: no I/O.
pub trait BaseOcrConfig {
    /// Supported OCR parameter keys for this provider/model.
    fn get_supported_ocr_params(&self, model: &str) -> Vec<String>;

    /// Filter `optional_params` down to the supported list, dropping the rest.
    fn map_ocr_params(
        &self,
        optional_params: &Map<String, Value>,
        model: &str,
    ) -> Map<String, Value> {
        let supported = self.get_supported_ocr_params(model);
        optional_params
            .iter()
            .filter(|(k, _)| supported.iter().any(|s| s == *k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Resolve the API key (param or env) and build request headers.
    fn validate_environment(
        &self,
        model: &str,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Headers, String>;

    /// Build the complete OCR endpoint URL.
    fn get_complete_url(&self, api_base: Option<&str>, model: &str) -> String;

    /// Build the request body for the upstream OCR call.
    fn transform_ocr_request(
        &self,
        model: &str,
        document: &DocumentType,
        optional_params: &Map<String, Value>,
    ) -> Result<Value, String>;

    /// Parse the upstream JSON body into the standard `OcrResponse`.
    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: &Value,
    ) -> Result<OcrResponse, String>;
}
