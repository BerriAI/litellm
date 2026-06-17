//! Mistral OCR transformation. Mirrors `mistral_ocr_transformation.py`. Pure: no I/O.

use serde_json::{Map, Value};

use crate::llms::base_llm::ocr::transformation::{
    BaseOcrConfig, DocumentType, Headers, OcrResponse,
};

/// Default Mistral API base.
pub const MISTRAL_DEFAULT_API_BASE: &str = "https://api.mistral.ai/v1";

/// Env var holding the Mistral API key.
pub const MISTRAL_API_KEY_ENV: &str = "MISTRAL_API_KEY";

/// Error message raised when no Mistral API key is available.
pub const MISSING_KEY_MESSAGE: &str = "Missing Mistral API Key - A call is being made to Mistral but no key is set either in the environment variables or via params";

/// Mistral OCR transformation configuration.
///
/// Reference: <https://docs.mistral.ai/api/#tag/ocr>
#[derive(Debug, Default, Clone)]
pub struct MistralOcrConfig;

impl MistralOcrConfig {
    pub fn new() -> Self {
        Self
    }
}

impl BaseOcrConfig for MistralOcrConfig {
    fn get_supported_ocr_params(&self, _model: &str) -> Vec<String> {
        [
            "pages",
            "include_image_base64",
            "image_limit",
            "image_min_size",
            "bbox_annotation_format",
            "document_annotation_format",
            "document_annotation_prompt",
            "extract_header",
            "extract_footer",
            "table_format",
            "confidence_scores_granularity",
            "id",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn validate_environment(
        &self,
        _model: &str,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Headers, String> {
        // Get API key from environment if not provided.
        let resolved = match api_key {
            Some(k) => Some(k.to_string()),
            None => env_lookup(MISTRAL_API_KEY_ENV),
        };

        let resolved = resolved.ok_or_else(|| MISSING_KEY_MESSAGE.to_string())?;

        Ok(vec![(
            "Authorization".to_string(),
            format!("Bearer {resolved}"),
        )])
    }

    fn get_complete_url(&self, api_base: Option<&str>, _model: &str) -> String {
        let base = api_base.unwrap_or(MISTRAL_DEFAULT_API_BASE);

        // Ensure no trailing slash.
        let base = base.trim_end_matches('/');

        // Remove /v1 if it's already in the base to avoid duplication.
        if base.ends_with("/v1") {
            format!("{base}/ocr")
        } else {
            format!("{base}/v1/ocr")
        }
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: &DocumentType,
        optional_params: &Map<String, Value>,
    ) -> Result<Value, String> {
        // Document must be a dict (Mistral-format), passed through as-is.
        if !document.is_object() {
            return Err(format!(
                "Expected document dict, got {}",
                json_type_name(document)
            ));
        }

        let mut data = Map::new();
        data.insert("model".to_string(), Value::String(model.to_string()));
        data.insert("document".to_string(), document.clone());

        // Add all (already-filtered) optional parameters.
        for (k, v) in optional_params {
            data.insert(k.clone(), v.clone());
        }

        Ok(Value::Object(data))
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: &Value,
    ) -> Result<OcrResponse, String> {
        let pages_value = response_json
            .get("pages")
            .cloned()
            .unwrap_or_else(|| Value::Array(vec![]));
        let pages = serde_json::from_value(pages_value)
            .map_err(|e| format!("Error parsing Mistral OCR pages: {e}"))?;

        let resolved_model = response_json
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or(model)
            .to_string();

        let document_annotation = response_json.get("document_annotation").cloned();

        let usage_info = match response_json.get("usage_info") {
            Some(Value::Null) | None => None,
            Some(v) => Some(
                serde_json::from_value(v.clone())
                    .map_err(|e| format!("Error parsing Mistral OCR usage_info: {e}"))?,
            ),
        };

        Ok(OcrResponse {
            pages,
            model: resolved_model,
            document_annotation,
            usage_info,
            object: "ocr".to_string(),
        })
    }
}

/// Human-readable JSON type name, matching Python's `type(...)`-style error text.
fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}
