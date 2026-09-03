use crate::error::{Error, json_type_name};
use crate::ocr::transformation::OcrProviderConfig;
use crate::ocr::types::{OcrRequestData, OcrResponseData};
use serde_json::{Map, Value};

const SUPPORTED_OCR_PARAMS: &[&str] = &[
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
    "include_blocks",
    "id",
];

/// Default Mistral API base, used when the caller does not override `api_base`.
pub const MISTRAL_DEFAULT_API_BASE: &str = "https://api.mistral.ai/v1";

/// Environment variable holding the Mistral API key.
pub const MISTRAL_API_KEY_ENV: &str = "MISTRAL_API_KEY";

/// Error message raised when no Mistral API key can be resolved.
pub const MISSING_KEY_MESSAGE: &str = "Missing Mistral API Key - A call is being made to Mistral but no key is set either in the environment variables or via params";

/// Build the complete OCR endpoint URL, de-duplicating a trailing `/v1`.
///
/// Blank/whitespace `api_base` is treated as absent (guard at resolution time).
pub fn complete_url(api_base: Option<&str>) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(MISTRAL_DEFAULT_API_BASE)
        .trim_end_matches('/');

    if base.ends_with("/v1") {
        format!("{base}/ocr")
    } else {
        format!("{base}/v1/ocr")
    }
}

/// Resolve the Mistral API key from the explicit param or the environment.
///
/// Blank/whitespace values are treated as absent. Returns `Error::Auth`
/// when no usable key is available.
///
/// Note: the env fallback only reads the process environment. Secret-manager
/// backends (AWS/Azure/GCP/Vault) are resolved on the Python side and passed in
/// via `api_key`; this fallback is a last resort for direct/standalone use.
pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(MISTRAL_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .ok_or_else(|| Error::Auth(MISSING_KEY_MESSAGE.to_string()))
}

pub struct MistralOcrConfig;

pub const MISTRAL_OCR_CONFIG: MistralOcrConfig = MistralOcrConfig;

impl OcrProviderConfig for MistralOcrConfig {
    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        SUPPORTED_OCR_PARAMS
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> Result<OcrRequestData, Error> {
        if !document.is_object() {
            return Err(Error::InvalidType {
                expected: "object",
                actual: json_type_name(&document),
            });
        }

        let mut data = Map::new();
        data.insert("model".to_string(), Value::String(model.to_string()));
        data.insert("document".to_string(), document);
        for (param, value) in optional_params {
            data.insert(param, value);
        }

        Ok(OcrRequestData {
            data: Value::Object(data),
            files: None,
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<OcrResponseData, Error> {
        let response_object = response_json
            .as_object()
            .ok_or_else(|| Error::InvalidType {
                expected: "object",
                actual: json_type_name(&response_json),
            })?;

        let pages = response_object
            .get("pages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let model = response_object
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(model)
            .to_string();
        let document_annotation = response_object.get("document_annotation").cloned();
        let usage_info = response_object.get("usage_info").cloned();

        Ok(OcrResponseData {
            pages,
            model,
            document_annotation,
            usage_info,
            object: "ocr".to_string(),
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_url(api_base))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        resolve_api_key(api_key, env_lookup)
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn supported_ocr_params() -> &'static [&'static str] {
    MISTRAL_OCR_CONFIG.supported_ocr_params()
}

pub fn map_ocr_params(non_default_params: &Map<String, Value>) -> Map<String, Value> {
    MISTRAL_OCR_CONFIG.map_ocr_params(non_default_params)
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn transform_ocr_request(
    model: &str,
    document: Value,
    optional_params: Map<String, Value>,
) -> Result<OcrRequestData, Error> {
    MISTRAL_OCR_CONFIG.transform_ocr_request(model, document, optional_params)
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn transform_ocr_response(model: &str, response_json: Value) -> Result<OcrResponseData, Error> {
    MISTRAL_OCR_CONFIG.transform_ocr_response(model, response_json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_header_is_a_supported_ocr_param() {
        assert!(supported_ocr_params().contains(&"extract_header"));
    }

    #[test]
    fn extract_footer_is_a_supported_ocr_param() {
        assert!(supported_ocr_params().contains(&"extract_footer"));
    }

    #[test]
    fn existing_ocr_params_remain_supported() {
        for param in [
            "pages",
            "include_image_base64",
            "image_limit",
            "image_min_size",
            "bbox_annotation_format",
            "document_annotation_format",
        ] {
            assert!(supported_ocr_params().contains(&param));
        }
    }

    #[test]
    fn map_ocr_params_forwards_extract_header() {
        let params = json!({"extract_header": true});
        assert_eq!(
            map_ocr_params(params.as_object().unwrap()),
            params.as_object().unwrap().clone()
        );
    }

    #[test]
    fn map_ocr_params_forwards_extract_footer() {
        let params = json!({"extract_footer": true});
        assert_eq!(
            map_ocr_params(params.as_object().unwrap()),
            params.as_object().unwrap().clone()
        );
    }

    #[test]
    fn map_ocr_params_forwards_extract_header_and_footer() {
        let params = json!({"extract_header": true, "extract_footer": false});
        assert_eq!(
            map_ocr_params(params.as_object().unwrap()),
            params.as_object().unwrap().clone()
        );
    }

    #[test]
    fn map_ocr_params_drops_unknown_params() {
        let params = json!({"extract_header": true, "unsupported_param": "value"});
        let mapped = map_ocr_params(params.as_object().unwrap());
        assert_eq!(mapped.get("extract_header"), Some(&json!(true)));
        assert!(!mapped.contains_key("unsupported_param"));
    }

    #[test]
    fn new_ocr_params_are_supported() {
        for param in [
            "table_format",
            "confidence_scores_granularity",
            "document_annotation_prompt",
            "include_blocks",
            "id",
        ] {
            assert!(supported_ocr_params().contains(&param));
        }
    }

    #[test]
    fn map_ocr_params_forwards_new_ocr_params() {
        for (param, value) in [
            ("table_format", json!("html")),
            ("confidence_scores_granularity", json!("word")),
            (
                "document_annotation_prompt",
                json!("Extract all invoice line items"),
            ),
            ("include_blocks", json!(true)),
            ("id", json!("req-123")),
        ] {
            let params = json!({param: value});
            assert_eq!(
                map_ocr_params(params.as_object().unwrap()),
                params.as_object().unwrap().clone()
            );
        }
    }

    #[test]
    fn transform_ocr_request_includes_each_optional_param() {
        let document = json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        });
        for (param, value) in [
            ("table_format", json!("html")),
            ("confidence_scores_granularity", json!("word")),
            (
                "document_annotation_prompt",
                json!("Extract all invoice line items"),
            ),
            ("id", json!("req-123")),
            ("extract_header", json!(true)),
            ("include_blocks", json!(true)),
            ("pages", json!([0, 1])),
        ] {
            let result = transform_ocr_request(
                "mistral-ocr-latest",
                document.clone(),
                json!({param: value}).as_object().unwrap().clone(),
            )
            .expect("request should transform");
            assert_eq!(result.data.get(param), Some(&value));
            assert_eq!(result.data.get("model"), Some(&json!("mistral-ocr-latest")));
            assert_eq!(result.data.get("document"), Some(&document));
            assert_eq!(result.files, None);
        }
    }

    #[test]
    fn transform_ocr_request_includes_multiple_new_params() {
        let document = json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        });
        let optional_params = json!({
            "table_format": "html",
            "confidence_scores_granularity": "page",
            "extract_header": true
        })
        .as_object()
        .unwrap()
        .clone();
        let result = transform_ocr_request("mistral-ocr-latest", document, optional_params)
            .expect("request should transform");
        assert_eq!(result.data.get("table_format"), Some(&json!("html")));
        assert_eq!(
            result.data.get("confidence_scores_granularity"),
            Some(&json!("page"))
        );
        assert_eq!(result.data.get("extract_header"), Some(&json!(true)));
    }

    #[test]
    fn transform_ocr_response_preserves_blocks_and_confidence_scores() {
        let blocks = json!([{"type": "title", "content": "Invoice"}]);
        let confidence_scores = json!({"page": 0.98});
        let response = json!({
            "pages": [{"index": 0, "markdown": "# Invoice", "blocks": blocks, "confidence_scores": confidence_scores}],
            "model": "mistral-ocr-4-0",
            "usage_info": {"pages_processed": 1}
        });
        let result =
            transform_ocr_response("mistral-ocr-4-0", response).expect("response should transform");
        assert_eq!(result.pages[0].get("blocks"), Some(&blocks));
        assert_eq!(
            result.pages[0].get("confidence_scores"),
            Some(&confidence_scores)
        );
    }

    #[test]
    fn transform_ocr_response_preserves_ocr4_page_fields() {
        let response = json!({
            "pages": [{"index": 0, "markdown": "table page", "tables": [{"rows": 2, "cols": 3}], "hyperlinks": ["https://example.com"], "header": "Acme Corp", "footer": "Page 1"}],
            "model": "mistral-ocr-4-0",
            "usage_info": {"pages_processed": 1}
        });
        let result = transform_ocr_response("mistral-ocr-4-0", response.clone())
            .expect("response should transform");
        assert_eq!(result.pages[0], response["pages"][0]);
    }

    #[test]
    fn transform_ocr_request_rejects_non_object_document() {
        let err = transform_ocr_request("mistral-ocr-latest", json!("bad"), Map::new())
            .expect_err("string document should be rejected");

        assert_eq!(
            err,
            Error::InvalidType {
                expected: "object",
                actual: "string",
            }
        );
    }

    #[test]
    fn transform_ocr_response_normalizes_mistral_json() {
        let response = json!({
            "pages": [{"index": 0, "markdown": "hello"}],
            "model": "mistral-ocr-2505-completion",
            "document_annotation": null,
            "usage_info": {"pages_processed": 1}
        });

        let result = transform_ocr_response("mistral-ocr-latest", response)
            .expect("response should transform");

        assert_eq!(result.pages, vec![json!({"index": 0, "markdown": "hello"})]);
        assert_eq!(result.model, "mistral-ocr-2505-completion");
        assert_eq!(result.document_annotation, Some(Value::Null));
        assert_eq!(result.usage_info, Some(json!({"pages_processed": 1})));
        assert_eq!(result.object, "ocr");
    }

    #[test]
    fn complete_url_defaults_and_dedupes_v1() {
        assert_eq!(complete_url(None), "https://api.mistral.ai/v1/ocr");
        assert_eq!(complete_url(Some("   ")), "https://api.mistral.ai/v1/ocr");
        assert_eq!(
            complete_url(Some("https://proxy.internal")),
            "https://proxy.internal/v1/ocr"
        );
        assert_eq!(
            complete_url(Some("https://proxy.internal/v1/")),
            "https://proxy.internal/v1/ocr"
        );
    }

    #[test]
    fn resolve_api_key_prefers_param_then_env() {
        let no_env = |_: &str| None;
        assert_eq!(
            resolve_api_key(Some("sk-param"), &no_env).unwrap(),
            "sk-param"
        );

        let with_env = |key: &str| (key == MISTRAL_API_KEY_ENV).then(|| "sk-env".to_string());
        assert_eq!(resolve_api_key(None, &with_env).unwrap(), "sk-env");
        // Blank param falls through to the environment.
        assert_eq!(resolve_api_key(Some("  "), &with_env).unwrap(), "sk-env");
    }

    #[test]
    fn resolve_api_key_errors_when_absent() {
        let err = resolve_api_key(None, &|_| None).expect_err("missing key should error");
        assert_eq!(err, Error::Auth(MISSING_KEY_MESSAGE.to_string()));
    }
}
