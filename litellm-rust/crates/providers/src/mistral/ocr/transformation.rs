//! Mistral OCR — typed request/response transforms.
//!
//! Pure functions: build the request body and parse the response. No network,
//! no env reads (the env fallback for the key takes a closure). Serde produces
//! the exact wire JSON, so there is no hand-built JSON here.

use litellm_core::error::{CoreError, CoreResult};
use litellm_core::ocr::types::{OcrDocument, OcrParams, OcrResponse};
use serde::Serialize;

/// Default Mistral API base, used when the caller does not override `api_base`.
pub const MISTRAL_DEFAULT_API_BASE: &str = "https://api.mistral.ai/v1";

/// Environment variable holding the Mistral API key.
pub const MISTRAL_API_KEY_ENV: &str = "MISTRAL_API_KEY";

/// Error message raised when no Mistral API key can be resolved.
pub const MISSING_KEY_MESSAGE: &str = "Missing Mistral API Key - A call is being made to Mistral but no key is set either in the environment variables or via params";

/// The Mistral OCR request body. `model` + `document` + the set OCR params,
/// flattened to the top level exactly as Mistral expects.
#[derive(Debug, Serialize)]
pub struct MistralRequestBody<'a> {
    pub model: &'a str,
    pub document: &'a OcrDocument,
    #[serde(flatten)]
    pub params: &'a OcrParams,
}

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
/// Blank/whitespace values are treated as absent. Returns `CoreError::Auth`
/// when no usable key is available.
///
/// Note: the env fallback only reads the process environment. Secret-manager
/// backends (AWS/Azure/GCP/Vault) are resolved on the Python side and passed in
/// via `api_key`; this fallback is a last resort for direct/standalone use.
pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(MISTRAL_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .ok_or_else(|| CoreError::Auth(MISSING_KEY_MESSAGE.to_string()))
}

/// Build the typed Mistral request body (borrows the request fields).
pub fn request_body<'a>(
    model: &'a str,
    document: &'a OcrDocument,
    params: &'a OcrParams,
) -> MistralRequestBody<'a> {
    MistralRequestBody {
        model,
        document,
        params,
    }
}

/// Parse a Mistral OCR response body into the standardized [`OcrResponse`].
///
/// Mistral already returns the canonical shape, so this is a typed deserialize.
/// Unknown fields are dropped (serde default). Falls back to the request `model`
/// when the response omits it, matching the Python path.
pub fn parse_response(model: &str, body: &str) -> CoreResult<OcrResponse> {
    let mut response: OcrResponse = serde_json::from_str(body)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    if response.model.is_empty() {
        response.model = model.to_string();
    }
    if response.object.is_empty() {
        response.object = "ocr".to_string();
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_core::ocr::types::TableFormat;
    use serde_json::json;

    fn document() -> OcrDocument {
        OcrDocument::DocumentUrl {
            document_url: "https://example.com/doc.pdf".to_string(),
        }
    }

    #[test]
    fn request_body_serializes_to_mistral_wire_shape() {
        let doc = document();
        let params = OcrParams {
            include_image_base64: Some(true),
            table_format: Some(TableFormat::Html),
            ..Default::default()
        };

        let body = request_body("mistral-ocr-latest", &doc, &params);
        let value = serde_json::to_value(&body).expect("serializes");

        assert_eq!(
            value,
            json!({
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": "https://example.com/doc.pdf"
                },
                "include_image_base64": true,
                "table_format": "html"
            })
        );
    }

    #[test]
    fn request_body_omits_unset_params() {
        let doc = document();
        let params = OcrParams::default();
        let value = serde_json::to_value(request_body("m", &doc, &params)).expect("serializes");

        // Only model + document; no param keys leak through.
        assert_eq!(
            value,
            json!({
                "model": "m",
                "document": {
                    "type": "document_url",
                    "document_url": "https://example.com/doc.pdf"
                }
            })
        );
    }

    #[test]
    fn parse_response_normalizes_and_drops_unknown_fields() {
        let body = r#"{
            "pages": [{"index": 0, "markdown": "hello", "secret_field": "dropped"}],
            "model": "mistral-ocr-2505-completion",
            "document_annotation": null,
            "usage_info": {"pages_processed": 1, "extra": "dropped"},
            "object": "ocr"
        }"#;

        let response = parse_response("mistral-ocr-latest", body).expect("parses");

        assert_eq!(response.pages.len(), 1);
        assert_eq!(response.pages[0].index, 0);
        assert_eq!(response.pages[0].markdown, "hello");
        assert_eq!(response.model, "mistral-ocr-2505-completion");
        assert_eq!(response.object, "ocr");
        assert_eq!(
            response.usage_info.as_ref().and_then(|u| u.pages_processed),
            Some(1)
        );
    }

    #[test]
    fn parse_response_falls_back_to_request_model_and_default_object() {
        // No model, no object in the body.
        let body = r#"{"pages": []}"#;
        let response = parse_response("mistral-ocr-latest", body).expect("parses");

        assert_eq!(response.model, "mistral-ocr-latest");
        assert_eq!(response.object, "ocr");
        assert!(response.pages.is_empty());
    }

    #[test]
    fn parse_response_rejects_non_json() {
        let err = parse_response("m", "not json").expect_err("should reject");
        assert!(matches!(err, CoreError::InvalidResponse(_)));
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
        assert_eq!(resolve_api_key(Some("  "), &with_env).unwrap(), "sk-env");
    }

    #[test]
    fn resolve_api_key_errors_when_absent() {
        let err = resolve_api_key(None, &|_| None).expect_err("missing key should error");
        assert_eq!(err, CoreError::Auth(MISSING_KEY_MESSAGE.to_string()));
    }
}
