use crate::error::{CoreError, CoreResult};
use crate::ocr::transformation::OcrProviderConfig;
use crate::ocr::types::{
    MistralOcrOptionalParams, MistralOcrResponseData, OcrDocument, OcrRequestData, OcrResponseData,
};

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

pub struct MistralOcrConfig;

pub const MISTRAL_OCR_CONFIG: MistralOcrConfig = MistralOcrConfig;

impl OcrProviderConfig for MistralOcrConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        SUPPORTED_OCR_PARAMS
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: MistralOcrOptionalParams,
    ) -> CoreResult<OcrRequestData> {
        Ok(OcrRequestData {
            model: model.to_string(),
            document,
            optional_params,
        })
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: MistralOcrResponseData,
    ) -> CoreResult<OcrResponseData> {
        Ok(OcrResponseData {
            pages: response_json.pages,
            model: response_json.model.unwrap_or_else(|| model.to_string()),
            document_annotation: response_json.document_annotation,
            usage_info: response_json.usage_info,
            object: "ocr".to_string(),
        })
    }
}

pub fn supported_ocr_params() -> &'static [&'static str] {
    MISTRAL_OCR_CONFIG.supported_ocr_params()
}

pub fn map_ocr_params(non_default_params: &MistralOcrOptionalParams) -> MistralOcrOptionalParams {
    MISTRAL_OCR_CONFIG.map_ocr_params(non_default_params)
}

pub fn transform_ocr_request(
    model: &str,
    document: OcrDocument,
    optional_params: MistralOcrOptionalParams,
) -> CoreResult<OcrRequestData> {
    MISTRAL_OCR_CONFIG.transform_ocr_request(model, document, optional_params)
}

pub fn transform_ocr_response(
    model: &str,
    response_json: MistralOcrResponseData,
) -> CoreResult<OcrResponseData> {
    MISTRAL_OCR_CONFIG.transform_ocr_response(model, response_json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::types::{OcrFieldValue, OcrObject};

    #[test]
    fn supported_params_match_python_mistral_ocr_config() {
        assert_eq!(
            supported_ocr_params(),
            &[
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
        );
    }

    #[test]
    fn map_ocr_params_drops_unknown_params() {
        let params = MistralOcrOptionalParams {
            extract_header: Some(true),
            pages: Some(vec![0, 1]),
            ..Default::default()
        };
        let mapped = map_ocr_params(&params);

        assert_eq!(mapped.extract_header, Some(true));
        assert_eq!(mapped.pages, Some(vec![0, 1]));
    }

    #[test]
    fn transform_ocr_request_builds_mistral_body() {
        let document = OcrDocument::DocumentUrl {
            document_url: "https://example.com/doc.pdf".to_string(),
        };
        let optional_params = MistralOcrOptionalParams {
            include_image_base64: Some(true),
            table_format: Some("html".to_string()),
            ..Default::default()
        };

        let result = transform_ocr_request("mistral-ocr-latest", document.clone(), optional_params)
            .expect("request should transform");

        assert_eq!(result.model, "mistral-ocr-latest");
        assert_eq!(result.document, document);
        assert_eq!(result.optional_params.include_image_base64, Some(true));
        assert_eq!(result.optional_params.table_format.as_deref(), Some("html"));
    }

    #[test]
    fn transform_ocr_response_normalizes_mistral_json() {
        let page = OcrObject::from([
            ("index".to_string(), OcrFieldValue::Number(0.into())),
            (
                "markdown".to_string(),
                OcrFieldValue::String("hello".to_string()),
            ),
        ]);
        let usage_info = OcrObject::from([(
            "pages_processed".to_string(),
            OcrFieldValue::Number(1.into()),
        )]);
        let response = MistralOcrResponseData {
            pages: vec![page.clone()],
            model: Some("mistral-ocr-2505-completion".to_string()),
            document_annotation: Some(OcrFieldValue::Null),
            usage_info: Some(usage_info.clone()),
        };

        let result = transform_ocr_response("mistral-ocr-latest", response)
            .expect("response should transform");

        assert_eq!(result.pages, vec![page]);
        assert_eq!(result.model, "mistral-ocr-2505-completion");
        assert_eq!(result.document_annotation, Some(OcrFieldValue::Null));
        assert_eq!(result.usage_info, Some(usage_info));
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
        assert_eq!(err, CoreError::Auth(MISSING_KEY_MESSAGE.to_string()));
    }
}
