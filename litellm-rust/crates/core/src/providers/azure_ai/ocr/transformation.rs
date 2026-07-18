use std::collections::BTreeSet;

use crate::error::{json_type_name, CoreError, CoreResult};
use crate::ocr::transformation::{OcrAuthStrategy, OcrProviderConfig, OcrResponseHandling};
use crate::ocr::types::{OcrRequestData, OcrResponseData};
use serde_json::{json, Map, Value};

use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;

const AZURE_AI_API_KEY_ENV: &str = "AZURE_AI_API_KEY";
const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";
const AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_API_KEY";
const AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT";
const AZURE_DOCUMENT_INTELLIGENCE_API_VERSION: &str = "2024-11-30";
const AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI: i64 = 96;

const AZURE_DOCUMENT_INTELLIGENCE_SUPPORTED_OCR_PARAMS: &[&str] = &["pages"];

pub struct AzureAiOcrConfig;
pub struct AzureDocumentIntelligenceOcrConfig;

pub const AZURE_AI_OCR_CONFIG: AzureAiOcrConfig = AzureAiOcrConfig;
pub const AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG: AzureDocumentIntelligenceOcrConfig =
    AzureDocumentIntelligenceOcrConfig;

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn resolve_value(
    explicit: Option<&str>,
    env_name: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    missing_message: &str,
) -> CoreResult<String> {
    non_empty(explicit)
        .map(str::to_string)
        .or_else(|| env_lookup(env_name).filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| CoreError::Auth(missing_message.to_string()))
}

pub fn resolve_azure_ai_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    resolve_value(
        api_key,
        AZURE_AI_API_KEY_ENV,
        env_lookup,
        "Missing Azure AI API Key - A call is being made to Azure AI but no key is set either in the environment variables or via params",
    )
}

pub fn resolve_azure_ai_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    resolve_value(
        api_base,
        AZURE_AI_API_BASE_ENV,
        env_lookup,
        "Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter",
    )
}

pub fn complete_azure_ai_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let base = resolve_azure_ai_api_base(api_base, env_lookup)?;
    Ok(format!(
        "{}/providers/mistral/azure/ocr",
        base.trim_end_matches('/')
    ))
}

pub fn resolve_document_intelligence_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    resolve_value(
        api_key,
        AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV,
        env_lookup,
        "Missing Azure Document Intelligence API Key - Set AZURE_DOCUMENT_INTELLIGENCE_API_KEY environment variable or pass api_key parameter",
    )
}

pub fn resolve_document_intelligence_endpoint(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    resolve_value(
        api_base,
        AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV,
        env_lookup,
        "Missing Azure Document Intelligence Endpoint - Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable or pass api_base parameter",
    )
}

fn encode_model_id(model: &str) -> String {
    let model_id = model.rsplit('/').next().unwrap_or(model);
    model_id
        .bytes()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                vec![byte as char]
            }
            _ => format!("%{byte:02X}").chars().collect(),
        })
        .collect()
}

fn pages_token_is_valid(token: &str) -> bool {
    let mut parts = token.split('-');
    let Some(start) = parts.next() else {
        return false;
    };
    if start.is_empty() || !start.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(end) => {
            !end.is_empty() && end.chars().all(|ch| ch.is_ascii_digit()) && parts.next().is_none()
        }
    }
}

fn normalize_pages_param(pages: &Value) -> CoreResult<Option<String>> {
    match pages {
        Value::String(value) => {
            let normalized = value
                .split(',')
                .map(str::trim)
                .collect::<Vec<_>>()
                .join(",");
            if normalized.split(',').all(pages_token_is_valid) {
                Ok(Some(normalized))
            } else {
                Err(CoreError::InvalidRequest(format!(
                    "Invalid `pages` string for Azure Document Intelligence: {value:?}. Expected format like '1-3,5,7-9'."
                )))
            }
        }
        Value::Array(values) => {
            if values.is_empty() {
                return Ok(None);
            }
            if values.iter().all(Value::is_i64) {
                let mut pages = BTreeSet::new();
                for value in values {
                    let page = value.as_i64().expect("checked is_i64");
                    if page < 0 {
                        return Err(CoreError::InvalidRequest(
                            "`pages` integers must be >= 0 (Mistral 0-based indices)".to_string(),
                        ));
                    }
                    pages.insert(page + 1);
                }
                return Ok(Some(
                    pages
                        .into_iter()
                        .map(|page| page.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ));
            }
            if values.iter().all(Value::is_string) {
                let normalized = values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .collect::<Vec<_>>()
                    .join(",");
                if normalized.split(',').all(pages_token_is_valid) {
                    return Ok(Some(normalized));
                }
                return Err(CoreError::InvalidRequest(format!(
                    "Invalid `pages` list for Azure Document Intelligence: {values:?}. Expected tokens like '1' or '3-5'."
                )));
            }
            Err(CoreError::InvalidRequest(
                "`pages` must be a list[int] (0-based, Mistral-style) or a string like '1-3,5,7-9'."
                    .to_string(),
            ))
        }
        _ => Err(CoreError::InvalidRequest(
            "`pages` must be a list[int] (0-based, Mistral-style) or a string like '1-3,5,7-9'."
                .to_string(),
        )),
    }
}

pub fn complete_document_intelligence_url(
    api_base: Option<&str>,
    model: &str,
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let endpoint = resolve_document_intelligence_endpoint(api_base, env_lookup)?;
    let mut url = format!(
        "{}/documentintelligence/documentModels/{}:analyze?api-version={}",
        endpoint.trim_end_matches('/'),
        encode_model_id(model),
        AZURE_DOCUMENT_INTELLIGENCE_API_VERSION
    );

    if let Some(pages) = optional_params.get("pages") {
        if let Some(normalized) = normalize_pages_param(pages)? {
            url.push_str("&pages=");
            url.push_str(&normalized);
        }
    }

    Ok(url)
}

fn document_url_from_mistral_document(document: &Value) -> CoreResult<&str> {
    let object = document.as_object().ok_or_else(|| CoreError::InvalidType {
        expected: "object",
        actual: json_type_name(document),
    })?;
    let doc_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(CoreError::MissingField("document.type"))?;
    let field_name = match doc_type {
        "document_url" => "document_url",
        "image_url" => "image_url",
        other => {
            return Err(CoreError::InvalidRequest(format!(
                "Invalid document type: {other}. Must be 'document_url' or 'image_url'"
            )))
        }
    };
    object
        .get(field_name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(CoreError::MissingField(field_name))
}

fn extract_base64_from_data_uri(data_uri: &str) -> &str {
    data_uri
        .split_once(',')
        .map(|(_, data)| data)
        .unwrap_or(data_uri)
}

fn page_markdown(page: &Map<String, Value>) -> String {
    page.get("lines")
        .and_then(Value::as_array)
        .map(|lines| {
            lines
                .iter()
                .filter_map(|line| line.get("content").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn page_dimensions(page: &Map<String, Value>) -> Value {
    let width = page.get("width").and_then(Value::as_f64).unwrap_or(8.5);
    let height = page.get("height").and_then(Value::as_f64).unwrap_or(11.0);
    let unit = page.get("unit").and_then(Value::as_str).unwrap_or("inch");
    let (width, height) = if unit == "inch" {
        (
            (width * AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI as f64) as i64,
            (height * AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI as f64) as i64,
        )
    } else {
        (width as i64, height as i64)
    };
    json!({
        "width": width,
        "height": height,
        "dpi": AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI,
    })
}

impl OcrProviderConfig for AzureAiOcrConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        MISTRAL_OCR_CONFIG.supported_ocr_params()
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        MISTRAL_OCR_CONFIG.transform_ocr_request(model, document, optional_params)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> CoreResult<OcrResponseData> {
        MISTRAL_OCR_CONFIG.transform_ocr_response(model, response_json)
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        complete_azure_ai_url(api_base, env_lookup)
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_azure_ai_api_key(api_key, env_lookup)
    }

    fn requires_data_uri_document(&self) -> bool {
        true
    }
}

impl OcrProviderConfig for AzureDocumentIntelligenceOcrConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        AZURE_DOCUMENT_INTELLIGENCE_SUPPORTED_OCR_PARAMS
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: Value,
        _optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        let document_url = document_url_from_mistral_document(&document)?;
        let mut data = Map::new();
        if document_url.starts_with("data:") {
            data.insert(
                "base64Source".to_string(),
                Value::String(extract_base64_from_data_uri(document_url).to_string()),
            );
        } else {
            data.insert(
                "urlSource".to_string(),
                Value::String(document_url.to_string()),
            );
        }
        Ok(OcrRequestData {
            data: Value::Object(data),
            files: None,
        })
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> CoreResult<OcrResponseData> {
        let response = response_json
            .as_object()
            .ok_or_else(|| CoreError::unexpected_response_type(&response_json))?;
        let status = response
            .get("status")
            .and_then(Value::as_str)
            .ok_or(CoreError::missing_response_field("status"))?;
        if status != "succeeded" {
            return Err(CoreError::InvalidResponse(format!(
                "Azure Document Intelligence analysis failed with status: {status}"
            )));
        }

        let azure_pages = response
            .get("analyzeResult")
            .and_then(|result| result.get("pages"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let pages = azure_pages
            .iter()
            .filter_map(Value::as_object)
            .map(|page| {
                let page_number = page.get("pageNumber").and_then(Value::as_i64).unwrap_or(1);
                json!({
                    "index": page_number - 1,
                    "markdown": page_markdown(page),
                    "dimensions": page_dimensions(page),
                })
            })
            .collect::<Vec<_>>();

        Ok(OcrResponseData {
            usage_info: Some(json!({
                "pages_processed": pages.len(),
                "doc_size_bytes": null,
            })),
            pages,
            model: model.to_string(),
            document_annotation: None,
            object: "ocr".to_string(),
        })
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        complete_document_intelligence_url(api_base, model, optional_params, env_lookup)
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_document_intelligence_api_key(api_key, env_lookup)
    }

    fn auth_strategy(&self) -> OcrAuthStrategy {
        OcrAuthStrategy::Header("Ocp-Apim-Subscription-Key")
    }

    fn response_handling(&self) -> OcrResponseHandling {
        OcrResponseHandling::AzureDocumentIntelligencePoll
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azure_ai_reuses_mistral_body_transform() {
        let body = AZURE_AI_OCR_CONFIG
            .transform_ocr_request(
                "pixtral-12b-2409",
                json!({"type": "document_url", "document_url": "data:application/pdf;base64,abc"}),
                serde_json::Map::from_iter([("include_image_base64".to_string(), json!(true))]),
            )
            .expect("request transforms")
            .data;

        assert_eq!(body["model"], "pixtral-12b-2409");
        assert_eq!(body["include_image_base64"], true);
        assert_eq!(
            body["document"]["document_url"],
            "data:application/pdf;base64,abc"
        );
    }

    #[test]
    fn document_intelligence_url_normalizes_zero_based_pages() {
        let params = serde_json::Map::from_iter([("pages".to_string(), json!([2, 0, 2]))]);
        let url = complete_document_intelligence_url(
            Some("https://example.cognitiveservices.azure.com/"),
            "azure_ai/doc-intelligence/prebuilt-layout",
            &params,
            &|_| None,
        )
        .expect("url builds");

        assert_eq!(
            url,
            "https://example.cognitiveservices.azure.com/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30&pages=1,3"
        );
    }

    #[test]
    fn document_intelligence_request_uses_base64_source_for_data_uri() {
        let body = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_request(
                "prebuilt-read",
                json!({"type": "document_url", "document_url": "data:application/pdf;base64,abc123"}),
                Map::new(),
            )
            .expect("request transforms")
            .data;

        assert_eq!(body, json!({"base64Source": "abc123"}));
    }

    #[test]
    fn document_intelligence_response_normalizes_pages() {
        let response = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response(
                "prebuilt-layout",
                json!({
                    "status": "succeeded",
                    "analyzeResult": {
                        "pages": [{
                            "pageNumber": 2,
                            "width": 8.5,
                            "height": 11,
                            "unit": "inch",
                            "lines": [{"content": "hello"}, {"content": "world"}]
                        }]
                    }
                }),
            )
            .expect("response transforms");

        assert_eq!(response.pages[0]["index"], 1);
        assert_eq!(response.pages[0]["markdown"], "hello\nworld");
        assert_eq!(response.pages[0]["dimensions"]["width"], 816);
        assert_eq!(
            response.usage_info,
            Some(json!({"pages_processed": 1, "doc_size_bytes": null}))
        );
    }

    #[test]
    fn document_intelligence_response_missing_status_is_invalid_response() {
        let err = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response("prebuilt-layout", json!({"analyzeResult": {"pages": []}}))
            .expect_err("missing status should be rejected");

        assert!(matches!(err, CoreError::InvalidResponse(_)));
        assert_eq!(err.public_status_code(), Some(500));
    }

    #[test]
    fn document_intelligence_response_non_object_is_invalid_response() {
        let err = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response("prebuilt-layout", json!("boom"))
            .expect_err("non-object provider response should be rejected");

        assert!(matches!(err, CoreError::InvalidResponse(_)));
        assert_eq!(err.public_status_code(), Some(500));
    }
}
