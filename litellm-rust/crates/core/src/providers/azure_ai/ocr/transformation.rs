use std::collections::BTreeSet;

use crate::error::{Error, json_type_name};
use crate::ocr::transformation::{OcrAuthStrategy, OcrProviderConfig, OcrResponseHandling};
use crate::ocr::types::{OcrRequestData, OcrResponseData};
use serde_json::{Map, Value, json};

use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;

const AZURE_AI_API_KEY_ENV: &str = "AZURE_AI_API_KEY";
const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";
const AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_API_KEY";
const AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT";
const AZURE_DOCUMENT_INTELLIGENCE_API_VERSION: &str = "2024-11-30";
const AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI: i64 = 96;

const AZURE_DOCUMENT_INTELLIGENCE_SUPPORTED_OCR_PARAMS: &[&str] =
    &["pages", "features", "req_format"];

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
) -> Result<String, Error> {
    non_empty(explicit)
        .map(str::to_string)
        .or_else(|| env_lookup(env_name).filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| Error::Auth(missing_message.to_string()))
}

pub fn resolve_azure_ai_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
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
) -> Result<String, Error> {
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
) -> Result<String, Error> {
    let base = resolve_azure_ai_api_base(api_base, env_lookup)?;
    Ok(format!(
        "{}/providers/mistral/azure/ocr",
        base.trim_end_matches('/')
    ))
}

pub fn resolve_document_intelligence_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
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
) -> Result<String, Error> {
    resolve_value(
        api_base,
        AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV,
        env_lookup,
        "Missing Azure Document Intelligence Endpoint - Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable or pass api_base parameter",
    )
}

fn prepend_auth_header(
    headers: Vec<(String, String)>,
    name: &str,
    value: String,
) -> Vec<(String, String)> {
    std::iter::once((name.to_string(), value))
        .chain(headers)
        .collect()
}

pub fn validate_azure_ai_environment(
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    azure_ad_token: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, Error> {
    if crate::http_utils::has_header(&headers, "Authorization")
        || crate::http_utils::has_header(&headers, "Api-Key")
    {
        return Ok(headers);
    }
    if let Ok(api_key) = resolve_azure_ai_api_key(api_key, env_lookup) {
        return Ok(prepend_auth_header(headers, "Api-Key", api_key));
    }
    non_empty(azure_ad_token)
        .map(|token| prepend_auth_header(headers, "Authorization", format!("Bearer {token}")))
        .ok_or_else(|| {
            Error::Auth(
                "Missing Azure AI credentials - set AZURE_AI_API_KEY or provide azure_ad_token"
                    .to_string(),
            )
        })
}

pub fn validate_document_intelligence_environment(
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    azure_ad_token: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, Error> {
    if crate::http_utils::has_header(&headers, "Authorization")
        || crate::http_utils::has_header(&headers, "Ocp-Apim-Subscription-Key")
    {
        return Ok(headers);
    }
    if let Ok(api_key) = resolve_document_intelligence_api_key(api_key, env_lookup) {
        return Ok(prepend_auth_header(
            headers,
            "Ocp-Apim-Subscription-Key",
            api_key,
        ));
    }
    non_empty(azure_ad_token)
        .map(|token| prepend_auth_header(headers, "Authorization", format!("Bearer {token}")))
        .ok_or_else(|| {
            Error::Auth(
                "Missing Azure Document Intelligence credentials - set AZURE_DOCUMENT_INTELLIGENCE_API_KEY or provide azure_ad_token"
                    .to_string(),
            )
        })
}

fn encode_model_id(model: &str) -> Result<String, Error> {
    let model_id = model.rsplit('/').next().unwrap_or(model);
    if matches!(model_id, "." | "..") {
        return Err(Error::InvalidRequest(
            "model_id cannot be a dot path segment".to_string(),
        ));
    }
    Ok(model_id
        .bytes()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                vec![byte as char]
            }
            _ => format!("%{byte:02X}").chars().collect(),
        })
        .collect())
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

fn normalize_pages_param(pages: &Value) -> Result<Option<String>, Error> {
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
                Err(Error::InvalidRequest(format!(
                    "Invalid `pages` string for Azure Document Intelligence: {value:?}. Expected format like '1-3,5,7-9'."
                )))
            }
        }
        Value::Array(values) => {
            if values.is_empty() {
                return Ok(None);
            }
            if values.iter().any(Value::is_boolean) {
                return Err(Error::InvalidRequest(
                    "`pages` must be integers, not booleans".to_string(),
                ));
            }
            if values.iter().all(Value::is_i64) {
                let mut pages = BTreeSet::new();
                for value in values {
                    let page = value.as_i64().expect("checked is_i64");
                    if page < 0 {
                        return Err(Error::InvalidRequest(
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
                return Err(Error::InvalidRequest(format!(
                    "Invalid `pages` list for Azure Document Intelligence: {values:?}. Expected tokens like '1' or '3-5'."
                )));
            }
            Err(Error::InvalidRequest(
                "`pages` must be a list[int] (0-based, Mistral-style) or a string like '1-3,5,7-9'."
                    .to_string(),
            ))
        }
        _ => Err(Error::InvalidRequest(
            "`pages` must be a list[int] (0-based, Mistral-style) or a string like '1-3,5,7-9'."
                .to_string(),
        )),
    }
}

fn feature_token_is_valid(token: &str) -> bool {
    let Some((first, rest)) = token.as_bytes().split_first() else {
        return false;
    };
    first.is_ascii_alphabetic() && rest.iter().all(u8::is_ascii_alphanumeric)
}

fn invalid_features_error(features: &Value) -> Error {
    Error::InvalidRequest(format!(
        "Invalid `features` for Azure Document Intelligence: {features:?}. Expected a list of feature names or a comma-separated string like 'keyValuePairs' or 'keyValuePairs,languages'."
    ))
}

fn normalize_features_param(features: &Value) -> Result<Option<String>, Error> {
    let normalized = match features {
        Value::String(value) => value
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(","),
        Value::Array(values) if values.is_empty() => return Ok(None),
        Value::Array(values) => values
            .iter()
            .map(Value::as_str)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid_features_error(features))?
            .into_iter()
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(","),
        _ => return Err(invalid_features_error(features)),
    };

    if normalized.split(',').all(feature_token_is_valid) {
        Ok(Some(normalized))
    } else {
        Err(invalid_features_error(features))
    }
}

fn normalize_req_format(req_format: &Value) -> Result<String, Error> {
    match req_format.as_str() {
        Some(value @ ("native" | "litellm")) => Ok(value.to_string()),
        _ => Err(Error::InvalidRequest(format!(
            "Invalid `req_format` for Azure Document Intelligence: {req_format:?}. Expected 'native' or 'litellm'."
        ))),
    }
}

pub fn map_document_intelligence_ocr_params(
    non_default_params: &Map<String, Value>,
) -> Result<Map<String, Value>, Error> {
    let mut mapped = Map::new();
    if let Some(pages) = non_default_params.get("pages")
        && let Some(normalized) = normalize_pages_param(pages)?
    {
        mapped.insert("pages".to_string(), Value::String(normalized));
    }
    if let Some(features) = non_default_params.get("features")
        && let Some(normalized) = normalize_features_param(features)?
    {
        mapped.insert("features".to_string(), Value::String(normalized));
    }
    if let Some(req_format) = non_default_params.get("req_format") {
        mapped.insert(
            "req_format".to_string(),
            Value::String(normalize_req_format(req_format)?),
        );
    }
    Ok(mapped)
}

pub fn complete_document_intelligence_url(
    api_base: Option<&str>,
    model: &str,
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let endpoint = resolve_document_intelligence_endpoint(api_base, env_lookup)?;
    let mut url = format!(
        "{}/documentintelligence/documentModels/{}:analyze?api-version={}",
        endpoint.trim_end_matches('/'),
        encode_model_id(model)?,
        AZURE_DOCUMENT_INTELLIGENCE_API_VERSION
    );

    if let Some(pages) = optional_params.get("pages")
        && let Some(normalized) = normalize_pages_param(pages)?
    {
        url.push_str("&pages=");
        url.push_str(&normalized);
    }

    if let Some(features) = optional_params.get("features")
        && let Some(normalized) = normalize_features_param(features)?
    {
        url.push_str("&features=");
        url.push_str(&normalized);
    }

    if let Some(req_format) = optional_params.get("req_format") {
        normalize_req_format(req_format)?;
    }

    Ok(url)
}

fn document_url_from_mistral_document(document: &Value) -> Result<&str, Error> {
    let object = document.as_object().ok_or_else(|| Error::InvalidType {
        expected: "object",
        actual: json_type_name(document),
    })?;
    let doc_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(Error::MissingField("document.type"))?;
    let field_name = match doc_type {
        "document_url" => "document_url",
        "image_url" => "image_url",
        other => {
            return Err(Error::InvalidRequest(format!(
                "Invalid document type: {other}. Must be 'document_url' or 'image_url'"
            )));
        }
    };
    object
        .get(field_name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(Error::MissingField(field_name))
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

fn transform_document_intelligence_response(
    model: &str,
    response_json: Value,
    preserve_native_response: bool,
) -> Result<OcrResponseData, Error> {
    let response = response_json
        .as_object()
        .ok_or_else(|| Error::InvalidType {
            expected: "object",
            actual: json_type_name(&response_json),
        })?;
    let status = response
        .get("status")
        .and_then(Value::as_str)
        .ok_or(Error::MissingField("status"))?;
    if status != "succeeded" {
        return Err(Error::InvalidResponse(format!(
            "Azure Document Intelligence analysis failed with status: {status}"
        )));
    }

    let analyze_result = response.get("analyzeResult").and_then(Value::as_object);
    let azure_pages = analyze_result
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
    let extra_fields = ["content", "tables", "keyValuePairs"]
        .into_iter()
        .map(|field| {
            (
                field.to_string(),
                analyze_result
                    .and_then(|result| result.get(field))
                    .cloned()
                    .unwrap_or(Value::Null),
            )
        })
        .collect();

    Ok(OcrResponseData {
        usage_info: Some(json!({
            "pages_processed": pages.len(),
            "doc_size_bytes": null,
        })),
        pages,
        model: model.to_string(),
        document_annotation: None,
        object: "ocr".to_string(),
        extra_fields,
        provider_native_response: preserve_native_response.then_some(response_json),
    })
}

impl OcrProviderConfig for AzureAiOcrConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        MISTRAL_OCR_CONFIG.supported_ocr_params()
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> Result<OcrRequestData, Error> {
        MISTRAL_OCR_CONFIG.transform_ocr_request(model, document, optional_params)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<OcrResponseData, Error> {
        MISTRAL_OCR_CONFIG.transform_ocr_response(model, response_json)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        complete_azure_ai_url(api_base, env_lookup)
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        resolve_azure_ai_api_key(api_key, env_lookup)
    }

    fn requires_data_uri_document(&self) -> bool {
        true
    }
}

impl OcrProviderConfig for AzureDocumentIntelligenceOcrConfig {
    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        AZURE_DOCUMENT_INTELLIGENCE_SUPPORTED_OCR_PARAMS
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(&self, non_default_params: &Map<String, Value>) -> Map<String, Value> {
        map_document_intelligence_ocr_params(non_default_params).unwrap_or_else(|_| {
            non_default_params
                .iter()
                .filter(|(name, _)| {
                    AZURE_DOCUMENT_INTELLIGENCE_SUPPORTED_OCR_PARAMS.contains(&name.as_str())
                })
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect()
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        _model: &str,
        document: Value,
        _optional_params: Map<String, Value>,
    ) -> Result<OcrRequestData, Error> {
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

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<OcrResponseData, Error> {
        transform_document_intelligence_response(model, response_json, false)
    }

    fn transform_ocr_response_with_params(
        &self,
        model: &str,
        response_json: Value,
        optional_params: &Map<String, Value>,
    ) -> Result<OcrResponseData, Error> {
        transform_document_intelligence_response(
            model,
            response_json,
            optional_params.get("req_format").and_then(Value::as_str) == Some("native"),
        )
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        complete_document_intelligence_url(api_base, model, optional_params, env_lookup)
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
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
    use rstest::{fixture, rstest};

    const ENDPOINT: &str = "https://example.cognitiveservices.azure.com";

    #[fixture]
    fn document_intelligence_config() -> AzureDocumentIntelligenceOcrConfig {
        AzureDocumentIntelligenceOcrConfig
    }

    fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(header_name, _)| header_name.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    #[fixture]
    fn native_operation() -> Value {
        json!({
            "status": "succeeded",
            "createdDateTime": "2026-07-02T00:00:00Z",
            "lastUpdatedDateTime": "2026-07-02T00:00:05Z",
            "analyzeResult": {
                "content": "Invoice\nInvoice No: INV-12345\nTotal: $100.00",
                "pages": [{
                    "pageNumber": 1,
                    "width": 8.5,
                    "height": 11,
                    "unit": "inch",
                    "angle": 0.13,
                    "lines": [
                        {"content": "Invoice"},
                        {"content": "Invoice No: INV-12345"},
                        {"content": "Total: $100.00"}
                    ],
                    "words": [{"content": "Invoice", "confidence": 0.994}]
                }],
                "tables": [
                    {
                        "rowCount": 2,
                        "columnCount": 2,
                        "cells": [
                            {"kind": "columnHeader", "rowIndex": 0, "columnIndex": 0, "content": "Item"},
                            {"kind": "columnHeader", "rowIndex": 0, "columnIndex": 1, "content": "Price"},
                            {"rowIndex": 1, "columnIndex": 0, "content": "Widget"},
                            {"rowIndex": 1, "columnIndex": 1, "content": "$100.00"}
                        ]
                    },
                    {
                        "rowCount": 1,
                        "columnCount": 1,
                        "cells": [{"rowIndex": 0, "columnIndex": 0, "content": "Totals"}]
                    }
                ],
                "keyValuePairs": [
                    {
                        "key": {"content": "Invoice No"},
                        "value": {"content": "INV-12345"},
                        "confidence": 0.98
                    },
                    {
                        "key": {"content": "Total"},
                        "value": {"content": "$100.00"},
                        "confidence": 0.95
                    }
                ],
                "paragraphs": [{"content": "Invoice"}]
            }
        })
    }

    fn assert_native_fields_preserved(response: &OcrResponseData, operation: &Value) {
        let analyze_result = &operation["analyzeResult"];

        assert_eq!(response.extra_fields["content"], analyze_result["content"]);
        assert_eq!(response.extra_fields["tables"], analyze_result["tables"]);
        assert_eq!(
            response.extra_fields["keyValuePairs"],
            analyze_result["keyValuePairs"]
        );
        assert_eq!(response.object, "ocr");
        assert_eq!(
            response.usage_info,
            Some(json!({"pages_processed": 1, "doc_size_bytes": null}))
        );
        assert_eq!(response.pages[0]["index"], 0);
        assert_eq!(
            response.pages[0]["markdown"],
            "Invoice\nInvoice No: INV-12345\nTotal: $100.00"
        );
        assert_eq!(
            response.pages[0]["dimensions"],
            json!({"width": 816, "height": 1056, "dpi": 96})
        );
    }

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
    fn document_intelligence_url_normalizes_features() {
        let params = serde_json::Map::from_iter([(
            "features".to_string(),
            json!("keyValuePairs, languages"),
        )]);
        let url = complete_document_intelligence_url(
            Some("https://example.cognitiveservices.azure.com"),
            "prebuilt-layout",
            &params,
            &|_| None,
        )
        .expect("url builds");

        assert_eq!(
            url,
            "https://example.cognitiveservices.azure.com/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30&features=keyValuePairs,languages"
        );
    }

    #[test]
    fn document_intelligence_url_combines_pages_and_feature_list() {
        let params = serde_json::Map::from_iter([
            ("pages".to_string(), json!([0, 1, 2])),
            (
                "features".to_string(),
                json!([" keyValuePairs ", "languages"]),
            ),
        ]);
        let url = complete_document_intelligence_url(
            Some("https://example.cognitiveservices.azure.com"),
            "prebuilt-layout",
            &params,
            &|_| None,
        )
        .expect("url builds");

        assert_eq!(
            url,
            "https://example.cognitiveservices.azure.com/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30&pages=1,2,3&features=keyValuePairs,languages"
        );
    }

    #[test]
    fn document_intelligence_url_omits_empty_feature_list() {
        let params = serde_json::Map::from_iter([("features".to_string(), json!([]))]);
        assert!(
            map_document_intelligence_ocr_params(&params)
                .expect("empty features map")
                .is_empty()
        );
        let url = complete_document_intelligence_url(
            Some("https://example.cognitiveservices.azure.com"),
            "prebuilt-layout",
            &params,
            &|_| None,
        )
        .expect("url builds");

        assert_eq!(
            url,
            "https://example.cognitiveservices.azure.com/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30"
        );
    }

    #[rstest]
    #[case::query_injection(json!("keyValuePairs&pages=9"))]
    #[case::spaces(json!("key value pairs"))]
    #[case::empty_string(json!(""))]
    #[case::integer_list(json!([1, 2]))]
    #[case::nested_list(json!([["keyValuePairs"]]))]
    #[case::object(json!({"feature": "keyValuePairs"}))]
    #[case::number(json!(5))]
    fn document_intelligence_mapping_rejects_invalid_features(#[case] features: Value) {
        let params = serde_json::Map::from_iter([("features".to_string(), features)]);
        let error =
            map_document_intelligence_ocr_params(&params).expect_err("invalid features must fail");

        assert!(matches!(
            error,
            Error::InvalidRequest(message) if message.contains("Invalid `features`")
        ));
    }

    #[rstest]
    #[case::single_list(json!(["keyValuePairs"]), "keyValuePairs")]
    #[case::multiple_list(
        json!(["keyValuePairs", "languages"]),
        "keyValuePairs,languages"
    )]
    #[case::single_string(json!("keyValuePairs"), "keyValuePairs")]
    #[case::comma_separated(json!("keyValuePairs,languages"), "keyValuePairs,languages")]
    #[case::spaces(json!("keyValuePairs, languages"), "keyValuePairs,languages")]
    fn document_intelligence_maps_features(#[case] features: Value, #[case] expected: &str) {
        let params = Map::from_iter([
            ("features".to_string(), features),
            ("unsupported".to_string(), json!(true)),
        ]);

        assert_eq!(
            AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG.map_ocr_params(&params),
            Map::from_iter([("features".to_string(), json!(expected))])
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

    #[rstest]
    fn document_intelligence_response_normalizes_pages(native_operation: Value) {
        let response = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response("prebuilt-layout", native_operation.clone())
            .expect("response transforms");

        assert_native_fields_preserved(&response, &native_operation);
    }

    #[test]
    fn azure_document_intelligence_model_id_is_encoded() {
        let url = complete_document_intelligence_url(
            Some(ENDPOINT),
            "prebuilt-layout?x=1#frag",
            &Map::new(),
            &|_| None,
        )
        .expect("url builds");

        assert_eq!(
            url,
            "https://example.cognitiveservices.azure.com/documentintelligence/documentModels/prebuilt-layout%3Fx%3D1%23frag:analyze?api-version=2024-11-30"
        );
    }

    #[test]
    fn azure_document_intelligence_dot_segment_model_id_is_rejected() {
        let error = complete_document_intelligence_url(
            Some(ENDPOINT),
            "azure_ai/doc-intelligence/..",
            &Map::new(),
            &|_| None,
        )
        .expect_err("dot segment must fail");

        assert_eq!(
            error,
            Error::InvalidRequest("model_id cannot be a dot path segment".to_string())
        );
    }

    #[rstest]
    fn document_intelligence_async_response_preserves_normalized_fields(native_operation: Value) {
        let response = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response(
                "azure_ai/doc-intelligence/prebuilt-layout",
                native_operation.clone(),
            )
            .expect("response transforms");

        assert_native_fields_preserved(&response, &native_operation);
    }

    #[test]
    fn document_intelligence_response_tolerates_missing_native_fields() {
        let response = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response(
                "azure_ai/doc-intelligence/prebuilt-read",
                json!({
                    "status": "succeeded",
                    "analyzeResult": {
                        "pages": [{
                            "pageNumber": 1,
                            "width": 8.5,
                            "height": 11,
                            "unit": "inch",
                            "lines": [{"content": "hello"}]
                        }]
                    }
                }),
            )
            .expect("missing optional fields are allowed");

        assert_eq!(response.pages[0]["markdown"], "hello");
        assert_eq!(response.extra_fields["content"], Value::Null);
        assert_eq!(response.extra_fields["tables"], Value::Null);
        assert_eq!(response.extra_fields["keyValuePairs"], Value::Null);
    }

    #[test]
    fn document_intelligence_non_succeeded_status_is_rejected() {
        let error = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response(
                "azure_ai/doc-intelligence/prebuilt-layout",
                json!({"status": "failed"}),
            )
            .expect_err("failed status must fail");

        assert_eq!(
            error,
            Error::InvalidResponse(
                "Azure Document Intelligence analysis failed with status: failed".to_string()
            )
        );
    }

    #[test]
    fn document_intelligence_supported_params_include_features() {
        assert_eq!(
            AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG.supported_ocr_params(),
            &["pages", "features", "req_format"]
        );
    }

    #[rstest]
    fn document_intelligence_native_format_carries_raw_operation(native_operation: Value) {
        let response = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response_with_params(
                "azure_ai/doc-intelligence/prebuilt-layout",
                native_operation.clone(),
                &Map::from_iter([("req_format".to_string(), json!("native"))]),
            )
            .expect("native response transforms");

        assert_eq!(
            response.provider_native_response,
            Some(native_operation.clone())
        );
        assert_native_fields_preserved(&response, &native_operation);
    }

    #[rstest]
    fn document_intelligence_async_native_format_carries_raw_operation(native_operation: Value) {
        let response = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response_with_params(
                "azure_ai/doc-intelligence/prebuilt-layout",
                native_operation.clone(),
                &Map::from_iter([("req_format".to_string(), json!("native"))]),
            )
            .expect("native response transforms");

        assert_eq!(
            response.provider_native_response,
            Some(native_operation.clone())
        );
        assert_native_fields_preserved(&response, &native_operation);
    }

    #[rstest]
    #[case::default(Map::new())]
    #[case::litellm(Map::from_iter([("req_format".to_string(), json!("litellm"))]))]
    fn document_intelligence_default_format_omits_raw_operation(
        #[case] optional_params: Map<String, Value>,
        native_operation: Value,
    ) {
        let response = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_response_with_params(
                "azure_ai/doc-intelligence/prebuilt-layout",
                native_operation.clone(),
                &optional_params,
            )
            .expect("response transforms");

        assert_eq!(response.provider_native_response, None);
        assert_native_fields_preserved(&response, &native_operation);
    }

    #[rstest]
    #[case::native("native")]
    #[case::litellm("litellm")]
    fn document_intelligence_maps_req_format(#[case] req_format: &str) {
        let mapped = map_document_intelligence_ocr_params(&Map::from_iter([(
            "req_format".to_string(),
            json!(req_format),
        )]))
        .expect("req_format maps");

        assert_eq!(
            mapped,
            Map::from_iter([("req_format".to_string(), json!(req_format))])
        );
    }

    #[test]
    fn document_intelligence_rejects_unknown_req_format() {
        let error = map_document_intelligence_ocr_params(&Map::from_iter([(
            "req_format".to_string(),
            json!("azure"),
        )]))
        .expect_err("unknown req_format must fail");

        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("Invalid `req_format`"))
        );
    }

    #[test]
    fn document_intelligence_url_omits_req_format() {
        let url = complete_document_intelligence_url(
            Some(ENDPOINT),
            "prebuilt-layout",
            &Map::from_iter([("req_format".to_string(), json!("native"))]),
            &|_| None,
        )
        .expect("url builds");

        assert!(!url.contains("req_format"));
    }

    #[test]
    fn document_intelligence_validate_environment_uses_subscription_key() {
        let headers =
            validate_document_intelligence_environment(Vec::new(), Some("my-key"), None, &|_| None)
                .expect("api key authenticates");

        assert_eq!(
            header_value(&headers, "Ocp-Apim-Subscription-Key"),
            Some("my-key")
        );
    }

    #[test]
    fn document_intelligence_validate_environment_falls_back_to_entra_token() {
        let headers = validate_document_intelligence_environment(
            Vec::new(),
            None,
            Some("entra-token"),
            &|_| None,
        )
        .expect("Entra token authenticates");

        assert_eq!(
            header_value(&headers, "Authorization"),
            Some("Bearer entra-token")
        );
        assert_eq!(header_value(&headers, "Ocp-Apim-Subscription-Key"), None);
    }

    #[test]
    fn document_intelligence_supported_params_include_pages_features_and_req_format() {
        assert_eq!(
            AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG.supported_ocr_params(),
            &["pages", "features", "req_format"]
        );
    }

    #[test]
    fn document_intelligence_maps_zero_based_page_list() {
        let mapped = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!([0, 1, 2]),
        )]))
        .expect("pages map");

        assert_eq!(
            mapped,
            Map::from_iter([("pages".to_string(), json!("1,2,3"))])
        );
    }

    #[test]
    fn document_intelligence_page_mapping_dedupes_and_sorts() {
        let mapped = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!([2, 0, 0, 1]),
        )]))
        .expect("pages map");

        assert_eq!(mapped["pages"], "1,2,3");
    }

    #[test]
    fn document_intelligence_page_mapping_omits_empty_list() {
        let mapped = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!([]),
        )]))
        .expect("empty pages map");

        assert!(mapped.is_empty());
    }

    #[test]
    fn document_intelligence_page_mapping_accepts_native_range() {
        let mapped = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!("3-9"),
        )]))
        .expect("range maps");

        assert_eq!(mapped["pages"], "3-9");
    }

    #[test]
    fn document_intelligence_page_mapping_strips_spaces() {
        let mapped = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!("1-3, 5"),
        )]))
        .expect("range maps");

        assert_eq!(mapped["pages"], "1-3,5");
    }

    #[test]
    fn document_intelligence_page_mapping_accepts_string_tokens() {
        let mapped = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!(["1", "3-5"]),
        )]))
        .expect("tokens map");

        assert_eq!(mapped["pages"], "1,3-5");
    }

    #[test]
    fn document_intelligence_page_mapping_rejects_invalid_string() {
        let error = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!("a,b"),
        )]))
        .expect_err("invalid pages must fail");

        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("Invalid `pages` string"))
        );
    }

    #[test]
    fn document_intelligence_page_mapping_rejects_negative_index() {
        let error = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!([-1]),
        )]))
        .expect_err("negative pages must fail");

        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("must be >= 0"))
        );
    }

    #[test]
    fn document_intelligence_page_mapping_rejects_bool_list() {
        let error = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!([true, false]),
        )]))
        .expect_err("boolean pages must fail");

        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("integers, not booleans"))
        );
    }

    #[test]
    fn document_intelligence_page_mapping_rejects_unsupported_type() {
        let error = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!(5),
        )]))
        .expect_err("unsupported pages must fail");

        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("Mistral-style"))
        );
    }

    #[test]
    fn document_intelligence_url_appends_pages_query() {
        let url = complete_document_intelligence_url(
            Some("https://example.cognitiveservices.azure.com/"),
            "azure_ai/doc-intelligence/prebuilt-layout",
            &Map::from_iter([("pages".to_string(), json!("1-3,5"))]),
            &|_| None,
        )
        .expect("url builds");

        assert!(url.contains("api-version=2024-11-30"));
        assert!(url.contains("pages=1-3,5"));
        assert!(url.contains("/documentintelligence/documentModels/prebuilt-layout:analyze"));
    }

    #[test]
    fn document_intelligence_url_has_no_pages_when_params_are_empty() {
        let url = complete_document_intelligence_url(
            Some(ENDPOINT),
            "prebuilt-layout",
            &Map::new(),
            &|_| None,
        )
        .expect("url builds");

        assert!(!url.contains("pages="));
    }

    #[rstest]
    fn document_intelligence_request_keeps_pages_out_of_body(
        document_intelligence_config: AzureDocumentIntelligenceOcrConfig,
    ) {
        let request = document_intelligence_config
            .transform_ocr_request(
                "prebuilt-layout",
                json!({"type": "document_url", "document_url": "https://example.com/x.pdf"}),
                Map::from_iter([("pages".to_string(), json!("1,2,3"))]),
            )
            .expect("request transforms");

        assert_eq!(
            request.data,
            json!({"urlSource": "https://example.com/x.pdf"})
        );
    }

    #[test]
    fn document_intelligence_mistral_pages_flow_to_query_only() {
        let mapped = map_document_intelligence_ocr_params(&Map::from_iter([(
            "pages".to_string(),
            json!([2, 3, 4, 5, 6, 7, 8]),
        )]))
        .expect("pages map");
        let url =
            complete_document_intelligence_url(Some(ENDPOINT), "prebuilt-layout", &mapped, &|_| {
                None
            })
            .expect("url builds");
        let request = AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
            .transform_ocr_request(
                "prebuilt-layout",
                json!({"type": "document_url", "document_url": "https://example.com/x.pdf"}),
                mapped,
            )
            .expect("request transforms");

        assert!(url.contains("pages=3,4,5,6,7,8,9"));
        assert_eq!(
            request.data,
            json!({"urlSource": "https://example.com/x.pdf"})
        );
    }

    #[test]
    fn document_intelligence_endpoint_ignores_generic_azure_ai_base() {
        let resolved = resolve_document_intelligence_endpoint(None, &|name| match name {
            AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV => Some(ENDPOINT.to_string()),
            AZURE_AI_API_BASE_ENV => Some("https://generic.example.com".to_string()),
            _ => None,
        })
        .expect("endpoint resolves");

        assert_eq!(resolved, ENDPOINT);
    }

    #[test]
    fn document_intelligence_endpoint_honors_explicit_api_base() {
        let resolved = resolve_document_intelligence_endpoint(
            Some("https://my-di.cognitiveservices.azure.com"),
            &|name| match name {
                AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV => Some(ENDPOINT.to_string()),
                AZURE_AI_API_BASE_ENV => Some("https://generic.example.com".to_string()),
                _ => None,
            },
        )
        .expect("endpoint resolves");

        assert_eq!(resolved, "https://my-di.cognitiveservices.azure.com");
    }

    #[test]
    fn azure_ai_mistral_ocr_uses_generic_api_base() {
        let resolved = resolve_azure_ai_api_base(None, &|name| match name {
            AZURE_AI_API_BASE_ENV => Some("https://generic-azure-ai.example.com".to_string()),
            AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV => Some(ENDPOINT.to_string()),
            _ => None,
        })
        .expect("api base resolves");

        assert_eq!(resolved, "https://generic-azure-ai.example.com");
    }

    #[test]
    fn azure_ai_ocr_authenticates_with_entra_token() {
        let headers =
            validate_azure_ai_environment(Vec::new(), None, Some("entra-token"), &|_| None)
                .expect("Entra token authenticates");

        assert_eq!(
            header_value(&headers, "Authorization"),
            Some("Bearer entra-token")
        );
    }
}
