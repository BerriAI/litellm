use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::OcrProviderConfig;
use litellm_core::CoreResult;
use serde_json::{Map, Value};

use litellm_core::providers::azure_ai::ocr::transformation::{
    AZURE_AI_OCR_CONFIG, AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
};
use litellm_core::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use litellm_core::providers::reducto::ocr::transformation::{
    REDUCTO_PARSE_LEGACY_CONFIG, REDUCTO_PARSE_V3_CONFIG,
};
use litellm_core::providers::vertex_ai::ocr::transformation as vertex_ai;
use litellm_core::providers::vertex_ai::ocr::transformation::{
    VERTEX_AI_DEEPSEEK_OCR_CONFIG, VERTEX_AI_OCR_CONFIG,
};

use crate::constants::{OCR_ERROR_BODY_MAX_BYTES, OCR_ERROR_BODY_MAX_CHARS};

pub(super) fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= OCR_ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(OCR_ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

pub(super) async fn read_error_body(mut response: reqwest::Response) -> String {
    let mut collected: Vec<u8> = Vec::new();
    while collected.len() < OCR_ERROR_BODY_MAX_BYTES {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                let take = (OCR_ERROR_BODY_MAX_BYTES - collected.len()).min(chunk.len());
                collected.extend_from_slice(&chunk[..take]);
                if take < chunk.len() {
                    break;
                }
            }
            Ok(None) | Err(_) => break,
        }
    }
    truncate_error_body(&String::from_utf8_lossy(&collected))
}

pub(super) fn ocr_provider_config(
    provider: &str,
    model: &str,
) -> Option<&'static dyn OcrProviderConfig> {
    match provider {
        "mistral" => Some(&MISTRAL_OCR_CONFIG),
        "azure_ai" if is_azure_document_intelligence_model(model) => {
            Some(&AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG)
        }
        "azure_ai/doc-intelligence" => Some(&AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG),
        "azure_ai" => Some(&AZURE_AI_OCR_CONFIG),
        "vertex_ai" if vertex_ai::is_deepseek_model(model) => Some(&VERTEX_AI_DEEPSEEK_OCR_CONFIG),
        "vertex_ai" => Some(&VERTEX_AI_OCR_CONFIG),
        "reducto" if model == "parse-v3" => Some(&REDUCTO_PARSE_V3_CONFIG),
        "reducto" if model == "parse-legacy" => Some(&REDUCTO_PARSE_LEGACY_CONFIG),
        _ => None,
    }
}

fn is_azure_document_intelligence_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doc-intelligence") || model.contains("documentintelligence")
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> CoreResult<Vec<(String, String)>> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    CoreError::InvalidRequest(format!(
                        "OCR extra_headers.{key} must be a string, got {}",
                        litellm_core::error::json_type_name(&value)
                    ))
                })
        })
        .collect()
}

pub(super) fn has_header(headers: &[(String, String)], name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}

pub(super) fn document_url_field(document: &Value) -> CoreResult<Option<(&str, &str)>> {
    let Some(object) = document.as_object() else {
        return Ok(None);
    };
    let Some(doc_type) = object.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };
    let field = match doc_type {
        "document_url" => "document_url",
        "image_url" => "image_url",
        _ => return Ok(None),
    };
    let Some(url) = object.get(field).and_then(Value::as_str) else {
        return Ok(None);
    };
    Ok(Some((field, url)))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::spawn_counting_server;
    use super::*;

    #[tokio::test]
    async fn oversized_error_body_without_content_length_is_bounded() {
        let mut raw = b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n".to_vec();
        raw.extend_from_slice(&vec![b'a'; 100_000]);
        let server = spawn_counting_server(raw).await;
        let url_str = format!("http://127.0.0.1:{}/err", server.addr.port());
        let response = reqwest::Client::new().get(&url_str).send().await.unwrap();
        assert!(response.content_length().is_none());
        assert!(!response.status().is_success());

        let body = read_error_body(response).await;

        let prefix = body
            .strip_suffix("... (truncated)")
            .expect("oversized error body must be truncated to the cap");
        assert_eq!(prefix.chars().count(), OCR_ERROR_BODY_MAX_CHARS);
        assert!(prefix.chars().all(|c| c == 'a'));
    }
}
