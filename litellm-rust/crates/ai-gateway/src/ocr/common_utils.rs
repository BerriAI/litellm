use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use litellm_core::CoreResult;
use litellm_core::error::CoreError;
use litellm_core::http_utils::safe_fetch::{SafeFetchOptions, safe_fetch};
use litellm_core::ocr::transformation::OcrProviderConfig;
use serde_json::{Map, Value};

use litellm_core::providers::azure_ai::ocr::transformation::{
    AZURE_AI_OCR_CONFIG, AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
};
use litellm_core::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use litellm_core::providers::vertex_ai::ocr::transformation as vertex_ai;
use litellm_core::providers::vertex_ai::ocr::transformation::{
    VERTEX_AI_DEEPSEEK_OCR_CONFIG, VERTEX_AI_OCR_CONFIG,
};

use crate::client::http_client;

const ERROR_BODY_MAX_CHARS: usize = 256;
const AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB: f64 = 50.0;

pub(super) fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
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
        "azure_ai" => Some(&AZURE_AI_OCR_CONFIG),
        "vertex_ai" if vertex_ai::is_deepseek_model(model) => Some(&VERTEX_AI_DEEPSEEK_OCR_CONFIG),
        "vertex_ai" => Some(&VERTEX_AI_OCR_CONFIG),
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

fn document_url_field(document: &Value) -> CoreResult<Option<(&str, &str)>> {
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

fn is_url_requiring_fetch(url: &str) -> bool {
    !url.starts_with("data:") && (url.starts_with("http://") || url.starts_with("https://"))
}

fn max_document_download_bytes() -> u64 {
    let max_size_mb = std::env::var("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(DEFAULT_MAX_IMAGE_URL_DOWNLOAD_SIZE_MB);
    (max_size_mb.max(0.0) * 1024.0 * 1024.0) as u64
}

pub(super) async fn convert_document_url_to_data_uri(document: Value) -> CoreResult<Value> {
    let Some((field, url)) = document_url_field(&document)? else {
        return Ok(document);
    };
    if !is_url_requiring_fetch(url) {
        return Ok(document);
    }

    let response = safe_fetch(url, SafeFetchOptions::new(max_document_download_bytes())).await?;
    let status = response.status;
    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&String::from_utf8_lossy(&response.body)),
        });
    }
    let content_type = response
        .headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    let data_uri = format!(
        "data:{content_type};base64,{}",
        BASE64_STANDARD.encode(response.body)
    );

    let mut transformed = document
        .as_object()
        .cloned()
        .ok_or_else(|| CoreError::InvalidRequest("OCR document must be an object".to_string()))?;
    transformed.insert(field.to_string(), Value::String(data_uri));
    Ok(Value::Object(transformed))
}

fn same_origin(left: &str, right: &str) -> bool {
    let Ok(left) = reqwest::Url::parse(left) else {
        return false;
    };
    let Ok(right) = reqwest::Url::parse(right) else {
        return false;
    };
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn retry_after_secs(response: &reqwest::Response) -> u64 {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2)
}

fn operation_status(response_json: &Value) -> CoreResult<&str> {
    let status = response_json
        .get("status")
        .and_then(Value::as_str)
        .ok_or(CoreError::MissingField("status"))?;
    match status {
        "succeeded" => Ok("succeeded"),
        "running" | "notStarted" => Ok("running"),
        "failed" => {
            let message = response_json
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown error");
            Err(CoreError::InvalidResponse(format!(
                "Azure Document Intelligence analysis failed: {message}"
            )))
        }
        other => Err(CoreError::InvalidResponse(format!(
            "Unknown operation status: {other}"
        ))),
    }
}

pub(super) async fn poll_document_intelligence(
    operation_url: &str,
    original_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> CoreResult<Value> {
    if !same_origin(operation_url, original_url) {
        return Err(CoreError::InvalidResponse(
            "Azure Document Intelligence: rejected cross-origin polling URL".to_string(),
        ));
    }

    let start = Instant::now();
    let timeout = timeout.unwrap_or(Duration::from_secs(
        AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS,
    ));
    loop {
        if start.elapsed() > timeout {
            return Err(CoreError::Network(format!(
                "Azure Document Intelligence operation polling timed out after {} seconds",
                timeout.as_secs()
            )));
        }

        let mut request_builder = http_client().get(operation_url);
        for (key, value) in headers {
            if key.eq_ignore_ascii_case("ocp-apim-subscription-key") {
                request_builder = request_builder.header(key, value);
            }
        }
        let response = request_builder
            .send()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        let retry_after = retry_after_secs(&response);
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        if !status.is_success() {
            return Err(CoreError::Http {
                status: status.as_u16(),
                body: truncate_error_body(&text),
            });
        }
        let response_json: Value = serde_json::from_str(&text).map_err(|err| {
            CoreError::InvalidResponse(format!("invalid Azure DI poll response JSON: {err}"))
        })?;
        if operation_status(&response_json)? == "succeeded" {
            return Ok(response_json);
        }
        tokio::time::sleep(Duration::from_secs(retry_after)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn convert_document_url_rejects_loopback_fetch() {
        let error = convert_document_url_to_data_uri(json!({
            "type": "image_url",
            "image_url": "http://127.0.0.1/image.png"
        }))
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            CoreError::InvalidRequest(message)
                if message.contains("not public")
        ));
    }

    #[tokio::test]
    async fn convert_document_url_leaves_data_uri_untouched() {
        let document = json!({
            "type": "image_url",
            "image_url": "data:image/png;base64,abcd"
        });

        let transformed = convert_document_url_to_data_uri(document.clone())
            .await
            .unwrap();

        assert_eq!(transformed, document);
    }
}
