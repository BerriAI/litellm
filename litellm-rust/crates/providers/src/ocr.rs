//! End-to-end OCR orchestration.
//!
//! Owns the whole Mistral OCR call so the Python side stays a thin bridge:
//! resolve the API key, build the URL + body via the pure transforms, POST it,
//! and normalize the response. The HTTP client is built once and reused.

use std::str::FromStr;
use std::sync::OnceLock;
use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::OcrProviderConfig;
use litellm_core::CoreResult;
use litellm_core::LlmProvider;
use serde_json::{Map, Value};

use crate::mistral::ocr::transformation as mistral;
use crate::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;

/// OCR over large documents can take a while; bound it generously rather than
/// hanging forever on an unresponsive upstream. The client-level limit is the
/// outer ceiling; callers can tighten it per request via ``run_ocr``'s ``timeout``.
const OCR_TIMEOUT_SECS: u64 = 600;

/// Maximum upstream body characters retained in error messages. OCR responses
/// can echo document contents and prompts; keep enough for debugging without
/// forwarding sensitive payloads across the host boundary.
const ERROR_BODY_MAX_CHARS: usize = 256;

/// Process-wide async HTTP client (connection pool + TLS reused across calls).
fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(OCR_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client")
    })
}

fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

fn ocr_config_for(provider: LlmProvider) -> Option<&'static dyn OcrProviderConfig> {
    match provider {
        LlmProvider::Mistral => Some(&MISTRAL_OCR_CONFIG),
        _ => None,
    }
}

fn string_headers(extra_headers: Option<Map<String, Value>>) -> CoreResult<Vec<(String, String)>> {
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

pub struct OcrRequest<'a> {
    pub model: &'a str,
    pub document: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: &'a str,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
}

/// Perform a Mistral OCR call end to end and return the normalized response as
/// JSON (the shape the Python `OCRResponse` model expects).
///
/// Async: intended to be awaited directly by the Python bridge's async entrypoint.
pub async fn ocr(request: OcrRequest<'_>) -> CoreResult<Value> {
    let model = request.model;
    let provider = LlmProvider::from_str(request.custom_llm_provider)?;
    let config =
        ocr_config_for(provider).ok_or_else(|| CoreError::InvalidProvider(provider.to_string()))?;

    // TODO: key and URL resolution are still Mistral-specific while Mistral is
    // the only Rust OCR provider. Move these onto the trait when another OCR
    // provider is added here.
    let api_key = mistral::resolve_api_key(request.api_key, &|key| std::env::var(key).ok())?;
    let url = mistral::complete_url(request.api_base);
    let filtered_params = config.map_ocr_params(&request.optional_params);
    let body = config
        .transform_ocr_request(model, request.document, filtered_params)?
        .data;

    let mut request_builder = http_client().post(&url).bearer_auth(&api_key).json(&body);
    for (key, value) in string_headers(request.extra_headers)? {
        request_builder = request_builder.header(&key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

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

    let response_json: Value = serde_json::from_str(&text)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(config
        .transform_ocr_response(model, response_json)?
        .into_json())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn truncate_error_body_passes_short_strings_through() {
        let body = "Unauthorized";
        assert_eq!(truncate_error_body(body), "Unauthorized");
    }

    #[test]
    fn truncate_error_body_caps_long_payloads() {
        let body = "x".repeat(ERROR_BODY_MAX_CHARS + 50);
        let truncated = truncate_error_body(&body);

        assert!(truncated.ends_with("... (truncated)"));
        let prefix_chars = truncated
            .strip_suffix("... (truncated)")
            .expect("truncated marker present")
            .chars()
            .count();
        assert_eq!(prefix_chars, ERROR_BODY_MAX_CHARS);
    }

    #[test]
    fn truncate_error_body_does_not_split_multibyte_chars() {
        let body = "é".repeat(ERROR_BODY_MAX_CHARS + 10);
        let truncated = truncate_error_body(&body);
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn ocr_registry_supports_only_mistral() {
        assert!(ocr_config_for(LlmProvider::Mistral).is_some());
        assert!(ocr_config_for(LlmProvider::Openai).is_none());
    }

    #[test]
    fn string_headers_accepts_string_values() {
        let headers = json!({
            "x-trace-id": "trace-1"
        })
        .as_object()
        .unwrap()
        .clone();

        assert_eq!(
            string_headers(Some(headers)).expect("string headers accepted"),
            vec![("x-trace-id".to_string(), "trace-1".to_string())]
        );
    }

    #[test]
    fn string_headers_rejects_non_string_values() {
        let headers = json!({
            "x-retry-count": 3
        })
        .as_object()
        .unwrap()
        .clone();

        let err = string_headers(Some(headers)).expect_err("non-string header rejected");
        assert_eq!(
            err,
            CoreError::InvalidRequest(
                "OCR extra_headers.x-retry-count must be a string, got number".to_string()
            )
        );
    }
}
