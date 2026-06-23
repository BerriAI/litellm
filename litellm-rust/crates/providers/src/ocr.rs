//! End-to-end OCR orchestration.
//!
//! Owns the whole Mistral OCR call so the Python side stays a thin bridge:
//! resolve the API key, build the URL + body via the pure transforms, POST it,
//! and normalize the response. The HTTP client is built once and reused.

use std::sync::OnceLock;
use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::OcrProviderConfig;
use litellm_core::CoreResult;
use serde_json::{Map, Value};

use crate::mistral::ocr::transformation as mistral;
use crate::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;

/// OCR over large documents can take a while; bound it generously rather than
/// hanging forever on an unresponsive upstream.
const OCR_TIMEOUT_SECS: u64 = 600;

/// Process-wide blocking HTTP client (connection pool + TLS reused across calls).
fn http_client() -> &'static reqwest::blocking::Client {
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(OCR_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client")
    })
}

/// Perform a Mistral OCR call end to end and return the normalized response as
/// JSON (the shape the Python `OCRResponse` model expects).
///
/// Blocking: intended to be called with the GIL released from the Python bridge.
pub fn run_ocr(
    model: &str,
    document: Value,
    api_key: Option<&str>,
    api_base: Option<&str>,
    optional_params: Map<String, Value>,
) -> CoreResult<Value> {
    let config = &MISTRAL_OCR_CONFIG;

    let api_key = mistral::resolve_api_key(api_key, &|key| std::env::var(key).ok())?;
    let url = mistral::complete_url(api_base);
    let filtered_params = config.map_ocr_params(&optional_params);
    let body = config
        .transform_ocr_request(model, document, filtered_params)?
        .data;

    let response = http_client()
        .post(&url)
        .bearer_auth(&api_key)
        .json(&body)
        .send()
        .map_err(|err| CoreError::Network(err.to_string()))?;

    let status = response.status();
    let text = response
        .text()
        .map_err(|err| CoreError::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: text,
        });
    }

    let response_json: Value = serde_json::from_str(&text)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(config
        .transform_ocr_response(model, response_json)?
        .into_json())
}
