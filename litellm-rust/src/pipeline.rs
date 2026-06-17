//! Pure orchestration of an OCR call using blocking HTTP. No PyO3.
//!
//! Builds url + headers + body via the pure transforms, POSTs, and parses the
//! response. The async path used by the HTTP server reuses the same transforms.

use serde_json::Value;

use crate::error::OcrError;
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, Headers, OcrRequest, OcrResponse};
use crate::llms::mistral::ocr::transformation::MistralOcrConfig;

/// A prepared OCR call: (url, headers, request body).
pub type PreparedCall = (String, Headers, Value);

/// Build (url, headers, body) for an OCR request using the pure transforms.
///
/// Shared by the blocking (Python) and async (server) code paths so both go
/// through exactly the same transform logic.
pub fn build_call(config: &dyn BaseOcrConfig, req: &OcrRequest) -> Result<PreparedCall, OcrError> {
    let headers = config
        .validate_environment(&req.model, req.api_key.as_deref(), &|key| {
            std::env::var(key).ok()
        })
        .map_err(OcrError::Auth)?;

    let url = config.get_complete_url(req.api_base.as_deref(), &req.model);

    let filtered = config.map_ocr_params(&req.optional_params, &req.model);
    let body = config
        .transform_ocr_request(&req.model, &req.document, &filtered)
        .map_err(OcrError::Transform)?;

    Ok((url, headers, body))
}

/// Parse an upstream response body into the standard `OcrResponse`.
pub fn parse_response(
    config: &dyn BaseOcrConfig,
    model: &str,
    response_json: &Value,
) -> Result<OcrResponse, OcrError> {
    config
        .transform_ocr_response(model, response_json)
        .map_err(OcrError::Parse)
}

/// Perform a blocking Mistral OCR call end to end.
///
/// Used by the Python extension (called synchronously from Python).
pub fn ocr_blocking(req: OcrRequest) -> Result<OcrResponse, OcrError> {
    let config = MistralOcrConfig::new();
    let (url, headers, body) = build_call(&config, &req)?;

    let client = reqwest::blocking::Client::new();
    let mut builder = client.post(&url).json(&body);
    for (k, v) in &headers {
        builder = builder.header(k, v);
    }

    let resp = builder
        .send()
        .map_err(|e| OcrError::Network(e.to_string()))?;

    let status = resp.status();
    let text = resp.text().map_err(|e| OcrError::Network(e.to_string()))?;

    if !status.is_success() {
        return Err(OcrError::Http {
            status: status.as_u16(),
            body: text,
        });
    }

    let response_json: Value =
        serde_json::from_str(&text).map_err(|e| OcrError::Parse(e.to_string()))?;

    parse_response(&config, &req.model, &response_json)
}
