//! End-to-end OCR orchestration.
//!
//! Owns supported OCR provider calls so the Python side stays a thin bridge:
//! resolve the API key, build the URL + body via the pure transforms, POST it,
//! and normalize the response. The HTTP client is built once and reused.

use std::sync::OnceLock;
use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::{
    OcrAuthStrategy, OcrDocumentPreparation, OcrResponseHandling,
};
use litellm_core::CoreResult;
use serde_json::{Map, Value};

use crate::config::resolve_env_reference;

mod common_utils;

use common_utils::{
    classify_reqwest_error, convert_document_url_to_data_uri, has_header, ocr_provider_config,
    poll_document_intelligence, string_headers, truncate_error_body, upload_reducto_document,
};

/// OCR over large documents can take a while; bound it generously rather than
/// hanging forever on an unresponsive upstream. The client-level limit is the
/// outer ceiling; callers can tighten it per request via ``run_ocr``'s ``timeout``.
const OCR_TIMEOUT_SECS: u64 = 600;

/// Process-wide async HTTP client (connection pool + TLS reused across calls).
///
/// The Python fallback path uses LiteLLM's standard `BaseLLMHTTPHandler`. This
/// Rust path is opt-in and owns end-to-end OCR I/O, so it cannot call the
/// Python handler directly; keep this route-scoped until litellm-rust has a
/// shared HTTP abstraction.
fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(OCR_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client")
    })
}

fn upstream_headers(
    headers: &[(String, String)],
    auth_strategy: OcrAuthStrategy,
    api_key: Option<&str>,
) -> Vec<(String, String)> {
    let auth_header = api_key.map(|api_key| match auth_strategy {
        OcrAuthStrategy::Bearer => ("Authorization".to_string(), format!("Bearer {api_key}")),
        OcrAuthStrategy::Header(header_name) => (header_name.to_string(), api_key.to_string()),
    });
    auth_header
        .into_iter()
        .chain(headers.iter().cloned())
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

/// Perform an OCR call end to end and return the normalized response as
/// JSON (the shape the Python `OCRResponse` model expects).
///
/// Async: intended to be awaited directly by the Python bridge's async entrypoint.
pub async fn ocr(request: OcrRequest<'_>) -> CoreResult<Value> {
    let env_lookup = |key: &str| std::env::var(key).ok();
    ocr_with_env(request, &env_lookup).await
}

async fn ocr_with_env(
    request: OcrRequest<'_>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> CoreResult<Value> {
    let model = request.model;
    let config = ocr_provider_config(request.custom_llm_provider, model).ok_or_else(|| {
        CoreError::InvalidProvider(format!(
            "no OCR provider '{}' registered for model '{model}'",
            request.custom_llm_provider
        ))
    })?;
    let api_key = resolve_env_reference(request.api_key, env_lookup);
    let api_base = resolve_env_reference(request.api_base, env_lookup);

    let headers = string_headers(request.extra_headers)?;
    let auth_strategy = config.auth_strategy();
    let api_key = (!has_header(&headers, auth_strategy.header_name()))
        .then(|| config.resolve_api_key(api_key.as_deref(), env_lookup))
        .transpose()?;
    let url = config.complete_url(
        api_base.as_deref(),
        model,
        &request.optional_params,
        env_lookup,
    )?;
    let filtered_params = config.map_ocr_params(&request.optional_params);
    let upstream_headers = upstream_headers(&headers, auth_strategy, api_key.as_deref());
    let document = match config.document_preparation() {
        OcrDocumentPreparation::None => request.document,
        OcrDocumentPreparation::DataUri => {
            convert_document_url_to_data_uri(request.document).await?
        }
        OcrDocumentPreparation::ReductoUpload => {
            upload_reducto_document(request.document, &url, &upstream_headers, request.timeout)
                .await?
        }
    };
    let body = config
        .transform_ocr_request(model, document, filtered_params)?
        .data;

    let mut request_builder = http_client().post(&url).json(&body);
    for (key, value) in &upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(classify_reqwest_error)?;

    let status = response.status();
    if config.response_handling() == OcrResponseHandling::AzureDocumentIntelligencePoll
        && status.as_u16() == 202
    {
        let operation_url = response
            .headers()
            .get("operation-location")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .ok_or_else(|| {
                CoreError::InvalidResponse(
                    "Azure Document Intelligence returned 202 but no Operation-Location header found"
                        .to_string(),
                )
            })?;
        let response_json =
            poll_document_intelligence(&operation_url, &url, &upstream_headers, request.timeout)
                .await?;
        return Ok(config
            .transform_ocr_response(model, response_json)?
            .into_json());
    }

    let text = response.text().await.map_err(classify_reqwest_error)?;

    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    if text.trim().is_empty() {
        return Err(CoreError::InvalidResponse(
            "OCR provider returned an empty success response".to_string(),
        ));
    }

    let response_json: Value = serde_json::from_str(&text)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(config
        .transform_ocr_response(model, response_json)?
        .into_json())
}

#[cfg(test)]
mod tests;
