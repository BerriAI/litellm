//! End-to-end OCR orchestration (async).
//!
//! The typed foundation: dispatch on [`OcrProvider`], build the URL + body via
//! the pure per-provider transforms, POST it with a shared async client, and
//! parse the response into a typed [`OcrResponse`]. Blocking work happens off the
//! Python GIL — the bridge either awaits this future (`aocr`) or `block_on`s it
//! (`ocr`).

use std::sync::OnceLock;
use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::ocr::types::{OcrProvider, OcrRequest, OcrResponse};
use litellm_core::CoreResult;

use crate::mistral::ocr::transformation as mistral;

/// OCR over large documents can take a while; bound it generously rather than
/// hanging forever on an unresponsive upstream. The client-level limit is the
/// outer ceiling; callers can tighten it per request via `OcrRequest::timeout`.
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

/// Wire label for a provider, for error messages on unsupported providers.
fn provider_label(provider: OcrProvider) -> &'static str {
    match provider {
        OcrProvider::Mistral => "mistral",
        OcrProvider::AzureAi => "azure_ai",
        OcrProvider::AzureDocumentIntelligence => "azure_document_intelligence",
        OcrProvider::VertexAi => "vertex_ai",
        OcrProvider::VertexDeepseek => "vertex_deepseek",
        OcrProvider::Reducto => "reducto",
    }
}

/// Perform an OCR call end to end and return the typed, standardized response.
///
/// Async: intended to be awaited (`aocr`) or `block_on`'d (`ocr`) from the
/// Python bridge, both of which keep the GIL free during the HTTP wait.
pub async fn ocr(request: OcrRequest) -> CoreResult<OcrResponse> {
    match request.provider {
        OcrProvider::Mistral => mistral_ocr(request).await,
        other => Err(CoreError::UnsupportedProvider(
            provider_label(other).to_string(),
        )),
    }
}

async fn mistral_ocr(request: OcrRequest) -> CoreResult<OcrResponse> {
    let api_key =
        mistral::resolve_api_key(request.api_key.as_deref(), &|key| std::env::var(key).ok())?;
    let url = mistral::complete_url(request.api_base.as_deref());
    let body = mistral::request_body(&request.model, &request.document, &request.params);

    // Let an explicit Authorization header in extra_headers win over the resolved key.
    let caller_set_auth = request
        .extra_headers
        .keys()
        .any(|name| name.eq_ignore_ascii_case("authorization"));

    let mut builder = http_client().post(&url).json(&body);
    if !caller_set_auth {
        builder = builder.bearer_auth(&api_key);
    }
    for (name, value) in &request.extra_headers {
        builder = builder.header(name, value);
    }
    if let Some(duration) = request.timeout {
        builder = builder.timeout(duration);
    }

    let response = builder
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

    mistral::parse_response(&request.model, &text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_core::ocr::types::{OcrDocument, OcrParams};
    use std::collections::BTreeMap;

    fn request(provider: OcrProvider) -> OcrRequest {
        OcrRequest {
            provider,
            model: "m".to_string(),
            document: OcrDocument::DocumentUrl {
                document_url: "https://example.com/doc.pdf".to_string(),
            },
            api_key: Some("sk-test".to_string()),
            api_base: None,
            extra_headers: BTreeMap::new(),
            timeout: None,
            params: OcrParams::default(),
        }
    }

    #[test]
    fn unsupported_provider_errors_with_label() {
        let err = futures_block_on(ocr(request(OcrProvider::Reducto)))
            .expect_err("reducto not wired yet");
        assert_eq!(err, CoreError::UnsupportedProvider("reducto".to_string()));
    }

    #[test]
    fn truncate_error_body_caps_long_payloads() {
        let body = "x".repeat(ERROR_BODY_MAX_CHARS + 50);
        let truncated = truncate_error_body(&body);
        assert!(truncated.ends_with("... (truncated)"));
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn truncate_error_body_passes_short_strings_through() {
        assert_eq!(truncate_error_body("Unauthorized"), "Unauthorized");
    }

    /// Minimal single-threaded executor so dispatch logic can be unit-tested
    /// without a network or a full Tokio runtime.
    fn futures_block_on<F: std::future::Future>(fut: F) -> F::Output {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        rt.block_on(fut)
    }
}
