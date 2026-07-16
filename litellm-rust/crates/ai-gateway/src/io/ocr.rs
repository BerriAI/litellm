//! End-to-end OCR orchestration.
//!
//! Owns supported OCR provider calls so the Python side stays a thin bridge:
//! resolve the API key, build the URL + body via the pure transforms, POST it,
//! and normalize the response. The HTTP client is built once and reused.

use std::sync::OnceLock;
use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::{
    OcrAuth, OcrAuthStrategy, OcrDocumentPreparation, OcrResponseHandling,
};
use litellm_core::providers::vertex_ai::ocr::transformation::{
    classify_vertex_bearer, VertexTokenSource,
};
use litellm_core::CoreResult;
use serde_json::{Map, Value};

use crate::io::vertex_auth;

mod common_utils;

use common_utils::{
    convert_document_url_to_data_uri, has_header, ocr_provider_config, poll_document_intelligence,
    string_headers, truncate_error_body, upload_reducto_document,
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

fn vertex_credentials(optional_params: &Map<String, Value>) -> Option<String> {
    ["vertex_credentials", "vertex_ai_credentials"]
        .iter()
        .find_map(|key| optional_params.get(*key))
        .and_then(|value| match value {
            Value::String(raw) => {
                let trimmed = raw.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            }
            Value::Object(_) => serde_json::to_string(value).ok(),
            _ => None,
        })
}

fn resolve_vertex_credentials(
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    vertex_credentials(optional_params).or_else(|| {
        env_lookup(crate::constants::VERTEXAI_CREDENTIALS_ENV)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
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
    let model = request.model;
    let config = ocr_provider_config(request.custom_llm_provider, model)
        .ok_or_else(|| CoreError::InvalidProvider(request.custom_llm_provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();

    let headers = string_headers(request.extra_headers)?;
    let auth_strategy = config.auth_strategy();
    let api_key = if has_header(&headers, auth_strategy.header_name()) {
        None
    } else {
        Some(match config.ocr_auth() {
            OcrAuth::ProviderKey => config.resolve_api_key(request.api_key, &env_lookup)?,
            OcrAuth::VertexOauth => match classify_vertex_bearer(request.api_key, &env_lookup)? {
                VertexTokenSource::Explicit(token) => token,
                VertexTokenSource::Mint => {
                    let credentials =
                        resolve_vertex_credentials(&request.optional_params, &env_lookup);
                    vertex_auth::mint_vertex_bearer(credentials.as_deref()).await?
                }
            },
        })
    };
    let url = config.complete_url(
        request.api_base,
        model,
        &request.optional_params,
        &env_lookup,
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
        .map_err(|err| CoreError::Network(err.to_string()))?;

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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    async fn read_http_headers(socket: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let n = socket.read(&mut buffer).await.expect("reads request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..n]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        String::from_utf8(request).expect("request is utf8")
    }

    #[test]
    fn truncate_error_body_passes_short_strings_through() {
        let body = "Unauthorized";
        assert_eq!(truncate_error_body(body), "Unauthorized");
    }

    #[test]
    fn truncate_error_body_caps_long_payloads() {
        let body = "x".repeat(306);
        let truncated = truncate_error_body(&body);

        assert!(truncated.ends_with("... (truncated)"));
        let prefix_chars = truncated
            .strip_suffix("... (truncated)")
            .expect("truncated marker present")
            .chars()
            .count();
        assert_eq!(prefix_chars, 256);
    }

    #[test]
    fn truncate_error_body_does_not_split_multibyte_chars() {
        let body = "é".repeat(266);
        let truncated = truncate_error_body(&body);
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn ocr_dispatch_supports_migrated_providers() {
        assert!(ocr_provider_config("mistral", "mistral-ocr-latest").is_some());
        assert!(ocr_provider_config("azure_ai", "pixtral-12b-2409")
            .expect("azure ai config resolves")
            .requires_data_uri_document());
        assert_eq!(
            ocr_provider_config("azure_ai", "doc-intelligence/prebuilt-read")
                .expect("document intelligence config resolves")
                .response_handling(),
            OcrResponseHandling::AzureDocumentIntelligencePoll
        );
        assert!(ocr_provider_config("vertex_ai", "deepseek-ocr-maas")
            .expect("vertex deepseek config resolves")
            .supported_ocr_params()
            .contains(&"temperature"));
        assert!(ocr_provider_config("reducto", "parse-v3").is_some());
        assert!(ocr_provider_config("reducto", "parse-legacy").is_some());
        assert!(ocr_provider_config("openai", "gpt-4o").is_none());
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
    fn auth_header_detection_is_case_insensitive() {
        let headers = vec![
            ("x-trace-id".to_string(), "trace-1".to_string()),
            ("authorization".to_string(), "Bearer sk-test".to_string()),
        ];

        assert!(has_header(&headers, "authorization"));

        let headers = vec![("Authorization".to_string(), "Bearer sk-test".to_string())];
        assert!(has_header(&headers, "authorization"));

        let headers = vec![("x-trace-id".to_string(), "trace-1".to_string())];
        assert!(!has_header(&headers, "authorization"));
    }

    #[tokio::test]
    async fn ocr_does_not_duplicate_authorization_header_when_header_is_supplied() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts one request");
            let request = read_http_headers(&mut socket).await;

            let response_body = r#"{"pages":[{"index":0,"markdown":"ok"}],"model":"mistral-ocr-latest","usage_info":{"pages_processed":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            request
        });

        let mut headers = Map::new();
        headers.insert(
            "Authorization".to_string(),
            Value::String("Bearer sk-from-python".to_string()),
        );
        headers.insert(
            "x-trace-id".to_string(),
            Value::String("trace-1".to_string()),
        );

        let response = ocr(OcrRequest {
            model: "mistral-ocr-latest",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("sk-for-rust-fallback"),
            api_base: Some(&format!("http://{addr}")),
            custom_llm_provider: "mistral",
            extra_headers: Some(headers),
            optional_params: Map::new(),
            timeout: Some(Duration::from_secs(5)),
        })
        .await
        .expect("ocr request succeeds");

        assert_eq!(response["pages"][0]["markdown"], "ok");

        let request = server.await.expect("server task completes");
        let authorization_count = request
            .lines()
            .filter(|line| line.to_ascii_lowercase().starts_with("authorization:"))
            .count();
        assert_eq!(authorization_count, 1, "{request}");
        assert!(
            request.contains("authorization: Bearer sk-from-python")
                || request.contains("Authorization: Bearer sk-from-python"),
            "{request}"
        );
    }

    #[test]
    fn vertex_credentials_reads_string_object_and_treats_blank_as_absent() {
        let mut inline = Map::new();
        inline.insert(
            "vertex_credentials".to_string(),
            Value::String("  /path/to/sa.json  ".into()),
        );
        assert_eq!(
            vertex_credentials(&inline).as_deref(),
            Some("/path/to/sa.json")
        );

        let mut object = Map::new();
        object.insert(
            "vertex_credentials".to_string(),
            json!({"type": "service_account"}),
        );
        assert_eq!(
            vertex_credentials(&object).as_deref(),
            Some("{\"type\":\"service_account\"}")
        );

        let mut blank = Map::new();
        blank.insert(
            "vertex_credentials".to_string(),
            Value::String("   ".into()),
        );
        assert_eq!(vertex_credentials(&blank), None);

        assert_eq!(vertex_credentials(&Map::new()), None);
    }

    #[test]
    fn resolve_vertex_credentials_prefers_optional_param_then_env() {
        let mut params = Map::new();
        params.insert(
            "vertex_credentials".to_string(),
            Value::String("/from/param.json".into()),
        );
        let env = |key: &str| {
            (key == crate::constants::VERTEXAI_CREDENTIALS_ENV)
                .then(|| "/from/env.json".to_string())
        };
        assert_eq!(
            resolve_vertex_credentials(&params, &env).as_deref(),
            Some("/from/param.json")
        );

        assert_eq!(
            resolve_vertex_credentials(&Map::new(), &env).as_deref(),
            Some("/from/env.json")
        );

        let blank_env = |key: &str| {
            (key == crate::constants::VERTEXAI_CREDENTIALS_ENV).then(|| "   ".to_string())
        };
        assert_eq!(resolve_vertex_credentials(&Map::new(), &blank_env), None);

        assert_eq!(resolve_vertex_credentials(&Map::new(), &|_| None), None);
    }

    #[tokio::test]
    async fn vertex_ocr_sends_explicit_oauth_bearer_to_raw_predict_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts one request");
            let request = read_http_headers(&mut socket).await;
            let response_body = r#"{"pages":[{"index":0,"markdown":"ok"}],"model":"mistral-ocr-maas","usage_info":{"pages_processed":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            request
        });

        let mut optional_params = Map::new();
        optional_params.insert("vertex_project".to_string(), Value::String("proj-1".into()));
        optional_params.insert(
            "vertex_location".to_string(),
            Value::String("global".into()),
        );

        let response = ocr(OcrRequest {
            model: "mistral-ocr-maas",
            document: json!({
                "type": "image_url",
                "image_url": "data:image/png;base64,abc"
            }),
            api_key: Some("ya29.explicit-oauth-token"),
            api_base: Some(&format!("http://{addr}")),
            custom_llm_provider: "vertex_ai",
            extra_headers: None,
            optional_params,
            timeout: Some(Duration::from_secs(5)),
        })
        .await
        .expect("vertex ocr request succeeds");

        assert_eq!(response["pages"][0]["markdown"], "ok");

        let request = server.await.expect("server task completes");
        let request_line = request.lines().next().unwrap_or_default();
        assert!(
            request_line
                .contains("/v1/projects/proj-1/locations/global/publishers/mistralai/models/mistral-ocr-maas:rawPredict"),
            "{request}"
        );
        let authorization_count = request
            .lines()
            .filter(|line| line.to_ascii_lowercase().starts_with("authorization:"))
            .count();
        assert_eq!(authorization_count, 1, "{request}");
        assert!(
            request.contains("authorization: Bearer ya29.explicit-oauth-token")
                || request.contains("Authorization: Bearer ya29.explicit-oauth-token"),
            "{request}"
        );
    }

    #[tokio::test]
    async fn vertex_ocr_rejects_google_api_key_shaped_token_before_calling_upstream() {
        let mut optional_params = Map::new();
        optional_params.insert("vertex_project".to_string(), Value::String("proj-1".into()));
        optional_params.insert(
            "vertex_location".to_string(),
            Value::String("global".into()),
        );

        let err = ocr(OcrRequest {
            model: "mistral-ocr-maas",
            document: json!({
                "type": "image_url",
                "image_url": "data:image/png;base64,abc"
            }),
            api_key: Some("AIzaSyExampleApiKeyValue000000000000000"),
            api_base: Some("http://192.0.2.1:9"),
            custom_llm_provider: "vertex_ai",
            extra_headers: None,
            optional_params,
            timeout: Some(Duration::from_secs(5)),
        })
        .await
        .expect_err("google api key is rejected");

        match err {
            CoreError::Auth(message) => assert!(message.contains("OAuth"), "{message}"),
            other => panic!("expected auth error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn document_intelligence_poll_uses_resolved_subscription_key() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");
        let operation_url = format!("http://{addr}/operations/1");

        let server = tokio::spawn(async move {
            let (mut post_socket, _) = listener.accept().await.expect("accepts post request");
            let post_request = read_http_headers(&mut post_socket).await;
            let post_response = format!(
                "HTTP/1.1 202 Accepted\r\noperation-location: {operation_url}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            );
            post_socket
                .write_all(post_response.as_bytes())
                .await
                .expect("writes post response");

            let (mut poll_socket, _) = listener.accept().await.expect("accepts poll request");
            let poll_request = read_http_headers(&mut poll_socket).await;
            let response_body = r#"{"status":"succeeded","analyzeResult":{"pages":[{"pageNumber":1,"lines":[{"content":"ok"}]}]}}"#;
            let poll_response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            poll_socket
                .write_all(poll_response.as_bytes())
                .await
                .expect("writes poll response");
            (post_request, poll_request)
        });

        let response = ocr(OcrRequest {
            model: "prebuilt-read",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("di-key"),
            api_base: Some(&format!("http://{addr}")),
            custom_llm_provider: "azure_ai/doc-intelligence",
            extra_headers: None,
            optional_params: Map::new(),
            timeout: Some(Duration::from_secs(5)),
        })
        .await
        .expect("document intelligence request succeeds");

        assert_eq!(response["pages"][0]["markdown"], "ok");

        let (post_request, poll_request) = server.await.expect("server task completes");
        assert!(
            post_request
                .to_ascii_lowercase()
                .contains("ocp-apim-subscription-key: di-key"),
            "{post_request}"
        );
        assert!(
            poll_request
                .to_ascii_lowercase()
                .contains("ocp-apim-subscription-key: di-key"),
            "{poll_request}"
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
