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

async fn read_http_request(socket: &mut TcpStream) -> (String, Value) {
    let mut raw = Vec::new();
    let mut buffer = [0_u8; 1024];
    loop {
        let n = socket.read(&mut buffer).await.expect("reads request");
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..n]);
        let header_end = raw
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|pos| pos + 4);
        if let Some(body_start) = header_end {
            let text = String::from_utf8(raw.clone()).expect("request is utf8");
            let content_length = text
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|value| value.trim().parse::<usize>().expect("content-length"))
                })
                .unwrap_or(0);
            if raw.len() >= body_start + content_length {
                let headers = text[..body_start].to_string();
                let body: Value =
                    serde_json::from_slice(&raw[body_start..body_start + content_length])
                        .expect("request body is json");
                return (headers, body);
            }
        }
    }
    panic!("did not receive a complete request");
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

#[tokio::test]
async fn ocr_resolves_api_key_and_base_references_in_rust() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let addr = listener.local_addr().expect("listener has local addr");
    let resolved_base = format!("http://{addr}");

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

    let env_lookup = |name: &str| match name {
        "OCR_TEST_API_KEY" => Some("sk-resolved".to_string()),
        "OCR_TEST_API_BASE" => Some(resolved_base.clone()),
        _ => None,
    };

    let response = ocr_with_env(
        OcrRequest {
            model: "mistral-ocr-latest",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("os.environ/OCR_TEST_API_KEY"),
            api_base: Some("os.environ/OCR_TEST_API_BASE"),
            custom_llm_provider: "mistral",
            extra_headers: None,
            optional_params: Map::new(),
            timeout: Some(Duration::from_secs(5)),
        },
        &env_lookup,
    )
    .await
    .expect("ocr request succeeds");

    assert_eq!(response["pages"][0]["markdown"], "ok");

    let request = server.await.expect("server task completes");
    assert!(request.starts_with("POST /v1/ocr HTTP/1.1"), "{request}");
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer sk-resolved"),
        "{request}"
    );
    assert!(!request.contains("os.environ/"), "{request}");
}

#[tokio::test]
async fn ocr_forwards_full_mistral_contract_and_filters_internal_params() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let addr = listener.local_addr().expect("listener has local addr");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts one request");
        let (_headers, body) = read_http_request(&mut socket).await;

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
        body
    });

    let document = json!({
        "type": "document_url",
        "document_url": "https://example.com/doc.pdf"
    });
    let optional_params = json!({
        "pages": [0, 2, 5],
        "include_image_base64": true,
        "image_limit": 10,
        "image_min_size": 64,
        "bbox_annotation_format": {"type": "text"},
        "document_annotation_format": {"type": "json_schema"},
        "document_annotation_prompt": "extract title",
        "extract_header": true,
        "extract_footer": false,
        "table_format": "html",
        "confidence_scores_granularity": "word",
        "include_blocks": true,
        "id": "ocr-req-9",
        "litellm_metadata": {"trace": "internal"},
        "metadata": {"trace": "internal"},
        "num_retries": 3
    })
    .as_object()
    .unwrap()
    .clone();

    ocr(OcrRequest {
        model: "mistral-ocr-latest",
        document: document.clone(),
        api_key: Some("sk-test"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: "mistral",
        extra_headers: None,
        optional_params,
        timeout: Some(Duration::from_secs(5)),
    })
    .await
    .expect("ocr request succeeds");

    let body = server.await.expect("server task completes");

    assert_eq!(
        body,
        json!({
            "model": "mistral-ocr-latest",
            "document": document,
            "pages": [0, 2, 5],
            "include_image_base64": true,
            "image_limit": 10,
            "image_min_size": 64,
            "bbox_annotation_format": {"type": "text"},
            "document_annotation_format": {"type": "json_schema"},
            "document_annotation_prompt": "extract title",
            "extract_header": true,
            "extract_footer": false,
            "table_format": "html",
            "confidence_scores_granularity": "word",
            "include_blocks": true,
            "id": "ocr-req-9"
        })
    );
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

async fn respond_once(status_line: &'static str, body: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let addr = listener.local_addr().expect("listener has local addr");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts one request");
        let _ = read_http_headers(&mut socket).await;
        let response = format!(
            "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("writes response");
    });
    format!("http://{addr}")
}

async fn run_mistral_ocr(api_base: String, timeout: Duration) -> CoreResult<Value> {
    ocr(OcrRequest {
        model: "mistral-ocr-latest",
        document: json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        }),
        api_key: Some("sk-test"),
        api_base: Some(&api_base),
        custom_llm_provider: "mistral",
        extra_headers: None,
        optional_params: Map::new(),
        timeout: Some(timeout),
    })
    .await
}

#[tokio::test]
async fn ocr_preserves_upstream_error_status() {
    for status in [
        "401 Unauthorized",
        "404 Not Found",
        "500 Internal Server Error",
    ] {
        let base = respond_once(status, r#"{"error":"nope"}"#.to_string()).await;
        let err = run_mistral_ocr(base, Duration::from_secs(5))
            .await
            .expect_err("upstream error surfaces");
        let expected = status[..3].parse::<u16>().expect("status prefix parses");
        match err {
            CoreError::Http { status: got, .. } => assert_eq!(got, expected),
            other => panic!("expected Http error, got {other:?}"),
        }
        assert_eq!(err.public_status_code(), Some(expected));
    }
}

#[tokio::test]
async fn ocr_bounds_oversized_error_body() {
    let base = respond_once("500 Internal Server Error", "x".repeat(6000)).await;
    let err = run_mistral_ocr(base, Duration::from_secs(5))
        .await
        .expect_err("oversized error surfaces");
    match err {
        CoreError::Http { body, .. } => {
            assert!(body.ends_with("... (truncated)"));
            assert!(
                body.chars().count() < 300,
                "body not bounded: {} chars",
                body.chars().count()
            );
        }
        other => panic!("expected Http error, got {other:?}"),
    }
}

#[tokio::test]
async fn ocr_rejects_invalid_json_success_body() {
    let base = respond_once("200 OK", "not json".to_string()).await;
    let err = run_mistral_ocr(base, Duration::from_secs(5))
        .await
        .expect_err("invalid JSON surfaces");
    assert!(matches!(err, CoreError::InvalidResponse(_)));
    assert_eq!(err.public_status_code(), Some(500));
}

#[tokio::test]
async fn ocr_rejects_empty_success_body() {
    let base = respond_once("200 OK", "   ".to_string()).await;
    let err = run_mistral_ocr(base, Duration::from_secs(5))
        .await
        .expect_err("empty success surfaces");
    match err {
        CoreError::InvalidResponse(message) => assert!(message.contains("empty")),
        other => panic!("expected InvalidResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn ocr_classifies_per_request_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let addr = listener.local_addr().expect("listener has local addr");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts one request");
        let _ = read_http_headers(&mut socket).await;
        tokio::time::sleep(Duration::from_secs(3)).await;
        let _ = socket.write_all(b"HTTP/1.1 200 OK\r\n\r\n").await;
    });
    let err = run_mistral_ocr(format!("http://{addr}"), Duration::from_millis(200))
        .await
        .expect_err("timeout surfaces");
    assert_eq!(err, CoreError::Timeout);
    assert_eq!(err.public_status_code(), Some(408));
}

#[tokio::test]
async fn ocr_maps_unregistered_provider_to_invalid_provider() {
    let err = ocr(OcrRequest {
        model: "some-model",
        document: json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        }),
        api_key: Some("sk-test"),
        api_base: None,
        custom_llm_provider: "definitely-not-a-provider",
        extra_headers: None,
        optional_params: Map::new(),
        timeout: Some(Duration::from_secs(5)),
    })
    .await
    .expect_err("unregistered provider surfaces");
    assert!(matches!(err, CoreError::InvalidProvider(_)));
    assert_eq!(err.public_status_code(), Some(400));
    assert_eq!(err.public_message(), "Invalid OCR request");
}
