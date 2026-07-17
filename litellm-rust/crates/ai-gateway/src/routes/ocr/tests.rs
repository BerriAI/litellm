use std::net::SocketAddr;
use std::sync::Arc;

use litellm_core::router::{Deployment, LiteLLMParams, Router as ModelRouter};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::io::realtime_pool::RealtimePool;
use crate::state::AppState;

const MASTER_KEY: &str = "sk-master-test";

async fn read_http_request(socket: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 2048];
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
    String::from_utf8_lossy(&request).into_owned()
}

async fn read_full_http_request(socket: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    let mut body_start: Option<usize> = None;
    let mut content_length = 0_usize;
    loop {
        let n = socket.read(&mut buffer).await.expect("reads request");
        if n == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..n]);
        if body_start.is_none() {
            if let Some(pos) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                let start = pos + 4;
                body_start = Some(start);
                let headers = String::from_utf8_lossy(&request[..pos]).to_ascii_lowercase();
                content_length = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
            }
        }
        if let Some(start) = body_start {
            if request.len() >= start + content_length {
                break;
            }
        }
    }
    String::from_utf8_lossy(&request).into_owned()
}

async fn spawn_mock_upstream_capture() -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds upstream");
    let addr = listener.local_addr().expect("upstream addr");
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts one request");
        let request = read_full_http_request(&mut socket).await;
        let body = r#"{"pages":[{"index":0,"markdown":"hello ocr"}],"model":"mistral-ocr-latest","usage_info":{"pages_processed":1}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("writes response");
        request
    });
    (format!("http://{addr}"), handle)
}

async fn spawn_mock_upstream() -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds upstream");
    let addr = listener.local_addr().expect("upstream addr");
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts one request");
        let request = read_http_request(&mut socket).await;
        let body = r#"{"pages":[{"index":0,"markdown":"hello ocr"}],"model":"mistral-ocr-latest","usage_info":{"pages_processed":1}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("writes response");
        request
    });
    (format!("http://{addr}"), handle)
}

async fn spawn_mock_upstream_error() -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds upstream");
    let addr = listener.local_addr().expect("upstream addr");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts one request");
        let _ = read_http_request(&mut socket).await;
        let body = r#"{"error":"invalid_api_key: sk-leaked-secret-value"}"#;
        let response = format!(
            "HTTP/1.1 403 Forbidden\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
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

fn app_with_deployment(api_base: &str) -> axum::Router {
    app_with_params("mistral/mistral-ocr-latest", None, api_base)
}

fn app_with_params(model: &str, custom_llm_provider: Option<&str>, api_base: &str) -> axum::Router {
    let router = ModelRouter::new(vec![Deployment {
        model_name: "rust-ocr-mistral".to_string(),
        litellm_params: LiteLLMParams {
            model: model.to_string(),
            api_key: Some("sk-upstream".to_string()),
            api_base: Some(api_base.to_string()),
            custom_llm_provider: custom_llm_provider.map(str::to_string),
        },
    }]);
    let state = AppState {
        router: Arc::new(router),
        master_key: Some(Arc::from(MASTER_KEY)),
        loggers: Arc::new(Vec::new()),
        realtime_pool: RealtimePool::disabled(),
    };
    crate::routes::app(state)
}

async fn serve(app: axum::Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds gateway");
    let addr = listener.local_addr().expect("gateway addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serves");
    });
    addr
}

#[tokio::test]
async fn json_document_url_returns_normalized_ocr_on_both_paths() {
    for path in ["/v1/ocr", "/ocr"] {
        let (upstream, upstream_handle) = spawn_mock_upstream().await;
        let addr = serve(app_with_deployment(&upstream)).await;

        let response = reqwest::Client::new()
            .post(format!("http://{addr}{path}"))
            .bearer_auth(MASTER_KEY)
            .json(&serde_json::json!({
                "model": "rust-ocr-mistral",
                "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
            }))
            .send()
            .await
            .expect("request sent");

        assert_eq!(response.status(), reqwest::StatusCode::OK, "path {path}");
        let body: Value = response.json().await.expect("json body");
        assert_eq!(body["object"], "ocr", "path {path}");
        assert_eq!(body["model"], "rust-ocr-mistral", "path {path}");
        assert_eq!(body["pages"][0]["markdown"], "hello ocr", "path {path}");

        let upstream_request = upstream_handle.await.expect("upstream served");
        assert!(
            upstream_request.contains("authorization: Bearer sk-upstream")
                || upstream_request.contains("Authorization: Bearer sk-upstream"),
            "upstream must receive the deployment credential: {upstream_request}"
        );
    }
}

#[tokio::test]
async fn explicit_custom_llm_provider_resolves_model_without_prefix() {
    let (upstream, upstream_handle) = spawn_mock_upstream().await;
    let addr = serve(app_with_params(
        "mistral-ocr-latest",
        Some("mistral"),
        &upstream,
    ))
    .await;

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .json(&serde_json::json!({
            "model": "rust-ocr-mistral",
            "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
        }))
        .send()
        .await
        .expect("request sent");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["object"], "ocr");
    assert_eq!(body["model"], "rust-ocr-mistral");

    let upstream_request = upstream_handle.await.expect("upstream served");
    assert!(upstream_request.starts_with("POST"), "{upstream_request}");
}

#[tokio::test]
async fn multipart_upload_returns_normalized_ocr() {
    let (upstream, upstream_handle) = spawn_mock_upstream().await;
    let addr = serve(app_with_deployment(&upstream)).await;

    let form = reqwest::multipart::Form::new()
        .text("model", "rust-ocr-mistral")
        .part(
            "file",
            reqwest::multipart::Part::bytes(b"%PDF-1.4 minimal".to_vec())
                .file_name("doc.pdf")
                .mime_str("application/pdf")
                .expect("mime"),
        );

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .multipart(form)
        .send()
        .await
        .expect("request sent");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["object"], "ocr");
    assert_eq!(body["pages"][0]["markdown"], "hello ocr");

    let upstream_request = upstream_handle.await.expect("upstream served");
    assert!(upstream_request.starts_with("POST"), "{upstream_request}");
}

#[tokio::test]
async fn multipart_unnamed_octet_stream_pdf_is_sniffed() {
    let (upstream, upstream_handle) = spawn_mock_upstream_capture().await;
    let addr = serve(app_with_deployment(&upstream)).await;

    let form = reqwest::multipart::Form::new()
        .text("model", "rust-ocr-mistral")
        .part(
            "file",
            reqwest::multipart::Part::bytes(b"%PDF-1.7 minimal pdf bytes".to_vec())
                .mime_str("application/octet-stream")
                .expect("mime"),
        );

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .multipart(form)
        .send()
        .await
        .expect("request sent");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let upstream_request = upstream_handle.await.expect("upstream served");
    assert!(
        upstream_request.contains("data:application/pdf;base64,"),
        "unnamed octet-stream PDF must be sniffed to application/pdf: {upstream_request}"
    );
}

#[tokio::test]
async fn multipart_unnamed_octet_stream_image_is_sniffed() {
    let (upstream, upstream_handle) = spawn_mock_upstream_capture().await;
    let addr = serve(app_with_deployment(&upstream)).await;

    let png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x01];
    let form = reqwest::multipart::Form::new()
        .text("model", "rust-ocr-mistral")
        .part(
            "file",
            reqwest::multipart::Part::bytes(png)
                .mime_str("application/octet-stream")
                .expect("mime"),
        );

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .multipart(form)
        .send()
        .await
        .expect("request sent");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let upstream_request = upstream_handle.await.expect("upstream served");
    assert!(
        upstream_request.contains("data:image/png;base64,"),
        "unnamed octet-stream PNG must be sniffed to image/png: {upstream_request}"
    );
}

#[tokio::test]
async fn upstream_error_status_is_propagated_without_leaking_provider_body() {
    let upstream = spawn_mock_upstream_error().await;
    let addr = serve(app_with_deployment(&upstream)).await;

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .json(&serde_json::json!({
            "model": "rust-ocr-mistral",
            "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
        }))
        .send()
        .await
        .expect("request sent");

    assert_eq!(response.status(), reqwest::StatusCode::FORBIDDEN);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "upstream_error");
    let message = body["error"]["message"].as_str().expect("message string");
    assert!(
        !message.contains("sk-leaked-secret-value") && !message.contains("invalid_api_key"),
        "provider body must not leak into the public error: {message}"
    );
}

#[tokio::test]
async fn missing_master_key_is_unauthorized() {
    let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .json(&serde_json::json!({
            "model": "rust-ocr-mistral",
            "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
        }))
        .send()
        .await
        .expect("request sent");
    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn unknown_model_is_not_found() {
    let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .json(&serde_json::json!({
            "model": "does-not-exist",
            "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}
        }))
        .send()
        .await
        .expect("request sent");
    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "not_found_error");
}

#[tokio::test]
async fn file_document_over_json_is_rejected() {
    let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .json(&serde_json::json!({
            "model": "rust-ocr-mistral",
            "document": {"type": "file", "file": "/etc/passwd"}
        }))
        .send()
        .await
        .expect("request sent");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn json_reserved_control_param_is_rejected() {
    let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .json(&serde_json::json!({
            "model": "rust-ocr-mistral",
            "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"},
            "api_base": "http://attacker.example"
        }))
        .send()
        .await
        .expect("request sent");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let message = body["error"]["message"].as_str().expect("message string");
    assert!(
        !message.contains("attacker.example"),
        "must not echo the attacker value: {message}"
    );
}

#[tokio::test]
async fn multipart_reserved_control_param_is_rejected() {
    let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
    let form = reqwest::multipart::Form::new()
        .text("model", "rust-ocr-mistral")
        .text("vertex_credentials", "/etc/gcp/service-account.json")
        .part(
            "file",
            reqwest::multipart::Part::bytes(b"%PDF-1.7 minimal".to_vec())
                .file_name("doc.pdf")
                .mime_str("application/pdf")
                .expect("mime"),
        );
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .multipart(form)
        .send()
        .await
        .expect("request sent");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn duplicate_file_multipart_is_rejected() {
    let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
    let form = reqwest::multipart::Form::new()
        .text("model", "rust-ocr-mistral")
        .part(
            "file",
            reqwest::multipart::Part::bytes(b"%PDF-1.7 first".to_vec())
                .file_name("a.pdf")
                .mime_str("application/pdf")
                .expect("mime"),
        )
        .part(
            "file",
            reqwest::multipart::Part::bytes(b"%PDF-1.7 second".to_vec())
                .file_name("b.pdf")
                .mime_str("application/pdf")
                .expect("mime"),
        );
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .multipart(form)
        .send()
        .await
        .expect("request sent");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn non_positive_timeout_is_rejected() {
    let addr = serve(app_with_deployment("http://127.0.0.1:1")).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/ocr"))
        .bearer_auth(MASTER_KEY)
        .json(&serde_json::json!({
            "model": "rust-ocr-mistral",
            "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"},
            "timeout": -1
        }))
        .send()
        .await
        .expect("request sent");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}
