use serde_json::{Map, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use super::{OcrRequest, ocr};

fn test_client() -> reqwest::Client {
    let headers = reqwest::header::HeaderMap::from_iter([(
        reqwest::header::HeaderName::from_static("x-injected-client"),
        reqwest::header::HeaderValue::from_static("true"),
    )]);
    reqwest::Client::builder()
        .default_headers(headers)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("test client builds")
}

async fn read_http_request(socket: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    let header_end = loop {
        let count = socket.read(&mut buffer).await.expect("reads request");
        if count == 0 {
            break request.len();
        }
        request.extend_from_slice(&buffer[..count]);
        if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    while request.len().saturating_sub(header_end) < content_length {
        let count = socket.read(&mut buffer).await.expect("reads body");
        if count == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..count]);
    }
    String::from_utf8(request).expect("request is utf8")
}

async fn write_response(socket: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("writes response");
}

fn base_request<'a>(model: &'a str, api_base: &'a str) -> OcrRequest<'a> {
    OcrRequest {
        model,
        document: json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        }),
        api_key: Some("sk-test"),
        api_base: Some(api_base),
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: Map::new(),
        timeout: None,
    }
}

#[tokio::test]
async fn core_builds_sends_and_parses_mistral_request() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let address = listener.local_addr().expect("listener has local address");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        write_response(
            &mut socket,
            "200 OK",
            r#"{"pages":[{"index":0,"markdown":"hello"}],"model":"mistral-ocr-latest","usage_info":{"pages_processed":1}}"#,
        )
        .await;
        request
    });
    let api_base = format!("http://{address}");
    let mut request = base_request("mistral/mistral-ocr-latest", &api_base);
    request.extra_headers = Some(Map::from_iter([(
        "x-trace-id".to_string(),
        json!("trace-1"),
    )]));
    request.optional_params = Map::from_iter([("pages".to_string(), json!([0]))]);

    let client = test_client();
    let response = ocr(&client, request).await.expect("Mistral OCR succeeds");

    assert_eq!(response["pages"][0]["markdown"], "hello");
    let provider_request = server.await.expect("server task completes");
    assert!(provider_request.starts_with("POST /v1/ocr HTTP/1.1"));
    assert!(
        provider_request
            .to_ascii_lowercase()
            .contains("authorization: bearer sk-test")
    );
    assert!(provider_request.contains("x-trace-id: trace-1"));
    assert!(provider_request.contains("x-injected-client: true"));
    assert_eq!(
        provider_request
            .lines()
            .filter(|line| line.to_ascii_lowercase().starts_with("content-type:"))
            .count(),
        1
    );
    assert!(provider_request.contains(r#""model":"mistral-ocr-latest""#));
    assert!(provider_request.contains(r#""pages":[0]"#));
}

#[tokio::test]
async fn vertex_routing_params_are_not_forwarded_in_provider_body() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let address = listener.local_addr().expect("listener has local address");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        write_response(
            &mut socket,
            "200 OK",
            r#"{"pages":[{"index":0,"markdown":"hello"}],"model":"mistral-ocr-maas","usage_info":{"pages_processed":1}}"#,
        )
        .await;
        request
    });
    let api_base = format!("http://{address}");
    let mut request = base_request("vertex_ai/mistral-ocr-maas", &api_base);
    request.document = json!({
        "type": "image_url",
        "image_url": "data:image/png;base64,aGVsbG8="
    });
    request.optional_params = Map::from_iter([
        ("vertex_project".to_string(), json!("proj-1")),
        ("vertex_location".to_string(), json!("europe-west4")),
        ("pages".to_string(), json!([0])),
    ]);

    let client = test_client();
    ocr(&client, request).await.expect("Vertex OCR succeeds");

    let provider_request = server.await.expect("server task completes");
    assert!(provider_request.starts_with(
        "POST /v1/projects/proj-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict HTTP/1.1"
    ));
    assert!(provider_request.contains(r#""pages":[0]"#));
    assert!(!provider_request.contains("vertex_project"));
    assert!(!provider_request.contains("vertex_location"));
}

#[tokio::test]
async fn provider_http_error_preserves_status_and_bounds_body() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let address = listener.local_addr().expect("listener has local address");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let _request = read_http_request(&mut socket).await;
        write_response(&mut socket, "401 Unauthorized", &"x".repeat(400)).await;
    });
    let api_base = format!("http://{address}");

    let client = test_client();
    let error = ocr(
        &client,
        base_request("mistral/mistral-ocr-latest", &api_base),
    )
    .await
    .expect_err("provider rejection should fail");

    let crate::Error::Http { status, body } = error else {
        panic!("expected provider HTTP error");
    };
    assert_eq!(status, 401);
    assert!(body.ends_with("... (truncated)"));
    assert!(body.len() < 400);
    server.await.expect("server task completes");
}

#[tokio::test]
async fn reducto_upload_and_parse_networking_is_owned_by_core() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let address = listener.local_addr().expect("listener has local address");
    let server = tokio::spawn(async move {
        let (mut upload_socket, _) = listener.accept().await.expect("accepts upload request");
        let upload_request = read_http_request(&mut upload_socket).await;
        write_response(
            &mut upload_socket,
            "200 OK",
            r#"{"file_id":"reducto://uploaded.pdf"}"#,
        )
        .await;

        let (mut parse_socket, _) = listener.accept().await.expect("accepts parse request");
        let parse_request = read_http_request(&mut parse_socket).await;
        write_response(
            &mut parse_socket,
            "200 OK",
            r#"{"job_id":"job_123","usage":{"num_pages":1,"credits":1},"result":{"chunks":[{"content":"Page 1","blocks":[{"content":"Page 1","bbox":{"page":1},"kind":"text"}]}]}}"#,
        )
        .await;
        (upload_request, parse_request)
    });
    let api_base = format!("http://{address}");
    let mut request = base_request("reducto/parse-v3", &api_base);
    request.api_key = None;
    request.extra_headers = Some(Map::from_iter([
        ("Authorization".to_string(), json!("Bearer test-key")),
        ("x-trace-id".to_string(), json!("trace-1")),
    ]));
    request.document = json!({
        "type": "document_url",
        "document_url": "data:application/pdf;base64,JVBERi0xLjQ="
    });
    request.optional_params =
        Map::from_iter([("settings".to_string(), json!({"ocr_system": "standard"}))]);

    let client = test_client();
    let response = ocr(&client, request).await.expect("Reducto OCR succeeds");

    assert_eq!(response["pages"][0]["markdown"], "Page 1");
    assert!(response.get("provider_native_response").is_none());
    let (upload_request, parse_request) = server.await.expect("server task completes");
    assert!(upload_request.starts_with("POST /upload HTTP/1.1"));
    assert!(upload_request.contains("x-injected-client: true"));
    assert!(upload_request.contains("%PDF-1.4"));
    assert!(parse_request.starts_with("POST /parse HTTP/1.1"));
    assert!(parse_request.contains("x-injected-client: true"));
    assert!(parse_request.contains(r#""input":"reducto://uploaded.pdf""#));
    assert!(parse_request.contains(r#""ocr_system":"standard""#));
}
