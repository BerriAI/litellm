use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::Error;
use crate::http_utils::has_header;
use crate::ocr::observers::{OcrObserver, OcrPostCall, OcrPreCall};
use crate::ocr::prepare::ocr_provider_config;
use crate::ocr::transformation::OcrResponseHandling;
use crate::ocr::{OcrRequest, ocr, ocr_with_observer};
use crate::provider_callbacks::CallbackDecision;
use crate::request_context::{LiteLlmRequestContext, RequestAttribution, RequestCapabilities};

struct ProviderObserver {
    events: Arc<Mutex<Vec<&'static str>>>,
    raw_response: Option<String>,
    reject: bool,
}

impl OcrObserver for ProviderObserver {
    type Error = &'static str;

    async fn pre_call(&mut self, input: &OcrPreCall) -> Result<CallbackDecision, Self::Error> {
        assert_eq!(input.call_id.as_deref(), Some("observer-test"));
        assert_eq!(input.trace_id.as_deref(), Some("trace-test"));
        assert_eq!(
            input.requested_model.as_deref(),
            Some("mistral/original-model")
        );
        assert_eq!(
            input.attribution.user_api_key_team_id.as_deref(),
            Some("team-1")
        );
        assert_eq!(input.capabilities.execution_mode.as_deref(), Some("async"));
        assert_eq!(input.model, "mistral-ocr-4-1");
        assert_eq!(
            input.request.data["document"]["document_url"],
            "https://example.com/document.pdf"
        );
        assert!(input.api_base.ends_with("/v1/ocr"));
        assert!(!input.headers.contains_key("authorization"));
        self.events.lock().unwrap().push("pre");
        if self.reject {
            Err("observer failure")
        } else {
            Ok(CallbackDecision::Unchanged)
        }
    }

    async fn post_call(&mut self, input: &OcrPostCall) -> Result<CallbackDecision, Self::Error> {
        self.events.lock().unwrap().push("post");
        self.raw_response = Some(input.original_response.clone());
        if self.reject {
            Err("observer failure")
        } else {
            Ok(CallbackDecision::Unchanged)
        }
    }
}

fn observer_request(api_base: &str) -> OcrRequest<'_> {
    OcrRequest {
        model: "mistral/mistral-ocr-4-1",
        document: json!({"type":"document_url","document_url":"https://example.com/document.pdf"}),
        api_key: Some("test-key"),
        api_base: Some(api_base),
        custom_llm_provider: Some("mistral"),
        extra_headers: None,
        optional_params: Map::new(),
        timeout: Some(Duration::from_secs(2)),
        litellm_call_id: Some("observer-test"),
        context: LiteLlmRequestContext {
            litellm_call_id: Some("observer-test".into()),
            trace_id: Some("trace-test".into()),
            request_model: Some("mistral/original-model".into()),
            attribution: RequestAttribution {
                user_api_key_team_id: Some("team-1".into()),
                ..Default::default()
            },
            capabilities: RequestCapabilities {
                execution_mode: Some("async".into()),
                ..Default::default()
            },
        },
    }
}

async fn observer_case(status: u16, body: &'static str) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let events = Arc::new(Mutex::new(Vec::new()));
    let provider_events = Arc::clone(&events);
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_http_request(&mut socket).await;
        assert!(request.starts_with("POST /v1/ocr "));
        provider_events.lock().unwrap().push("http");
        let response = format!(
            "HTTP/1.1 {status} Test\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    let mut observer = ProviderObserver {
        events: Arc::clone(&events),
        raw_response: None,
        reject: false,
    };
    let result = ocr_with_observer(observer_request(&url), &mut observer).await;
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap();
    if status != 200 {
        assert!(matches!(result, Err(Error::Http { status: actual, .. }) if actual == status));
        assert_eq!(*events.lock().unwrap(), ["pre", "http"]);
        assert_eq!(observer.raw_response, None);
    } else {
        assert_eq!(*events.lock().unwrap(), ["pre", "http", "post"]);
        assert_eq!(observer.raw_response.as_deref(), Some(body));
        if body == "invalid-json" {
            assert!(matches!(result, Err(Error::InvalidResponse(_))));
        } else {
            assert_eq!(result.unwrap()["pages"][0]["markdown"], "ok");
        }
    }
}

#[tokio::test]
async fn provider_observers_surround_http() {
    observer_case(200, r#"{"pages":[{"index":0,"markdown":"ok"}]}"#).await;
    observer_case(200, "invalid-json").await;
    observer_case(401, r#"{"error":"rejected"}"#).await;
}

#[tokio::test]
async fn provider_pre_call_failure_is_terminal_before_http() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut observer = ProviderObserver {
        events: Arc::clone(&events),
        raw_response: None,
        reject: true,
    };

    let result = ocr_with_observer(observer_request(&url), &mut observer).await;

    assert!(
        matches!(result, Err(Error::InvalidResponse(message)) if message.contains("observer failure"))
    );
    assert_eq!(*events.lock().unwrap(), ["pre"]);
    assert!(
        tokio::time::timeout(Duration::from_millis(50), listener.accept())
            .await
            .is_err()
    );
}

struct ReplacingObserver;

impl OcrObserver for ReplacingObserver {
    type Error = &'static str;

    async fn pre_call(&mut self, _input: &OcrPreCall) -> Result<CallbackDecision, Self::Error> {
        Ok(CallbackDecision::Replace {
            payload: json!({
                "model": "mistral-ocr-4-1",
                "document": {
                    "type": "document_url",
                    "document_url": "https://example.com/replaced.pdf"
                }
            }),
        })
    }

    async fn post_call(&mut self, _input: &OcrPostCall) -> Result<CallbackDecision, Self::Error> {
        Ok(CallbackDecision::Replace {
            payload: json!({"pages":[{"index":0,"markdown":"replaced"}]}),
        })
    }
}

#[tokio::test]
async fn provider_callback_replacement_reaches_wire_and_response_parser() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_http_request(&mut socket).await;
        assert!(request.contains("https://example.com/replaced.pdf"));
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        let body = r#"{"pages":[{"index":0,"markdown":"original"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });
    let result = ocr_with_observer(observer_request(&url), &mut ReplacingObserver)
        .await
        .unwrap();
    server.await.unwrap();
    assert_eq!(result["pages"][0]["markdown"], "replaced");
}

#[tokio::test]
async fn reducto_upload_and_submission_are_distinct_authorized_operations() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut upload, _) = listener.accept().await.unwrap();
        let upload_request = read_http_request(&mut upload).await;
        assert!(upload_request.starts_with("POST /upload "));
        assert!(upload_request.contains("%PDF-1.4"));
        assert!(
            upload_request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        let upload_body = r#"{"file_id":"reducto://uploaded.pdf"}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{upload_body}",
            upload_body.len()
        );
        upload.write_all(response.as_bytes()).await.unwrap();

        let (mut submission, _) = listener.accept().await.unwrap();
        let submission_request = read_http_request(&mut submission).await;
        assert!(submission_request.starts_with("POST /parse "));
        assert!(submission_request.contains("reducto://uploaded.pdf"));
        assert!(
            submission_request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        let submission_body = r#"{"result":{"chunks":[]},"usage":{"num_pages":1}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{submission_body}",
            submission_body.len()
        );
        submission.write_all(response.as_bytes()).await.unwrap();
    });
    let result = ocr(OcrRequest {
        model: "reducto/parse-v3",
        document: json!({
            "type": "document_url",
            "document_url": "data:application/pdf;base64,JVBERi0xLjQ="
        }),
        api_key: Some("test-key"),
        api_base: Some(&api_base),
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: Map::new(),
        timeout: Some(Duration::from_secs(2)),
        litellm_call_id: Some("reducto-operation-test"),
        context: Default::default(),
    })
    .await
    .unwrap();
    server.await.unwrap();
    assert_eq!(result["usage_info"]["pages_processed"], 1);
}

#[tokio::test]
async fn invalid_ocr_preparation_does_not_call_observers_or_provider() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut observer = ProviderObserver {
        events: Arc::clone(&events),
        raw_response: None,
        reject: false,
    };
    let request = OcrRequest {
        document: json!(42),
        ..observer_request(&url)
    };
    assert!(ocr_with_observer(request, &mut observer).await.is_err());
    assert!(events.lock().unwrap().is_empty());
    assert!(
        tokio::time::timeout(Duration::from_millis(50), listener.accept())
            .await
            .is_err()
    );
}

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

async fn read_http_request(socket: &mut TcpStream) -> String {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 1024];
    let header_end = loop {
        let n = socket.read(&mut buffer).await.expect("reads request");
        if n == 0 {
            break request.len();
        }
        request.extend_from_slice(&buffer[..n]);
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
        let n = socket.read(&mut buffer).await.expect("reads body");
        if n == 0 {
            break;
        }
        request.extend_from_slice(&buffer[..n]);
    }
    String::from_utf8(request).expect("request is utf8")
}

#[test]
fn ocr_dispatch_supports_migrated_providers() {
    assert!(ocr_provider_config("mistral", "mistral-ocr-latest").is_some());
    assert!(
        ocr_provider_config("azure_ai", "pixtral-12b-2409")
            .expect("azure ai config resolves")
            .requires_data_uri_document()
    );
    assert_eq!(
        ocr_provider_config("azure_ai", "doc-intelligence/prebuilt-read")
            .expect("document intelligence config resolves")
            .response_handling(),
        OcrResponseHandling::AzureDocumentIntelligencePoll
    );
    assert!(
        ocr_provider_config("vertex_ai", "deepseek-ocr-maas")
            .expect("vertex deepseek config resolves")
            .supported_ocr_params()
            .contains(&"temperature")
    );
    assert!(ocr_provider_config("openai", "gpt-4o").is_none());
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
        custom_llm_provider: Some("mistral"),
        extra_headers: Some(headers),
        optional_params: Map::new(),
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: None,
        context: Default::default(),
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
        model: "doc-intelligence/prebuilt-read",
        document: json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        }),
        api_key: Some("di-key"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("azure_ai"),
        extra_headers: None,
        optional_params: Map::new(),
        timeout: Some(Duration::from_secs(5)),
        litellm_call_id: None,
        context: Default::default(),
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
