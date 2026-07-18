use std::sync::{Arc, Mutex};
use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::OcrResponseHandling;
use serde_json::{json, Map, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use super::common_utils::{has_header, ocr_provider_config, string_headers, truncate_error_body};
use super::{ocr, OcrRequest};
use crate::integrations::custom_guardrail::{
    CustomGuardrail, GuardrailContext, GuardrailDecision, GuardrailError, GuardrailEventHook,
    GuardrailFuture, GuardrailRequest,
};
use crate::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogFuture, ModelCallDetails,
};
use crate::integrations::types::RequestMetadata;

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

#[derive(Clone, Debug, PartialEq)]
struct RecordedLogEvent {
    hook: &'static str,
    model: String,
    call_type: String,
    user_id: Option<String>,
    response_object: Option<String>,
    error_message: Option<String>,
    error_kind: Option<String>,
}

#[derive(Default)]
struct RecordingOcrLogger {
    events: Mutex<Vec<RecordedLogEvent>>,
}

impl RecordingOcrLogger {
    fn events(&self) -> Vec<RecordedLogEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl CustomLogger for RecordingOcrLogger {
    fn async_log_success_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: &'a CallbackValue,
        _timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            self.events.lock().unwrap().push(RecordedLogEvent {
                hook: "async_log_success_event",
                model: model_call_details.model.clone(),
                call_type: model_call_details.call_type.to_string(),
                user_id: model_call_details.metadata.user_api_key_user_id.clone(),
                response_object: Some(response_obj.object.clone()),
                error_message: None,
                error_kind: None,
            });
            Ok(())
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: Option<&'a CallbackValue>,
        _timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            self.events.lock().unwrap().push(RecordedLogEvent {
                hook: "async_log_failure_event",
                model: model_call_details.model.clone(),
                call_type: model_call_details.call_type.to_string(),
                user_id: model_call_details.metadata.user_api_key_user_id.clone(),
                response_object: response_obj.map(|value| value.object.clone()),
                error_message: model_call_details
                    .failure_error
                    .as_ref()
                    .map(|error| error.message.clone()),
                error_kind: model_call_details
                    .failure_error
                    .as_ref()
                    .map(|error| error.kind.clone()),
            });
            Ok(())
        })
    }
}

struct RecordingOcrGuardrail {
    hooks: Vec<GuardrailEventHook>,
    events: Mutex<Vec<&'static str>>,
    block_pre_call: bool,
}

impl RecordingOcrGuardrail {
    fn new(hooks: Vec<GuardrailEventHook>) -> Self {
        Self {
            hooks,
            events: Mutex::new(Vec::new()),
            block_pre_call: false,
        }
    }

    fn blocking_pre_call() -> Self {
        Self {
            hooks: vec![GuardrailEventHook::PreCall],
            events: Mutex::new(Vec::new()),
            block_pre_call: true,
        }
    }

    fn events(&self) -> Vec<&'static str> {
        self.events.lock().unwrap().clone()
    }
}

impl CustomGuardrail for RecordingOcrGuardrail {
    fn guardrail_name(&self) -> &str {
        "recording-ocr-guardrail"
    }

    fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
        &self.hooks
    }

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a GuardrailContext,
        mut request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        Box::pin(async move {
            self.events.lock().unwrap().push("async_pre_call_hook");
            if self.block_pre_call {
                return Ok(GuardrailDecision::Block(GuardrailError::blocked(
                    "blocked before provider",
                )));
            }
            request.data["document"]["guarded_pre"] = json!(true);
            Ok(GuardrailDecision::Mask(request))
        })
    }

    fn async_moderation_hook<'a>(
        &'a self,
        _context: &'a GuardrailContext,
        mut request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        Box::pin(async move {
            self.events.lock().unwrap().push("async_moderation_hook");
            request.data["body"]["guarded_during"] = json!(true);
            Ok(GuardrailDecision::Mask(request))
        })
    }
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
async fn ocr_lifecycle_runs_pre_during_and_success_hooks() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let addr = listener.local_addr().expect("listener has local addr");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts one request");
        let request = read_http_request(&mut socket).await;
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

    let logger = Arc::new(RecordingOcrLogger::default());
    let guardrail = Arc::new(RecordingOcrGuardrail::new(vec![
        GuardrailEventHook::PreCall,
        GuardrailEventHook::DuringCall,
    ]));
    let response = ocr(OcrRequest {
        model: "mistral-ocr-latest",
        document: json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        }),
        api_key: Some("sk-test"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("mistral"),
        extra_headers: None,
        optional_params: Map::new(),
        timeout: Some(Duration::from_secs(5)),
        callbacks: vec![logger.clone()],
        guardrails: vec![guardrail.clone()],
        request_metadata: RequestMetadata {
            user_api_key_user_id: Some("user-1".to_string()),
            ..Default::default()
        },
        litellm_call_id: Some("ocr-call-1"),
    })
    .await
    .expect("ocr request succeeds");

    assert_eq!(response["pages"][0]["markdown"], "ok");
    assert_eq!(
        guardrail.events(),
        vec!["async_pre_call_hook", "async_moderation_hook"]
    );
    assert_eq!(
        logger.events(),
        vec![RecordedLogEvent {
            hook: "async_log_success_event",
            model: "mistral-ocr-latest".to_string(),
            call_type: "ocr".to_string(),
            user_id: Some("user-1".to_string()),
            response_object: Some("ocr".to_string()),
            error_message: None,
            error_kind: None,
        }]
    );

    let request = server.await.expect("server task completes");
    assert!(request.contains(r#""guarded_pre":true"#), "{request}");
    assert!(request.contains(r#""guarded_during":true"#), "{request}");
}

#[tokio::test]
async fn ocr_lifecycle_runs_failure_hook_on_provider_error() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let addr = listener.local_addr().expect("listener has local addr");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts one request");
        let _request = read_http_request(&mut socket).await;
        let response_body = "provider failed";
        let response = format!(
            "HTTP/1.1 500 Internal Server Error\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("writes response");
    });

    let logger = Arc::new(RecordingOcrLogger::default());
    let err = ocr(OcrRequest {
        model: "mistral-ocr-latest",
        document: json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        }),
        api_key: Some("sk-test"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("mistral"),
        extra_headers: None,
        optional_params: Map::new(),
        timeout: Some(Duration::from_secs(5)),
        callbacks: vec![logger.clone()],
        guardrails: Vec::new(),
        request_metadata: RequestMetadata::default(),
        litellm_call_id: Some("ocr-call-2"),
    })
    .await
    .expect_err("provider error propagates");

    assert!(matches!(err, CoreError::Http { status: 500, .. }));
    server.await.expect("server task completes");
    assert_eq!(
        logger.events(),
        vec![RecordedLogEvent {
            hook: "async_log_failure_event",
            model: "mistral-ocr-latest".to_string(),
            call_type: "ocr".to_string(),
            user_id: None,
            response_object: Some("error".to_string()),
            error_message: Some("OCR request failed with status 500".to_string()),
            error_kind: Some("HttpError".to_string()),
        }]
    );
}

#[tokio::test]
async fn ocr_lifecycle_pre_call_block_skips_provider_socket() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener binds");
    let addr = listener.local_addr().expect("listener has local addr");
    let logger = Arc::new(RecordingOcrLogger::default());
    let guardrail = Arc::new(RecordingOcrGuardrail::blocking_pre_call());

    let err = ocr(OcrRequest {
        model: "mistral-ocr-latest",
        document: json!({
            "type": "document_url",
            "document_url": "https://example.com/doc.pdf"
        }),
        api_key: Some("sk-test"),
        api_base: Some(&format!("http://{addr}")),
        custom_llm_provider: Some("mistral"),
        extra_headers: None,
        optional_params: Map::new(),
        timeout: Some(Duration::from_millis(100)),
        callbacks: vec![logger.clone()],
        guardrails: vec![guardrail.clone()],
        request_metadata: RequestMetadata::default(),
        litellm_call_id: Some("ocr-call-3"),
    })
    .await
    .expect_err("guardrail blocks request");

    assert!(matches!(err, CoreError::InvalidRequest(_)));
    assert_eq!(guardrail.events(), vec!["async_pre_call_hook"]);
    assert_eq!(
        logger.events(),
        vec![RecordedLogEvent {
            hook: "async_log_failure_event",
            model: "mistral-ocr-latest".to_string(),
            call_type: "ocr".to_string(),
            user_id: None,
            response_object: Some("error".to_string()),
            error_message: Some("Invalid OCR request".to_string()),
            error_kind: Some("InvalidRequest".to_string()),
        }]
    );
    let accepted = tokio::time::timeout(Duration::from_millis(100), listener.accept()).await;
    assert!(accepted.is_err(), "provider socket should not be touched");
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
        callbacks: Vec::new(),
        guardrails: Vec::new(),
        request_metadata: RequestMetadata::default(),
        litellm_call_id: None,
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
        callbacks: Vec::new(),
        guardrails: Vec::new(),
        request_metadata: RequestMetadata::default(),
        litellm_call_id: None,
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
