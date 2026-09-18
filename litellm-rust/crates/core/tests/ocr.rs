use std::sync::{Arc, Mutex};

use litellm_callbacks::{
    event::{CallEvent, WireRequest},
    host::{Host, HostOp, HostResult},
    machine::{HostFailure, Machine, MachineStep},
};
use litellm_http::{HttpClientConfig, HttpClientPool, HttpSettings, Verify};
use litellm_llms::{
    base_llm::ocr::{
        error::Error as OcrError,
        transformation::{LiteLLMOcrResponse, OCR_RESPONSE_MAX_BYTES, OcrTransportConfig},
    },
    custom_httpx::llm_http_handler::OcrClient,
};
use rstest::rstest;
use serde_json::{Value, json};

use super::{
    test_support::{
        MockResponse, mock_server, ocr_client, perform_ocr, perform_ocr_with, wire_request,
    },
    wire::{OcrWireRequest, decode_request},
};
use crate::ocr::route::{LocalOcrHost, OcrOp, OcrOpResult, ocr_machine};

#[rstest]
#[case::mistral("mistral/model", json!({}))]
#[case::vertex("vertex_ai/mistral-ocr-latest", json!({"vertex_project":"test-project", "vertex_location":"us-central1"}))]
#[tokio::test]
async fn ocr_contract_upstream_error_preserves_status_body_and_headers(
    #[case] model: &str,
    #[case] options: Value,
) {
    let payload = json!({"message": format!("{} END-OF-PROVIDER-BODY", "x".repeat(4096))});
    let expected_body = serde_json::to_string(&payload).unwrap();
    let (base, seen, server) = mock_server(vec![MockResponse {
        status: 422,
        headers: vec![
            ("Retry-After", "17".into()),
            ("X-Request-ID", "request-123".into()),
            ("X-Future-Header", "retained".into()),
        ],
        body: payload,
    }])
    .await;
    let error = perform_ocr(wire_request(model, &base, options))
        .await
        .unwrap_err();
    server.await.unwrap();
    assert_eq!(seen.lock().unwrap().len(), 1);
    let OcrError::Provider {
        status,
        body,
        headers,
    } = error
    else {
        panic!("expected provider error, got {error:?}");
    };
    assert_eq!(status, 422);
    for (name, value) in [
        ("retry-after", "17"),
        ("x-request-id", "request-123"),
        ("x-future-header", "retained"),
    ] {
        assert!(
            headers
                .iter()
                .any(|(key, actual)| key.eq_ignore_ascii_case(name) && actual == value)
        );
    }
    assert_eq!(
        body.len(),
        expected_body.len(),
        "provider error body was truncated"
    );
    assert_eq!(body, expected_body);
}

#[test]
fn request_boundary_selects_mistral_and_rejects_unknown_providers() {
    let request = OcrWireRequest {
        model: "mistral/model".into(),
        document: json!({"type":"document_url","document_url":"https://example.com/doc.pdf"}),
        api_key: Some("key".into()),
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: json!({"extract_header":true,"unknown":42})
            .as_object()
            .unwrap()
            .clone(),
        input_sources: Default::default(),
        timeout_seconds: None,
    };
    assert!(decode_request(request).is_ok());
    assert!(
        decode_request(OcrWireRequest {
            model: "model".into(),
            document: json!({"type":"document_url","document_url":"https://example.com/doc.pdf"}),
            api_key: Some("key".into()),
            api_base: None,
            custom_llm_provider: Some("unknown".into()),
            extra_headers: None,
            optional_params: serde_json::Map::new(),
            input_sources: Default::default(),
            timeout_seconds: None,
        })
        .is_err()
    );
}

#[tokio::test]
async fn facade_executes_direct_mistral_once() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "pages":[{"index":0,"markdown":"hello","custom":"preserved"}],
        "usage_info":{"pages_processed":1}
    }))])
    .await;
    let result = perform_ocr(wire_request(
        "mistral/model",
        &base,
        json!({"pages":"0,2-4","extract_header":true,"unknown":"ignored"}),
    ))
    .await
    .unwrap();
    server.await.unwrap();
    assert_eq!(result.pages[0].markdown, "hello");
    assert_eq!(result.pages[0].extra_fields["custom"], "preserved");
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with("POST /v1/ocr "));
    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key\r\n")
    );
    let body: Value = serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body,
        json!({
            "model":"model",
            "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
            "pages":"0,2-4",
            "extract_header":true,
            "unknown":"ignored"
        })
    );
}

#[tokio::test]
async fn facade_retains_native_response_when_requested() {
    let provider_response = json!({
        "pages":[{"index":0,"markdown":"hello"}],
        "usage_info":{"pages_processed":1},
        "provider_only":"preserved"
    });
    let (base, _, server) = mock_server(vec![MockResponse::json(provider_response.clone())]).await;
    let response = perform_ocr(wire_request(
        "mistral/model",
        &base,
        json!({"req_format":"native"}),
    ))
    .await
    .unwrap();

    server.await.unwrap();
    assert_eq!(
        response.provider_native_response.map(Value::Object),
        Some(provider_response)
    );
}

#[tokio::test]
async fn facade_uses_the_injected_http_pool_configuration() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let settings = HttpSettings {
        user_agent: Some("host-owned/1".into()),
        ..HttpSettings::default()
    };
    let config = HttpClientConfig::resolve(&settings, None).unwrap();
    crate::ocr::client::ocr(
        &HttpClientPool::new(),
        &config,
        wire_request("mistral/model", &base, json!({})),
    )
    .await
    .unwrap();
    server.await.unwrap();
    assert!(seen.lock().unwrap()[0].contains("user-agent: host-owned/1"));
}

#[tokio::test]
async fn unbuildable_http_configuration_fails_before_dispatch() {
    let (base, _seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let config = HttpClientConfig {
        verify: Verify::CaBundle(std::env::temp_dir().join("litellm-ocr-missing-bundle.pem")),
        ..HttpClientConfig::resolve(&HttpSettings::default(), None).unwrap()
    };
    let error = crate::ocr::client::ocr(
        &HttpClientPool::new(),
        &config,
        wire_request("mistral/model", &base, json!({})),
    )
    .await
    .unwrap_err();
    server.abort();
    assert!(matches!(
        error,
        OcrError::Transport(litellm_llms::custom_httpx::transport::Error::Connect(_))
    ));
    assert!(error.to_string().contains("litellm-ocr-missing-bundle.pem"));
}

fn event_name(event: &CallEvent) -> &'static str {
    match event {
        CallEvent::ResponseReceived { .. } => "response",
        CallEvent::Succeeded { .. } => "success",
        CallEvent::Failed { .. } => "failure",
    }
}

fn recording_host(
    request: crate::ocr::types::LiteLLMOcrRequest,
    events: Arc<Mutex<Vec<&'static str>>>,
    block: bool,
) -> LocalOcrHost {
    let before_send_events = events.clone();
    LocalOcrHost::new(request)
        .with_before_send(move |wire, _| {
            before_send_events.lock().unwrap().push("before_send");
            if block {
                return Err(OcrError::InvalidRequest("blocked".into()));
            }
            Ok(wire)
        })
        .with_observer(move |event| events.lock().unwrap().push(event_name(event)))
}

#[tokio::test]
async fn lifecycle_sends_headers_returned_by_the_before_send_operation() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let host = LocalOcrHost::new(wire_request("mistral/model", &base, json!({}))).with_before_send(
        |mut wire, _| {
            wire.headers
                .push(("x-core-callback".into(), "edited".into()));
            Ok(wire)
        },
    );

    perform_ocr_with(host).await.unwrap();
    server.await.unwrap();

    assert!(seen.lock().unwrap()[0].contains("x-core-callback: edited"));
}

#[tokio::test]
async fn before_send_context_names_passthrough_fields_and_secrets() {
    let (base, _, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let observed = Arc::new(Mutex::new(None));
    let captured = observed.clone();
    let host = LocalOcrHost::new(wire_request(
        "mistral/model",
        &base,
        json!({"pages": [0], "req_format": "native"}),
    ))
    .with_before_send(move |wire, context| {
        *captured.lock().unwrap() = Some((wire.clone(), context.clone()));
        Ok(wire)
    });
    perform_ocr_with(host).await.unwrap();
    server.await.unwrap();
    let (wire, context) = observed.lock().unwrap().take().unwrap();
    assert_eq!(context.custom_llm_provider, "mistral");
    assert_eq!(context.model, "model");
    assert_eq!(wire.body["pages"], json!([0]));
    assert!(context.passthrough_fields.contains("pages"));
    assert!(context.passthrough_fields.contains("document"));
    assert!(context.secret_fields.is_empty());
    assert_eq!(context.optional_params["req_format"], "native");

    let (base, _, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let observed = Arc::new(Mutex::new(None));
    let captured = observed.clone();
    let request = wire_request(
        "azure_ai/model",
        &base,
        json!({"client_secret": "shh", "tenant_id": "t"}),
    );
    let request = request.with_document(crate::ocr::types::OcrDocumentInput::Bytes {
        bytes: b"abc".as_slice().into(),
        file_name: None,
        mime_type: Some("application/pdf".into()),
    });
    let host = LocalOcrHost::new(request).with_before_send(move |wire, context| {
        *captured.lock().unwrap() = Some(context.clone());
        Ok(wire)
    });
    perform_ocr_with(host).await.unwrap();
    server.await.unwrap();
    let context = observed.lock().unwrap().take().unwrap();
    assert!(!context.passthrough_fields.contains("document"));
    assert_eq!(context.secret_fields, ["client_secret"]);
}

#[tokio::test]
async fn lifecycle_orders_hooks_and_emits_one_success() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let host = recording_host(
        wire_request("mistral/model", &base, json!({})),
        events.clone(),
        false,
    );
    perform_ocr_with(host).await.unwrap();
    server.await.unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        ["before_send", "response", "success"]
    );
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn lifecycle_blocking_prevents_execution_and_emits_one_failure() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let host = recording_host(
        wire_request("mistral/model", "http://127.0.0.1:1", json!({})),
        events.clone(),
        true,
    );
    let error = perform_ocr_with(host).await.unwrap_err();
    assert!(matches!(error, OcrError::InvalidRequest(message) if message == "blocked"));
    assert_eq!(*events.lock().unwrap(), ["before_send", "failure"]);
}

#[tokio::test]
async fn upstream_failure_emits_one_terminal_failure() {
    let (base, seen, server) = mock_server(vec![MockResponse {
        status: 500,
        headers: vec![],
        body: json!({"error":"failed"}),
    }])
    .await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let host = recording_host(
        wire_request("mistral/model", &base, json!({})),
        events.clone(),
        false,
    );
    assert!(perform_ocr_with(host).await.is_err());
    server.await.unwrap();
    assert_eq!(*events.lock().unwrap(), ["before_send", "failure"]);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

/// Drives the machine by hand, answering every op through `host` except `before_send`,
/// which `intercept` answers so a test can fail or cancel exactly there.
async fn drive_until(
    client: OcrClient,
    host: &LocalOcrHost,
    mut intercept: impl FnMut(WireRequest) -> Result<WireRequest, HostFailure<OcrError>>,
) -> (
    Result<LiteLLMOcrResponse, OcrError>,
    Vec<&'static str>,
    crate::ocr::route::OcrMachine,
) {
    let mut machine = ocr_machine(client);
    let mut result = None;
    let mut ops = Vec::new();
    let outcome = loop {
        let op = match machine.resume(result.take()).await {
            Ok(MachineStep::Host(op)) => op,
            Ok(MachineStep::Complete(response)) => break Ok(response),
            Err(error) => break Err(error),
        };
        let answer = match op {
            HostOp::Route(op) => {
                ops.push(match op {
                    OcrOp::ProjectRequest => "ProjectRequest",
                    OcrOp::ReadDocument => "ReadDocument",
                    OcrOp::AcquireAzureAdToken => "AcquireAzureAdToken",
                });
                host.route(op)
                    .await
                    .map(HostResult::Route)
                    .map_err(HostFailure::Error)
            }
            HostOp::BeforeSend { wire, .. } => {
                ops.push("BeforeSend");
                intercept(*wire).map(|wire| HostResult::BeforeSend(Box::new(wire)))
            }
            HostOp::Emit(event) => {
                ops.push(event_name(&event));
                host.emit(&event)
                    .await
                    .map(|()| HostResult::Emitted)
                    .map_err(HostFailure::Error)
            }
        };
        match answer {
            Ok(answer) => result = Some(answer),
            Err(failure) => break machine.interrupt(failure).await,
        }
    };
    (outcome, ops, machine)
}

#[tokio::test]
async fn failed_before_send_does_not_replay_or_reach_transport() {
    let host = LocalOcrHost::new(wire_request(
        "mistral/model",
        "http://127.0.0.1:1",
        json!({}),
    ));
    let (outcome, ops, mut machine) = drive_until(ocr_client(), &host, |_| {
        Err(HostFailure::Error(OcrError::InvalidRequest(
            "before_send failed".into(),
        )))
    })
    .await;
    assert!(
        matches!(outcome, Err(OcrError::InvalidRequest(message)) if message == "before_send failed")
    );
    assert_eq!(ops, ["ProjectRequest", "BeforeSend"]);
    assert!(machine.resume(None).await.is_err());
}

#[tokio::test]
async fn invalid_provider_response_emits_response_received_before_normalization_failure() {
    let (base, seen, server) =
        mock_server(vec![MockResponse::json(json!({"pages":"invalid"}))]).await;
    let responses_received = Arc::new(Mutex::new(Vec::new()));
    let observed = responses_received.clone();
    let host = LocalOcrHost::new(wire_request("mistral/model", &base, json!({}))).with_observer(
        move |event| {
            if let CallEvent::ResponseReceived { raw } = event {
                observed.lock().unwrap().push(raw.body.clone());
            }
        },
    );
    let error = perform_ocr_with(host).await.unwrap_err();
    server.await.unwrap();
    assert!(matches!(error, OcrError::ResponseField { .. }));
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(
        *responses_received.lock().unwrap(),
        [r#"{"pages":"invalid"}"#]
    );
}

#[tokio::test]
async fn direct_native_host_drives_the_same_state_machine() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "pages":[{"index":0,"markdown":"native"}]
    }))])
    .await;
    let host = LocalOcrHost::new(wire_request("mistral/model", &base, json!({})));
    let (outcome, ops, mut machine) = drive_until(ocr_client(), &host, Ok).await;
    server.await.unwrap();
    assert_eq!(outcome.unwrap().pages[0].markdown, "native");
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(ops, ["ProjectRequest", "BeforeSend", "response"]);
    assert!(matches!(
        machine.resume(None).await,
        Err(OcrError::InvalidRequest(_))
    ));
}

async fn drive_native_file_call(
    request: crate::ocr::types::LiteLLMOcrRequest<crate::ocr::types::OcrDocumentInput>,
    content: Result<crate::ocr::types::OcrFileContent, OcrError>,
) -> (Result<LiteLLMOcrResponse, OcrError>, usize) {
    let reads = Arc::new(Mutex::new(0));
    let counted = reads.clone();
    let content = Mutex::new(Some(content));
    let host = LocalOcrHost::new(request).with_reader(move || {
        *counted.lock().unwrap() += 1;
        content.lock().unwrap().take().unwrap()
    });
    let outcome = perform_ocr_with(host).await;
    let reads = *reads.lock().unwrap();
    (outcome, reads)
}

#[tokio::test]
async fn host_reader_documents_are_read_once_at_the_core_selected_point_and_encoded() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "pages":[{"index":0,"markdown":"file"}]
    }))])
    .await;
    let request = wire_request("mistral/model", &base, json!({})).with_document(
        crate::ocr::types::OcrDocumentInput::HostReader {
            mime_type: Some("application/pdf".into()),
        },
    );
    let (response, reads) = drive_native_file_call(
        request,
        Ok(crate::ocr::types::OcrFileContent {
            bytes: b"abc".as_slice().into(),
            file_name: Some("scan.png".into()),
        }),
    )
    .await;
    server.await.unwrap();
    assert_eq!(response.unwrap().pages[0].markdown, "file");
    assert_eq!(reads, 1);
    assert!(seen.lock().unwrap()[0].contains("data:application/pdf;base64,YWJj"));
}

#[tokio::test]
async fn host_reader_failures_and_empty_files_fail_before_the_provider_is_called() {
    let (base, seen, _server) = mock_server(vec![]).await;
    let request = wire_request("mistral/model", &base, json!({}));
    let failure = OcrError::InvalidRequest("reader exploded".into());
    let (response, reads) = drive_native_file_call(
        request.with_document(crate::ocr::types::OcrDocumentInput::HostReader { mime_type: None }),
        Err(failure.clone()),
    )
    .await;
    assert!(
        matches!(response.unwrap_err(), OcrError::InvalidRequest(message) if message == "reader exploded")
    );
    assert_eq!(reads, 1);

    let request = wire_request("mistral/model", &base, json!({}));
    let (response, _) = drive_native_file_call(
        request.with_document(crate::ocr::types::OcrDocumentInput::HostReader { mime_type: None }),
        Ok(crate::ocr::types::OcrFileContent {
            bytes: Default::default(),
            file_name: None,
        }),
    )
    .await;
    assert!(matches!(response.unwrap_err(), OcrError::EmptyFile));
    assert!(seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn path_documents_are_read_by_core_without_a_host_operation() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "pages":[{"index":0,"markdown":"path"}]
    }))])
    .await;
    let dir = std::env::temp_dir().join(format!("litellm-ocr-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("scan.png");
    std::fs::write(&path, b"abc").unwrap();
    let request = wire_request("mistral/model", &base, json!({})).with_document(
        crate::ocr::types::OcrDocumentInput::Path {
            path: path.clone(),
            mime_type: None,
        },
    );
    let (response, reads) =
        drive_native_file_call(request, Err(OcrError::InvalidRequest("unused".into()))).await;
    server.await.unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(response.unwrap().pages[0].markdown, "path");
    assert_eq!(reads, 0);
    assert!(seen.lock().unwrap()[0].contains("data:image/png;base64,YWJj"));

    let (base, seen, _server) = mock_server(vec![]).await;
    let request = wire_request("mistral/model", &base, json!({}));
    let (response, _) = drive_native_file_call(
        request.with_document(crate::ocr::types::OcrDocumentInput::Path {
            path: path.clone(),
            mime_type: None,
        }),
        Err(OcrError::InvalidRequest("unused".into())),
    )
    .await;
    assert!(matches!(
        response.unwrap_err(),
        OcrError::FileRead { path: failed, source } if failed == path && source.kind() == std::io::ErrorKind::NotFound
    ));
    assert!(seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn cancellation_at_before_send_prevents_execution_and_further_resumption() {
    let host = LocalOcrHost::new(wire_request(
        "mistral/model",
        "http://127.0.0.1:1",
        json!({}),
    ));
    let (outcome, ops, mut machine) = drive_until(ocr_client(), &host, |_| {
        Err(HostFailure::Cancelled(OcrError::InvalidRequest(
            "cancelled".into(),
        )))
    })
    .await;
    assert!(matches!(outcome, Err(OcrError::InvalidRequest(message)) if message == "cancelled"));
    assert_eq!(ops, ["ProjectRequest", "BeforeSend"]);
    assert!(machine.resume(Some(HostResult::Emitted)).await.is_err());
}

#[tokio::test]
async fn missing_host_result_preserves_pending_operation() {
    let request = wire_request("mistral/model", "http://127.0.0.1:1", json!({}));
    let mut machine = ocr_machine(ocr_client());
    assert!(matches!(
        machine.resume(None).await.unwrap(),
        MachineStep::Host(HostOp::Route(OcrOp::ProjectRequest))
    ));
    assert!(machine.resume(None).await.is_err());
    assert!(matches!(
        machine
            .resume(Some(HostResult::Route(OcrOpResult::Request {
                request: Box::new(request),
                caller_token: false,
            })))
            .await
            .unwrap(),
        MachineStep::Host(HostOp::BeforeSend { .. })
    ));
}

async fn read_bounded_response(response: Vec<u8>, limit: usize) -> Result<bytes::Bytes, OcrError> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        assert!(socket.read(&mut request).await.unwrap() > 0);
        socket.write_all(&response).await.unwrap();
        std::future::pending::<()>().await;
    });
    let response = reqwest::Client::new()
        .get(format!("http://{address}"))
        .send()
        .await
        .unwrap();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        litellm_llms::custom_httpx::llm_http_handler::read_response_bytes(response, limit),
    )
    .await;
    server.abort();
    let _ = server.await;
    result.expect("bounded reads must finish without waiting for the rest of an oversized body")
}

#[tokio::test]
async fn response_limit_accepts_exact_size_and_rejects_declared_and_chunked_overflow() {
    use litellm_llms::base_llm::ocr::error::Error;

    for response in [
        "HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nabcdefgh",
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n4\r\nefgh\r\n0\r\n\r\n",
    ] {
        assert_eq!(
            read_bounded_response(response.as_bytes().to_vec(), 8)
                .await
                .unwrap(),
            "abcdefgh"
        );
    }
    for response in [
        "HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n",
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n5\r\nefghi\r\n",
    ] {
        assert!(matches!(
            read_bounded_response(response.as_bytes().to_vec(), 8).await,
            Err(Error::TooLarge { limit: 8 })
        ));
    }
}

#[rstest]
#[case::declared("Content-Length: 1000000")]
#[case::chunked("Transfer-Encoding: chunked")]
#[tokio::test]
async fn oversized_error_retains_http_status_and_bounded_diagnostics_without_draining(
    #[case] headers: &str,
) {
    let prefix = "x".repeat(4096);
    let body = if headers.starts_with("Transfer") {
        format!("{:x}\r\n{prefix}\r\n", prefix.len())
    } else {
        prefix.clone()
    };
    let response = format!("HTTP/1.1 429 Too Many Requests\r\n{headers}\r\n\r\n{body}");
    let error = read_bounded_response(response.into_bytes(), prefix.len())
        .await
        .unwrap_err();
    match error {
        OcrError::Transport(litellm_llms::custom_httpx::transport::Error::Http {
            status,
            body,
        }) => {
            assert_eq!(status, 429);
            assert_eq!(body, prefix);
        }
        error => panic!("unexpected error: {error}"),
    }
}

#[test]
fn response_limit_is_validated_and_not_forwarded_to_the_provider() {
    let request = wire_request(
        "mistral/model",
        "http://localhost",
        json!({"max_response_bytes": 123}),
    );
    assert_eq!(request.transport.max_response_bytes, 123);
    assert!(!request.optional_params.contains_key("max_response_bytes"));
    for value in [
        json!(0),
        json!(-1),
        json!(true),
        json!("123"),
        json!(1.5),
        json!(OCR_RESPONSE_MAX_BYTES + 1),
        Value::Null,
    ] {
        let wire = serde_json::from_value(json!({
            "model": "mistral/model", "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
            "optional_params": {"max_response_bytes": value}
        })).unwrap();
        let Err(error) = decode_request(wire) else {
            panic!("invalid response limit accepted")
        };
        assert!(error.to_string().contains("max_response_bytes"));
    }
}

#[derive(Debug)]
struct PendingToken {
    entered: Arc<tokio::sync::Notify>,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}

struct TokenFutureDrop(Arc<std::sync::atomic::AtomicBool>);

impl Drop for TokenFutureDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

impl litellm_auth::TokenProvider for PendingToken {
    fn acquire(&self) -> litellm_auth::TokenFuture<'_> {
        Box::pin(async move {
            let _guard = TokenFutureDrop(self.dropped.clone());
            self.entered.notify_one();
            std::future::pending().await
        })
    }
}

#[tokio::test]
async fn interrupt_drops_provider_captures_before_returning() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let entered = Arc::new(tokio::sync::Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let request = wire_request("azure_ai/mistral-ocr", "https://example.invalid", json!({}));
    let request = crate::ocr::types::LiteLLMOcrRequest {
        transport: OcrTransportConfig {
            extra_headers: vec![("authorization".into(), "Bearer test-key".into())],
            ..request.transport
        },
        azure_ad_token_provider: Some(litellm_auth::TokenProviderHandle::new(Arc::new(
            PendingToken {
                entered: entered.clone(),
                dropped: dropped.clone(),
            },
        ))),
        ..request
    };
    let host = LocalOcrHost::new(request);
    let mut machine = ocr_machine(ocr_client());
    let mut result = None;
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            tokio::select! {
                _ = entered.notified() => break,
                step = machine.resume(result.take()) => {
                    result = Some(match step.unwrap() {
                        MachineStep::Host(HostOp::Route(op)) => HostResult::Route(host.route(op).await.unwrap()),
                        MachineStep::Host(HostOp::BeforeSend { wire, .. }) => {
                            HostResult::BeforeSend(wire)
                        }
                        MachineStep::Host(HostOp::Emit(_)) => HostResult::Emitted,
                        MachineStep::Complete(_) => panic!("pending provider completed"),
                    });
                }
            }
        }
    })
    .await
    .unwrap();
    assert!(!dropped.load(Ordering::SeqCst));
    let selected = OcrError::InvalidRequest("cancelled".into());
    let acknowledgement = machine.interrupt(HostFailure::Cancelled(selected.clone()));
    assert!(
        dropped.load(Ordering::SeqCst),
        "interrupt returned while provider captures were still alive"
    );
    assert!(
        matches!(acknowledgement.await, Err(OcrError::InvalidRequest(message)) if message == "cancelled")
    );
}

struct CallerTokenHost {
    request: Mutex<Option<crate::ocr::types::LiteLLMOcrRequest>>,
    trace: Mutex<Vec<String>>,
}

impl Host<crate::ocr::route::Ocr> for CallerTokenHost {
    async fn route(&self, op: OcrOp) -> Result<OcrOpResult, OcrError> {
        match op {
            OcrOp::ProjectRequest => {
                self.trace.lock().unwrap().push("project".into());
                Ok(OcrOpResult::Request {
                    request: Box::new(self.request.lock().unwrap().take().unwrap()),
                    caller_token: true,
                })
            }
            OcrOp::AcquireAzureAdToken => {
                self.trace.lock().unwrap().push("token".into());
                Ok(OcrOpResult::AzureAdToken(
                    litellm_auth::ResolvedCredential::Static(litellm_auth::SecretValue::new(
                        "caller-token",
                    )),
                ))
            }
            OcrOp::ReadDocument => Err(OcrError::InvalidRequest("no reader".into())),
        }
    }

    async fn before_send(
        &self,
        wire: WireRequest,
        _: &litellm_callbacks::event::RequestContext,
    ) -> Result<WireRequest, OcrError> {
        let is_authorization = |name: &str| name.eq_ignore_ascii_case("authorization");
        let authorization = wire
            .headers
            .iter()
            .find(|(name, _)| is_authorization(name))
            .map(|(_, value)| value.clone())
            .unwrap_or_default();
        self.trace
            .lock()
            .unwrap()
            .push(format!("before_send:{authorization}"));
        let headers = wire
            .headers
            .into_iter()
            .map(|(name, value)| match is_authorization(&name) {
                true => (name, "Bearer edited".to_string()),
                false => (name, value),
            })
            .collect();
        Ok(WireRequest { headers, ..wire })
    }
}

#[tokio::test]
async fn the_callers_azure_token_is_acquired_before_before_send_which_can_still_replace_it() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let mut request = wire_request("azure_ai/model", &base, json!({}));
    request.credentials.api_key = None;
    let host = CallerTokenHost {
        request: Mutex::new(Some(request)),
        trace: Mutex::new(Vec::new()),
    };

    litellm_callbacks::run::run(ocr_machine(ocr_client()), &host)
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(
        *host.trace.lock().unwrap(),
        ["project", "token", "before_send:Bearer caller-token"]
    );
    assert!(
        seen.lock().unwrap()[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer edited\r\n")
    );
}

#[tokio::test]
async fn interrupting_an_in_flight_provider_request_closes_its_connection() {
    use tokio::io::AsyncReadExt;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let received = Arc::new(tokio::sync::Notify::new());
    let server_received = received.clone();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0u8; 4096];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.unwrap();
            request.extend_from_slice(&buffer[..read]);
        }
        server_received.notify_one();
        loop {
            if socket.read(&mut buffer).await.unwrap() == 0 {
                break;
            }
        }
    });
    let host = LocalOcrHost::new(wire_request("mistral/model", &base, json!({})));
    let mut machine = ocr_machine(ocr_client());
    let mut result = None;
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            tokio::select! {
                _ = received.notified() => break,
                step = machine.resume(result.take()) => {
                    result = Some(match step.unwrap() {
                        MachineStep::Host(HostOp::Route(op)) => HostResult::Route(host.route(op).await.unwrap()),
                        MachineStep::Host(HostOp::BeforeSend { wire, .. }) => HostResult::BeforeSend(wire),
                        MachineStep::Host(HostOp::Emit(_)) => HostResult::Emitted,
                        MachineStep::Complete(_) => panic!("the stalled provider completed"),
                    });
                }
            }
        }
    })
    .await
    .unwrap();

    let cancelled = OcrError::InvalidRequest("cancelled".into());
    assert!(
        machine
            .interrupt(HostFailure::Cancelled(cancelled))
            .await
            .is_err()
    );
    tokio::time::timeout(std::time::Duration::from_secs(1), server)
        .await
        .expect("the provider connection stayed open after the interrupt")
        .unwrap();
}
