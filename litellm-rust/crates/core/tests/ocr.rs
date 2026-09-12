use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::OcrClient;
use super::hooks::{
    OcrDuringCallRequest, OcrHookFuture, OcrHooks, OcrLogFuture, OcrPostCallRequest,
    OcrPreCallRequest,
};
use super::test_support::{MockResponse, mock_server, perform_ocr, wire_request};
use super::wire::{OcrWireRequest, decode_request};
use super::{
    NativeOutcome, NoopOcrHost, OcrAdmission, OcrCall, OcrCallStep, OcrDecline, OcrHost,
    OcrHostOperation, OcrHostResult,
};
use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};

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
        json!({"extract_header":true,"unknown":"ignored"}),
    ))
    .await
    .unwrap();
    server.await.unwrap();
    assert_eq!(result.pages[0]["markdown"], "hello");
    assert_eq!(result.pages[0]["custom"], "preserved");
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
            "extract_header":true
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
    assert_eq!(response.provider_native_response, Some(provider_response));
}

#[tokio::test]
async fn facade_uses_the_injected_http_client() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let mut default_headers = reqwest::header::HeaderMap::new();
    default_headers.insert(
        "x-transport-owner",
        reqwest::header::HeaderValue::from_static("host"),
    );
    let provider_http = reqwest::Client::builder()
        .default_headers(default_headers)
        .build()
        .unwrap();
    OcrClient::new(provider_http)
        .unwrap()
        .perform(wire_request("mistral/model", &base, json!({})))
        .await
        .unwrap();
    server.await.unwrap();
    assert!(seen.lock().unwrap()[0].contains("x-transport-owner: host"));
}

struct RecordingHooks {
    events: Arc<Mutex<Vec<&'static str>>>,
    block: bool,
}

impl OcrHooks for RecordingHooks {
    fn has_guardrails(&self) -> bool {
        true
    }

    fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
        Box::pin(async move {
            self.events.lock().unwrap().push("pre");
            if self.block {
                return Err(crate::Error::InvalidRequest("blocked".into()));
            }
            Ok(request)
        })
    }

    fn during_call(
        &self,
        request: super::hooks::OcrDuringCallRequest,
    ) -> OcrHookFuture<'_, super::hooks::OcrDuringCallRequest> {
        Box::pin(async move {
            self.events.lock().unwrap().push("during");
            Ok(request)
        })
    }

    fn post_call(&self, request: OcrPostCallRequest) -> OcrHookFuture<'_, OcrPostCallRequest> {
        Box::pin(async move {
            self.events.lock().unwrap().push("post");
            Ok(request)
        })
    }

    fn success<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _response: &'a super::LiteLLMOcrResponse,
        _timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async move {
            self.events.lock().unwrap().push("success");
        })
    }

    fn failure<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _error: &'a crate::Error,
        _timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async move {
            self.events.lock().unwrap().push("failure");
        })
    }
}

struct HeaderEditHooks;

impl OcrHooks for HeaderEditHooks {
    fn has_guardrails(&self) -> bool {
        true
    }

    fn during_call(
        &self,
        mut request: OcrDuringCallRequest,
    ) -> OcrHookFuture<'_, OcrDuringCallRequest> {
        request
            .headers
            .push(("x-core-callback".into(), "edited".into()));
        Box::pin(async move { Ok(request) })
    }
}

#[tokio::test]
async fn lifecycle_sends_headers_returned_by_the_typed_during_call_operation() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(HeaderEditHooks),
        ..wire_request("mistral/model", &base, json!({}))
    };

    perform_ocr(request).await.unwrap();
    server.await.unwrap();

    assert!(seen.lock().unwrap()[0].contains("x-core-callback: edited"));
}

#[tokio::test]
async fn lifecycle_orders_hooks_and_emits_one_success() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let request = wire_request("mistral/model", &base, json!({}));
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(RecordingHooks {
            events: events.clone(),
            block: false,
        }),
        ..request
    };
    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        ["pre", "during", "post", "success"]
    );
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn lifecycle_blocking_prevents_execution_and_emits_one_failure() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let request = wire_request("mistral/model", "http://127.0.0.1:1", json!({}));
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(RecordingHooks {
            events: events.clone(),
            block: true,
        }),
        ..request
    };
    let error = perform_ocr(request).await.unwrap_err();
    assert!(matches!(error, crate::Error::InvalidRequest(_)));
    assert_eq!(*events.lock().unwrap(), ["pre", "failure"]);
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
    let request = wire_request("mistral/model", &base, json!({}));
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(RecordingHooks {
            events: events.clone(),
            block: false,
        }),
        ..request
    };
    assert!(perform_ocr(request).await.is_err());
    server.await.unwrap();
    assert_eq!(*events.lock().unwrap(), ["pre", "during", "failure"]);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

struct AdmissionSpy {
    effects: Arc<Mutex<usize>>,
}

impl OcrHooks for AdmissionSpy {
    fn has_guardrails(&self) -> bool {
        *self.effects.lock().unwrap() += 1;
        true
    }

    fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
        *self.effects.lock().unwrap() += 1;
        Box::pin(async move { Ok(request) })
    }
}

#[test]
fn admission_declines_without_invoking_hooks_or_transport() {
    for (admission, expected) in [
        (
            OcrAdmission {
                provider_workflow: false,
                host_operations: true,
                azure_ad_token_provider: false,
                asynchronous: false,
            },
            OcrDecline::ProviderWorkflow,
        ),
        (
            OcrAdmission {
                provider_workflow: true,
                host_operations: false,
                azure_ad_token_provider: false,
                asynchronous: false,
            },
            OcrDecline::HostOperations,
        ),
    ] {
        let outcome = OcrCall::admit(super::test_support::ocr_client(), admission);
        assert!(matches!(outcome, NativeOutcome::Declined(reason) if reason == expected));
    }
}

#[tokio::test]
async fn fallible_host_phases_do_not_replay_or_reach_transport() {
    for failure_phase in ["pre", "during"] {
        let request = super::LiteLLMOcrRequest {
            hooks: Arc::new(AdmissionSpy {
                effects: Arc::new(Mutex::new(0)),
            }),
            ..wire_request("mistral/model", "http://127.0.0.1:1", json!({}))
        };
        let NativeOutcome::Completed(mut call) =
            OcrCall::admit(super::test_support::ocr_client(), OcrAdmission::all())
        else {
            panic!("supported call declined")
        };
        let mut request = Some(request);
        let mut result = None;
        let mut phases = Vec::new();
        let error = loop {
            match call.resume(result.take()).await {
                Ok(OcrCallStep::Host(operation)) => match operation {
                    OcrHostOperation::Lifecycle(_)
                    | OcrHostOperation::ConstructResponse(_)
                    | OcrHostOperation::MapFailure(_)
                    | OcrHostOperation::Success { .. }
                    | OcrHostOperation::Failure { .. } => {
                        result = Some(OcrHostResult::Lifecycle(Ok(())))
                    }
                    OcrHostOperation::ProjectRequest => {
                        result = Some(OcrHostResult::Request(Ok((
                            Box::new(request.take().unwrap()),
                            false,
                        ))))
                    }
                    OcrHostOperation::AcquireAzureAdToken => {
                        panic!("test request has no token provider")
                    }
                    OcrHostOperation::PreCall(request) => {
                        phases.push("pre");
                        result = Some(OcrHostResult::PreCall(if failure_phase == "pre" {
                            Err(crate::Error::InvalidRequest("pre failed".into()))
                        } else {
                            Ok(request)
                        }));
                    }
                    OcrHostOperation::DuringCall(request) => {
                        phases.push("during");
                        result = Some(OcrHostResult::DuringCall(if failure_phase == "during" {
                            Err(crate::Error::InvalidRequest("during failed".into()))
                        } else {
                            Ok(request)
                        }));
                    }
                    OcrHostOperation::PostCall(_) => panic!("transport should not be reached"),
                },
                Err(error) => break error,
                Ok(OcrCallStep::Complete(_)) => panic!("failed call completed"),
            }
        };
        assert!(matches!(error, crate::Error::InvalidRequest(_)));
        assert_eq!(
            phases
                .iter()
                .filter(|phase| **phase == failure_phase)
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn invalid_provider_response_runs_post_call_before_normalization_failure() {
    let (base, seen, server) =
        mock_server(vec![MockResponse::json(json!({"pages":"invalid"}))]).await;
    let mut request = Some(wire_request("mistral/model", &base, json!({})));
    let NativeOutcome::Completed(mut call) =
        OcrCall::admit(super::test_support::ocr_client(), OcrAdmission::all())
    else {
        panic!("supported call declined")
    };
    let host = NoopOcrHost;
    let mut result = None;
    let mut post_calls = Vec::new();
    let error = loop {
        match call.resume(result.take()).await {
            Ok(OcrCallStep::Host(OcrHostOperation::ProjectRequest)) => {
                result = Some(OcrHostResult::Request(Ok((
                    Box::new(request.take().unwrap()),
                    false,
                ))));
            }
            Ok(OcrCallStep::Host(operation)) => {
                if let OcrHostOperation::PostCall(request) = &operation {
                    post_calls.push(request.original_response.clone());
                }
                result = Some(host.invoke(operation).await);
            }
            Err(error) => break error,
            Ok(OcrCallStep::Complete(_)) => panic!("invalid provider response completed"),
        }
    };
    server.await.unwrap();
    assert!(matches!(error, crate::Error::InvalidResponse(_)));
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(post_calls, [json!(r#"{"pages":"invalid"}"#)]);
}

#[tokio::test]
async fn direct_native_host_drives_the_same_state_machine() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "pages":[{"index":0,"markdown":"native"}]
    }))])
    .await;
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(AdmissionSpy {
            effects: Arc::new(Mutex::new(0)),
        }),
        ..wire_request("mistral/model", &base, json!({}))
    };
    let NativeOutcome::Completed(mut call) = OcrCall::admit(
        super::test_support::ocr_client(),
        OcrAdmission {
            asynchronous: true,
            ..OcrAdmission::all()
        },
    ) else {
        panic!("supported call declined")
    };
    let mut request = Some(request);
    let host = NoopOcrHost;
    let mut result = None;
    let mut operations = Vec::new();
    let response = loop {
        match call.resume(result.take()).await.unwrap() {
            OcrCallStep::Host(operation) => {
                operations.push(match &operation {
                    OcrHostOperation::ProjectRequest => "ProjectRequest".into(),
                    OcrHostOperation::Lifecycle(phase) => format!("{phase:?}"),
                    OcrHostOperation::PreCall(_) => "PreCall".into(),
                    OcrHostOperation::DuringCall(_) => "DuringCall".into(),
                    OcrHostOperation::PostCall(_) => "PostCall".into(),
                    OcrHostOperation::ConstructResponse(_) => "ConstructResponse".into(),
                    OcrHostOperation::Success { response, .. } => {
                        assert_eq!(response.pages[0]["markdown"], "native");
                        "Success".into()
                    }
                    _ => panic!("unexpected OCR operation"),
                });
                result = Some(match operation {
                    OcrHostOperation::ProjectRequest => {
                        OcrHostResult::Request(Ok((Box::new(request.take().unwrap()), false)))
                    }
                    operation => host.invoke(operation).await,
                });
            }
            OcrCallStep::Complete(response) => break response,
        }
    };
    server.await.unwrap();
    assert_eq!(response.pages[0]["markdown"], "native");
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(
        operations,
        [
            "Setup",
            "DeploymentPreCall",
            "Prepare",
            "ProjectRequest",
            "PreCall",
            "DuringCall",
            "PostCall",
            "ConstructResponse",
            "DeploymentPostCall",
            "Finalize",
            "Success",
        ]
    );
    assert!(matches!(
        call.resume(None).await,
        Err(crate::Error::InvalidRequest(_))
    ));
}

#[tokio::test]
async fn public_finalization_failure_never_dispatches_success_or_replays_provider() {
    use crate::call_lifecycle::host::{HostFailure, HostPhase};

    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let mut request = Some(wire_request("mistral/model", &base, json!({})));
    let NativeOutcome::Completed(mut call) = OcrCall::admit(
        super::test_support::ocr_client(),
        OcrAdmission {
            asynchronous: true,
            ..OcrAdmission::all()
        },
    ) else {
        panic!("supported call declined")
    };
    let selected = crate::Error::InvalidRequest("public metadata failed".into());
    let host = NoopOcrHost;
    let mut result = None;
    let mut failures = Vec::new();
    let error = loop {
        match call.resume(result.take()).await {
            Ok(OcrCallStep::Host(operation)) => {
                result = Some(match operation {
                    OcrHostOperation::Lifecycle(HostPhase::Finalize) => {
                        OcrHostResult::Lifecycle(Err(HostFailure::Error(selected.clone())))
                    }
                    OcrHostOperation::Failure { error, .. } => {
                        assert_eq!(error, selected);
                        failures.push("sync");
                        OcrHostResult::Lifecycle(Err(HostFailure::Error(
                            crate::Error::InvalidRequest("failure callback failed".into()),
                        )))
                    }
                    OcrHostOperation::Lifecycle(HostPhase::AsyncFailure) => {
                        failures.push("async");
                        OcrHostResult::Lifecycle(Ok(()))
                    }
                    OcrHostOperation::Success { .. }
                    | OcrHostOperation::MapFailure(_)
                    | OcrHostOperation::Lifecycle(HostPhase::DeploymentFailure) => {
                        panic!("finalization failure used provider/success dispatch")
                    }
                    OcrHostOperation::ProjectRequest => {
                        OcrHostResult::Request(Ok((Box::new(request.take().unwrap()), false)))
                    }
                    operation => host.invoke(operation).await,
                });
            }
            Ok(OcrCallStep::Complete(_)) => panic!("failed call completed successfully"),
            Err(error) => break error,
        }
    };
    server.await.unwrap();
    assert_eq!(error, selected);
    assert_eq!(failures, ["sync", "async"]);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn cancellation_at_provider_hook_prevents_execution_and_further_resumption() {
    use crate::call_lifecycle::host::HostFailure;

    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(AdmissionSpy {
            effects: Arc::new(Mutex::new(0)),
        }),
        ..wire_request("mistral/model", "http://127.0.0.1:1", json!({}))
    };
    let NativeOutcome::Completed(mut call) =
        OcrCall::admit(super::test_support::ocr_client(), OcrAdmission::all())
    else {
        panic!("supported call declined")
    };
    let mut request = Some(request);
    let host = NoopOcrHost;
    let mut result = None;
    loop {
        match call.resume(result.take()).await.unwrap() {
            OcrCallStep::Host(OcrHostOperation::PreCall(_)) => break,
            OcrCallStep::Host(OcrHostOperation::ProjectRequest) => {
                result = Some(OcrHostResult::Request(Ok((
                    Box::new(request.take().unwrap()),
                    false,
                ))))
            }
            OcrCallStep::Host(operation) => result = Some(host.invoke(operation).await),
            OcrCallStep::Complete(_) => panic!("provider executed before pre-call result"),
        }
    }
    let selected = crate::Error::InvalidRequest("cancelled".into());
    assert!(matches!(
        call.interrupt(HostFailure::Cancelled(selected.clone())).await,
        Err(error) if error == selected
    ));
    assert!(
        call.resume(Some(OcrHostResult::Lifecycle(Ok(()))))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn missing_host_result_preserves_pending_operation() {
    use crate::call_lifecycle::host::HostPhase;

    let NativeOutcome::Completed(mut call) =
        OcrCall::admit(super::test_support::ocr_client(), OcrAdmission::all())
    else {
        panic!("supported call declined")
    };
    assert!(matches!(
        call.resume(None).await.unwrap(),
        OcrCallStep::Host(OcrHostOperation::Lifecycle(HostPhase::Setup))
    ));
    assert!(call.resume(None).await.is_err());
    assert!(matches!(
        call.resume(Some(OcrHostResult::Lifecycle(Ok(()))))
            .await
            .unwrap(),
        OcrCallStep::Host(OcrHostOperation::Lifecycle(HostPhase::Prepare))
    ));
}

async fn read_bounded_response(
    response: Vec<u8>,
    limit: usize,
) -> Result<bytes::Bytes, super::error::OcrError> {
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
        super::client::read_response_bytes(response, limit),
    )
    .await;
    server.abort();
    let _ = server.await;
    result.expect("bounded reads must finish without waiting for the rest of an oversized body")
}

#[tokio::test]
async fn response_limit_accepts_exact_size_and_rejects_declared_and_chunked_overflow() {
    use super::error::{OcrError, OcrResponseError};

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
            Err(OcrError::Response(OcrResponseError::TooLarge { limit: 8 }))
        ));
    }
}

#[tokio::test]
async fn oversized_error_retains_http_status_and_bounded_diagnostics_without_draining() {
    let prefix = "x".repeat(4 * (crate::constants::UPSTREAM_ERROR_BODY_MAX_CHARS + 1));
    for headers in ["Content-Length: 1000000", "Transfer-Encoding: chunked"] {
        let body = if headers.starts_with("Transfer") {
            format!("{:x}\r\n{prefix}\r\n", prefix.len())
        } else {
            prefix.clone()
        };
        let response = format!("HTTP/1.1 429 Too Many Requests\r\n{headers}\r\n\r\n{body}");
        let error = read_bounded_response(response.into_bytes(), 4096)
            .await
            .unwrap_err();
        match error {
            super::error::OcrError::Transport(crate::error::TransportError::Http {
                status,
                body,
            }) => {
                assert_eq!(status, 429);
                assert_eq!(
                    body,
                    format!(
                        "{}... (truncated)",
                        "x".repeat(crate::constants::UPSTREAM_ERROR_BODY_MAX_CHARS)
                    )
                );
            }
            error => panic!("unexpected error: {error}"),
        }
    }
}

#[test]
fn response_limit_is_validated_and_not_forwarded_to_the_provider() {
    let request = wire_request(
        "mistral/model",
        "http://localhost",
        json!({"max_response_bytes": 123}),
    );
    assert_eq!(request.connection.max_response_bytes, 123);
    assert!(!request.optional_params.contains_key("max_response_bytes"));
    for value in [
        json!(0),
        json!(-1),
        json!(true),
        json!("123"),
        json!(1.5),
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

impl crate::auth::TokenProvider for PendingToken {
    fn acquire(&self) -> crate::auth::TokenFuture<'_> {
        Box::pin(async move {
            let _guard = TokenFutureDrop(self.dropped.clone());
            self.entered.notify_one();
            std::future::pending().await
        })
    }
}

#[tokio::test]
async fn cancellation_waits_for_provider_capture_drop_even_when_acknowledgement_is_cancelled() {
    use crate::call_lifecycle::host::HostFailure;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::Poll;

    for interrupt_acknowledgement in [false, true] {
        let entered = Arc::new(tokio::sync::Notify::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let request = wire_request("azure_ai/mistral-ocr", "https://example.invalid", json!({}));
        let request = super::LiteLLMOcrRequest {
            connection: super::OcrConnection {
                extra_headers: vec![("authorization".into(), "Bearer test-key".into())],
                ..request.connection
            },
            azure_ad_token_provider: Some(crate::auth::TokenProviderHandle::new(Arc::new(
                PendingToken {
                    entered: entered.clone(),
                    dropped: dropped.clone(),
                },
            ))),
            ..request
        };
        let NativeOutcome::Completed(mut call) =
            OcrCall::admit(super::test_support::ocr_client(), OcrAdmission::all())
        else {
            panic!("supported call declined")
        };
        let mut request = Some(request);
        let mut result = None;
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                tokio::select! {
                    _ = entered.notified() => break,
                    step = call.resume(result.take()) => {
                        result = Some(match step.unwrap() {
                            OcrCallStep::Host(OcrHostOperation::ProjectRequest) => OcrHostResult::Request(Ok((Box::new(request.take().unwrap()), false))),
                            OcrCallStep::Host(operation) => NoopOcrHost.invoke(operation).await,
                            OcrCallStep::Complete(_) => panic!("pending provider completed"),
                        });
                    }
                }
            }
        }).await.unwrap();
        assert!(!dropped.load(Ordering::SeqCst));
        let selected = crate::Error::InvalidRequest("cancelled".into());
        if interrupt_acknowledgement {
            let mut acknowledgement =
                Box::pin(call.interrupt(HostFailure::Cancelled(selected.clone())));
            std::future::poll_fn(|cx| {
                assert!(acknowledgement.as_mut().poll(cx).is_pending());
                Poll::Ready(())
            })
            .await;
            drop(acknowledgement);
            assert!(!dropped.load(Ordering::SeqCst));
        }
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            call.interrupt(HostFailure::Cancelled(selected.clone())),
        )
        .await
        .unwrap();
        assert!(matches!(result, Err(error) if error == selected));
        assert!(
            dropped.load(Ordering::SeqCst),
            "cancellation returned while provider captures were still alive"
        );
    }
}
