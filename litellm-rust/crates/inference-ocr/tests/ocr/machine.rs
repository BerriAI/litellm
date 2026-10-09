use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use litellm_host::{
    hooks::NativeHooks,
    interceptors::WireRequest,
    lifecycle::{CallEvent, ExecutionEvent},
    machine::{HostFailure, Machine, MachineStep},
    protocol::{HostRequest, InterceptRequest},
};
use litellm_host_native::services::HostCallHandler;
use litellm_inference_ocr::{
    route::{OcrCall, OcrMachine, OcrOp},
    types::OcrDocumentInput,
};
use litellm_llms::base_llm::ocr::transformation::OcrTransportConfig;
use rstest::rstest;
use tokio::{io::AsyncReadExt, net::TcpListener, sync::Notify};

use super::*;

/// Drives the machine by hand, answering every op through `host` except `before_provider_request`,
/// which `intercept` answers so a test can fail or cancel exactly there.
async fn drive_until(
    host: &LocalOcrHost,
    mut intercept: impl FnMut(WireRequest) -> Result<WireRequest, HostFailure<Error>>,
) -> (
    Result<LiteLLMOcrResponse, Error>,
    Vec<&'static str>,
    OcrMachine,
) {
    let mut machine = ocr_route().machine(host.request().unwrap());
    let mut ops = Vec::new();
    let outcome = loop {
        let op = match machine.resume().await {
            Ok(MachineStep::Suspended(op)) => op,
            Ok(MachineStep::Complete(response)) => break Ok(completed(response)),
            Err(error) => break Err(error),
        };
        let answer = match op {
            HostRequest::Intercept(InterceptRequest::ResultReady { facts, reply }) => {
                host.on_event(&CallEvent::Execution(ExecutionEvent::ResultReady {
                    facts: facts.clone(),
                }));
                host.result_ready(&facts)
                    .map(|()| reply.send(()))
                    .map_err(|error| HostFailure::Error(Error::from(error)))
            }
            HostRequest::Stream(stream) => match stream {
                litellm_host::protocol::StreamDelivery::Open(head, _) => match head {},
                litellm_host::protocol::StreamDelivery::Chunk(chunk, _) => match chunk {},
            },
            HostRequest::HostCall(op) => {
                ops.push(match op {
                    OcrOp::AcquireAzureAdToken(_) => "AcquireAzureAdToken",
                });
                host.handle_host_call(op).await.map_err(HostFailure::Error)
            }
            HostRequest::Intercept(InterceptRequest::BeforeProviderRequest {
                wire, reply, ..
            }) => {
                ops.push("BeforeSend");
                intercept(*wire).map(|wire| reply.send(wire))
            }
            HostRequest::Intercept(InterceptRequest::AfterProviderResponse { raw, reply }) => {
                ops.push("response");
                host.on_event(&CallEvent::Execution(
                    ExecutionEvent::ProviderResponseReceived { raw },
                ));
                reply.send(());
                Ok(())
            }
        };
        if let Err(failure) = answer {
            break machine.interrupt(failure).await.map(completed);
        }
    };
    (outcome, ops, machine)
}

/// Answers every op until `stop` fires, leaving the machine suspended mid-call.
async fn drive_until_notified(machine: &mut OcrMachine, host: &LocalOcrHost, stop: &Notify) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            tokio::select! {
                _ = stop.notified() => break,
                step = machine.resume() => {
                    match step.unwrap() {
                        MachineStep::Suspended(HostRequest::Intercept(InterceptRequest::ResultReady { reply, .. })) => reply.send(()),
                        MachineStep::Suspended(HostRequest::HostCall(op)) => host.handle_host_call(op).await.unwrap(),
                        MachineStep::Suspended(HostRequest::Intercept(InterceptRequest::BeforeProviderRequest { wire, reply, .. })) => reply.send(*wire),
                        MachineStep::Suspended(HostRequest::Intercept(InterceptRequest::AfterProviderResponse { reply, .. })) => reply.send(()),
                        MachineStep::Suspended(HostRequest::Stream(stream)) => match stream {
                            litellm_host::protocol::StreamDelivery::Open(head, _) => match head {},
                            litellm_host::protocol::StreamDelivery::Chunk(chunk, _) => match chunk {},
                        },
                        MachineStep::Complete(_) => panic!("the stalled call completed"),
                    }
                }
            }
        }
    })
    .await
    .expect("the call reached the stall point");
}

#[rstest::rstest]
#[tokio::test]
async fn a_hand_driven_machine_performs_the_same_call() {
    let upstream = upstream([json_response(json!({
        "pages": [{"index": 0, "markdown": "native"}]
    }))])
    .await;
    let host = LocalOcrHost::new(ocr_request("mistral/model", &upstream.uri(), json!({})));

    let (outcome, ops, mut machine) = drive_until(&host, Ok).await;

    assert_eq!(outcome.unwrap().pages[0].markdown, "native");
    assert_eq!(received(&upstream).await.len(), 1);
    assert_eq!(ops, ["BeforeSend", "response"]);
    assert!(matches!(
        machine.resume().await,
        Err(Error::InvalidRequest(_))
    ));
}

#[rstest::rstest]
#[tokio::test]
async fn a_path_document_is_read_by_core_without_a_host_operation() {
    let upstream = upstream([json_response(json!({
        "pages": [{"index": 0, "markdown": "path"}]
    }))])
    .await;
    let dir = std::env::temp_dir().join(format!("litellm-ocr-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("scan.png");
    std::fs::write(&path, b"abc").unwrap();
    let request = ocr_request("mistral/model", &upstream.uri(), json!({})).with_document(
        OcrDocumentInput::Path {
            path,
            mime_type: None,
        },
    );

    let (response, ops, _) = drive_until(&LocalOcrHost::new(request), Ok).await;
    std::fs::remove_dir_all(&dir).unwrap();

    assert_eq!(response.unwrap().pages[0].markdown, "path");
    assert_eq!(ops, ["BeforeSend", "response"]);
    assert_eq!(
        only_request(&upstream).await.json()["document"]["image_url"],
        "data:image/png;base64,YWJj"
    );
}

#[rstest]
#[case::failed(HostFailure::Error(Error::InvalidRequest("before_provider_request failed".into())), "before_provider_request failed")]
#[case::cancelled(HostFailure::Cancelled(Error::InvalidRequest("cancelled".into())), "cancelled")]
#[tokio::test]
async fn a_before_send_failure_ends_the_call_without_reaching_transport(
    #[case] failure: HostFailure<Error>,
    #[case] message: &str,
) {
    let upstream = upstream([pages_response()]).await;
    let host = LocalOcrHost::new(ocr_request("mistral/model", &upstream.uri(), json!({})));
    let failure = Arc::new(std::sync::Mutex::new(Some(failure)));

    let (outcome, ops, mut machine) = drive_until(&host, |_| {
        Err(failure
            .lock()
            .unwrap()
            .take()
            .expect("before_provider_request is asked once"))
    })
    .await;

    assert!(
        matches!(&outcome, Err(Error::InvalidRequest(actual)) if actual == message),
        "{outcome:?}"
    );
    assert_eq!(ops, ["BeforeSend"]);
    assert!(machine.resume().await.is_err());
    assert!(received(&upstream).await.is_empty());
}

#[rstest::rstest]
#[tokio::test]
async fn resuming_before_answering_keeps_the_pending_operation() {
    let upstream = upstream([pages_response()]).await;
    let request = ocr_request("mistral/model", &upstream.uri(), json!({}));
    let mut machine = ocr_route().machine(OcrCall {
        request,
        caller_token: false,
    });
    let Ok(MachineStep::Suspended(HostRequest::Intercept(
        InterceptRequest::BeforeProviderRequest { wire, reply, .. },
    ))) = machine.resume().await
    else {
        panic!("expected the provider request hook");
    };
    assert!(machine.resume().await.is_err());
    reply.send(*wire);
    assert!(matches!(
        machine.resume().await,
        Ok(MachineStep::Suspended(HostRequest::Intercept(
            InterceptRequest::AfterProviderResponse { .. }
        )))
    ));
}

#[derive(Debug)]
struct PendingToken {
    entered: Arc<Notify>,
    dropped: Arc<AtomicBool>,
}

struct TokenFutureDrop(Arc<AtomicBool>);

impl Drop for TokenFutureDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
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

#[rstest::rstest]
#[tokio::test]
async fn interrupt_drops_provider_captures_before_returning() {
    let entered = Arc::new(Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let mut request = ocr_request("azure_ai/mistral-ocr", "https://example.invalid", json!({}));
    request.transport = OcrTransportConfig {
        extra_headers: vec![("authorization".into(), "Bearer test-key".into())],
        ..request.transport
    };
    request.azure_ad_token_provider = Some(litellm_auth::TokenProviderHandle::new(Arc::new(
        PendingToken {
            entered: entered.clone(),
            dropped: dropped.clone(),
        },
    )));
    let host = LocalOcrHost::new(request);
    let mut machine = ocr_route().machine(host.request().unwrap());

    drive_until_notified(&mut machine, &host, &entered).await;
    assert!(!dropped.load(Ordering::SeqCst));
    let acknowledgement = machine.interrupt(HostFailure::Cancelled(Error::InvalidRequest(
        "cancelled".into(),
    )));

    assert!(
        dropped.load(Ordering::SeqCst),
        "interrupt returned while provider captures were still alive"
    );
    assert!(
        matches!(acknowledgement.await, Err(Error::InvalidRequest(message)) if message == "cancelled")
    );
}

#[rstest::rstest]
#[tokio::test]
async fn interrupting_an_in_flight_provider_request_closes_its_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let received = Arc::new(Notify::new());
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
        while socket.read(&mut buffer).await.unwrap() != 0 {}
    });
    let host = LocalOcrHost::new(ocr_request("mistral/model", &base, json!({})));
    let mut machine = ocr_route().machine(host.request().unwrap());

    drive_until_notified(&mut machine, &host, &received).await;
    let cancelled = Error::InvalidRequest("cancelled".into());

    assert!(
        machine
            .interrupt(HostFailure::Cancelled(cancelled))
            .await
            .is_err()
    );
    tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("the provider connection stayed open after the interrupt")
        .unwrap();
}
