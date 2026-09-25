use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use litellm_core::ocr::{
    route::{OcrMachine, OcrOp, OcrProjection},
    types::OcrDocumentInput,
};
use litellm_host::{
    event::{CallEvent, WireRequest},
    host::{Host, HostOp},
    machine::{HostFailure, Machine, MachineStep},
};
use litellm_llms::base_llm::ocr::transformation::OcrTransportConfig;
use rstest::rstest;
use tokio::{io::AsyncReadExt, net::TcpListener, sync::Notify};

use super::{lifecycle::event_name, *};

/// Drives the machine by hand, answering every op through `host` except `before_send`,
/// which `intercept` answers so a test can fail or cancel exactly there.
async fn drive_until(
    host: &LocalOcrHost,
    mut intercept: impl FnMut(WireRequest) -> Result<WireRequest, HostFailure<Error>>,
) -> (
    Result<LiteLLMOcrResponse, Error>,
    Vec<&'static str>,
    OcrMachine,
) {
    let mut machine = ocr_machine(ocr_client());
    let mut ops = Vec::new();
    let outcome = loop {
        let op = match machine.resume().await {
            Ok(MachineStep::Host(op)) => op,
            Ok(MachineStep::Complete(response)) => break Ok(response),
            Err(error) => break Err(error),
        };
        let answer = match op {
            HostOp::Project(reply) => {
                ops.push("Project");
                host.project()
                    .await
                    .map(|projection| reply.send(projection))
                    .map_err(HostFailure::Error)
            }
            HostOp::Custom(op) => {
                ops.push(match op {
                    OcrOp::AcquireAzureAdToken(_) => "AcquireAzureAdToken",
                });
                host.custom_op(op).await.map_err(HostFailure::Error)
            }
            HostOp::PreRequest { request, reply } => {
                ops.push("PreRequest");
                host.pre_request(*request)
                    .await
                    .map(|params| reply.send(params))
                    .map_err(HostFailure::Error)
            }
            HostOp::BeforeSend { wire, reply, .. } => {
                ops.push("BeforeSend");
                intercept(*wire).map(|wire| reply.send(wire))
            }
            HostOp::AfterResponse { response, reply } => {
                ops.push("AfterResponse");
                host.after_response(*response)
                    .await
                    .map(|verdict| reply.send(verdict))
                    .map_err(HostFailure::Error)
            }
            HostOp::Emit(event, reply) => {
                let event = CallEvent::Machine(event);
                ops.push(event_name(&event));
                host.emit(&event)
                    .await
                    .map(|()| reply.send(()))
                    .map_err(HostFailure::Error)
            }
        };
        if let Err(failure) = answer {
            break machine.interrupt(failure).await;
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
                        MachineStep::Host(HostOp::Project(reply)) => reply.send(host.project().await.unwrap()),
                        MachineStep::Host(HostOp::Custom(op)) => host.custom_op(op).await.unwrap(),
                        MachineStep::Host(HostOp::PreRequest { request, reply }) => reply.send(host.pre_request(*request).await.unwrap()),
                        MachineStep::Host(HostOp::BeforeSend { wire, reply, .. }) => reply.send(*wire),
                        MachineStep::Host(HostOp::AfterResponse { response, reply }) => reply.send(host.after_response(*response).await.unwrap()),
                        MachineStep::Host(HostOp::Emit(_, reply)) => reply.send(()),
                        MachineStep::Complete(_) => panic!("the stalled call completed"),
                    }
                }
            }
        }
    })
    .await
    .expect("the call reached the stall point");
}

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
    assert_eq!(ops, ["Project", "BeforeSend", "response"]);
    assert!(matches!(
        machine.resume().await,
        Err(Error::InvalidRequest(_))
    ));
}

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
    assert_eq!(ops, ["Project", "BeforeSend", "response"]);
    assert_eq!(
        only_request(&upstream).await.json()["document"]["image_url"],
        "data:image/png;base64,YWJj"
    );
}

#[rstest]
#[case::failed(HostFailure::Error(Error::InvalidRequest("before_send failed".into())), "before_send failed")]
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
            .expect("before_send is asked once"))
    })
    .await;

    assert!(
        matches!(&outcome, Err(Error::InvalidRequest(actual)) if actual == message),
        "{outcome:?}"
    );
    assert_eq!(ops, ["Project", "BeforeSend"]);
    assert!(machine.resume().await.is_err());
    assert!(received(&upstream).await.is_empty());
}

#[tokio::test]
async fn resuming_before_answering_keeps_the_pending_operation() {
    let request = ocr_request("mistral/model", UNREACHABLE_BASE, json!({}));
    let mut machine = ocr_machine(ocr_client());
    let Ok(MachineStep::Host(HostOp::Project(reply))) = machine.resume().await else {
        panic!("expected the projection op first");
    };

    assert!(machine.resume().await.is_err());
    reply.send(OcrProjection {
        request,
        caller_token: false,
    });
    assert!(matches!(
        machine.resume().await,
        Ok(MachineStep::Host(HostOp::BeforeSend { .. }))
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
    let mut machine = ocr_machine(ocr_client());

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
    let mut machine = ocr_machine(ocr_client());

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
