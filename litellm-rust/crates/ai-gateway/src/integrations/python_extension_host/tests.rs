use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use futures_util::{Stream, StreamExt, stream};
use litellm_python_extension_protocol::python_extension_host_server::{
    PythonExtensionHost, PythonExtensionHostServer,
};
use litellm_python_extension_protocol::*;
use serde_json::json;
use tokio::sync::Notify;
use tonic::{Request, Response, Status};

use crate::integrations::custom_guardrail::{
    CustomGuardrail, CustomGuardrailRunner, GuardrailContext, GuardrailEventHook, GuardrailRequest,
};
use crate::integrations::custom_logger::{
    CallType, CallbackTiming, CallbackValue, CustomLogger, ModelCallDetails,
};

use super::adapters::{RemoteCustomGuardrail, RemoteCustomLogger};
use super::client::{ActivationState, PythonExtensionClient};
use super::config::{
    ManifestExtension, ManifestExtensionKind, PythonExtensionManifest, PythonExtensionSettings,
};
use super::stream::RemoteStreamTransformer;

const TOKEN: &str = "rust-test-token";

#[derive(Default)]
struct MockHost {
    callback_count: AtomicUsize,
    callback_notify: Notify,
    fail_operations: AtomicBool,
}

#[tonic::async_trait]
impl PythonExtensionHost for MockHost {
    async fn get_capabilities(
        &self,
        request: Request<GetCapabilitiesRequest>,
    ) -> Result<Response<HostCapabilities>, Status> {
        assert_token(&request);
        Ok(Response::new(HostCapabilities {
            protocol_major: 1,
            protocol_minor: 0,
            supported_hooks: vec![
                "async_pre_call_hook".to_string(),
                "async_moderation_hook".to_string(),
                "async_post_call_success_hook".to_string(),
            ],
            supports_duplex_streaming: true,
            supports_callback_batching: true,
            max_callback_batch_size: 50,
            ..Default::default()
        }))
    }

    async fn prepare_revision(
        &self,
        request: Request<PrepareRevisionRequest>,
    ) -> Result<Response<PrepareRevisionResponse>, Status> {
        assert_token(&request);
        let descriptors = request
            .get_ref()
            .extensions
            .iter()
            .map(|extension| ExtensionDescriptor {
                id: extension.id.clone(),
                kind: extension.kind,
                hooks: vec![
                    "async_pre_call_hook".to_string(),
                    "async_moderation_hook".to_string(),
                    "async_post_call_success_hook".to_string(),
                ],
                ..Default::default()
            })
            .collect();
        Ok(Response::new(PrepareRevisionResponse {
            operation: Some(ok()),
            extensions: descriptors,
        }))
    }

    async fn commit_revision(
        &self,
        request: Request<CommitRevisionRequest>,
    ) -> Result<Response<OperationResult>, Status> {
        assert_token(&request);
        Ok(Response::new(ok()))
    }

    async fn retire_revision(
        &self,
        request: Request<RetireRevisionRequest>,
    ) -> Result<Response<OperationResult>, Status> {
        assert_token(&request);
        Ok(Response::new(ok()))
    }

    async fn execute_guardrail(
        &self,
        request: Request<GuardrailInvocation>,
    ) -> Result<Response<GuardrailResult>, Status> {
        assert_token(&request);
        let invocation = request.into_inner();
        if self.fail_operations.load(Ordering::SeqCst) {
            return Ok(Response::new(GuardrailResult {
                operation: Some(operation_error()),
                decision: GuardrailDecision::Error.into(),
                ..Default::default()
            }));
        }
        let body = if invocation.hook_phase == HookPhase::PostCall as i32 {
            invocation.response_json.unwrap_or_default()
        } else {
            invocation.request_json
        };
        let mut value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        if value.get("block") == Some(&json!(true)) {
            return Ok(Response::new(GuardrailResult {
                operation: Some(ok()),
                decision: GuardrailDecision::Block.into(),
                public_error: Some(PublicError {
                    r#type: "GuardrailRaisedException".to_string(),
                    message: "blocked by mock".to_string(),
                    status_code: Some(400),
                    ..Default::default()
                }),
                ..Default::default()
            }));
        }
        value["remote"] = json!(true);
        let replacement = serde_json::to_vec(&value).unwrap();
        Ok(Response::new(GuardrailResult {
            operation: Some(ok()),
            decision: if invocation.hook_phase == HookPhase::PostCall as i32 {
                GuardrailDecision::ReplaceResponse.into()
            } else {
                GuardrailDecision::ReplaceRequest.into()
            },
            request_json: (invocation.hook_phase != HookPhase::PostCall as i32)
                .then_some(replacement.clone()),
            response_json: (invocation.hook_phase == HookPhase::PostCall as i32)
                .then_some(replacement),
            ..Default::default()
        }))
    }

    async fn publish_callback_events(
        &self,
        request: Request<PublishCallbackEventsRequest>,
    ) -> Result<Response<PublishCallbackEventsResponse>, Status> {
        assert_token(&request);
        let count = request.get_ref().events.len();
        self.callback_count.fetch_add(count, Ordering::SeqCst);
        self.callback_notify.notify_one();
        Ok(Response::new(PublishCallbackEventsResponse {
            operations: (0..count)
                .map(|_| {
                    if self.fail_operations.load(Ordering::SeqCst) {
                        operation_error()
                    } else {
                        ok()
                    }
                })
                .collect(),
        }))
    }

    type TransformStreamStream = Pin<Box<dyn Stream<Item = Result<StreamFrame, Status>> + Send>>;

    async fn transform_stream(
        &self,
        request: Request<tonic::Streaming<StreamFrame>>,
    ) -> Result<Response<Self::TransformStreamStream>, Status> {
        assert_token(&request);
        let mut input = request.into_inner();
        let (sender, receiver) = tokio::sync::mpsc::channel(8);
        tokio::spawn(async move {
            while let Some(Ok(frame)) = input.next().await {
                match StreamFrameKind::try_from(frame.kind).unwrap_or(StreamFrameKind::Error) {
                    StreamFrameKind::Open => {
                        let fail_stream = frame
                            .open
                            .as_ref()
                            .and_then(|open| {
                                serde_json::from_slice::<serde_json::Value>(&open.request_json).ok()
                            })
                            .and_then(|request| request.get("fail_stream").cloned())
                            == Some(json!(true));
                        if fail_stream {
                            let _ = sender
                                .send(Ok(StreamFrame {
                                    kind: StreamFrameKind::Error.into(),
                                    stream_id: frame.stream_id,
                                    error: Some(PublicError {
                                        r#type: "plugin_error".to_string(),
                                        message: "plugin failed".to_string(),
                                        ..Default::default()
                                    }),
                                    ..Default::default()
                                }))
                                .await;
                            break;
                        }
                    }
                    StreamFrameKind::InputChunk => {
                        let mut value: serde_json::Value =
                            serde_json::from_slice(&frame.chunk_json.unwrap()).unwrap();
                        value["transformed"] = json!(true);
                        let _ = sender
                            .send(Ok(StreamFrame {
                                kind: StreamFrameKind::OutputChunk.into(),
                                stream_id: frame.stream_id,
                                chunk_json: Some(serde_json::to_vec(&value).unwrap()),
                                ..Default::default()
                            }))
                            .await;
                    }
                    StreamFrameKind::End => {
                        let _ = sender
                            .send(Ok(StreamFrame {
                                kind: StreamFrameKind::End.into(),
                                stream_id: frame.stream_id,
                                ..Default::default()
                            }))
                            .await;
                        break;
                    }
                    StreamFrameKind::Error => {
                        let _ = sender.send(Ok(frame)).await;
                        break;
                    }
                    _ => break,
                }
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(receiver),
        )))
    }
}

#[tokio::test]
async fn remote_guardrail_blocks_before_provider_and_mutates_all_phases() {
    let (host, client, _) = start_client().await;
    let guardrail = Arc::new(RemoteCustomGuardrail::new(
        "remote".to_string(),
        "guardrail-1".to_string(),
        vec![
            GuardrailEventHook::PreCall,
            GuardrailEventHook::DuringCall,
            GuardrailEventHook::PostCall,
        ],
        client,
    ));
    let runner = CustomGuardrailRunner::new(vec![guardrail.clone()]);
    let context = GuardrailContext::new(CallType::Ocr);
    let provider_called = Arc::new(AtomicBool::new(false));
    let called = provider_called.clone();
    let result = runner
        .run_before_provider(
            GuardrailEventHook::PreCall,
            &context,
            GuardrailRequest::new(json!({"block": true})),
            move |_| async move {
                called.store(true, Ordering::SeqCst);
                Ok(())
            },
        )
        .await;
    assert!(result.is_err());
    assert!(!provider_called.load(Ordering::SeqCst));

    let (request, _) = runner
        .run_pre_call(&context, GuardrailRequest::new(json!({"model": "ocr"})))
        .await
        .unwrap();
    assert_eq!(request.data["remote"], json!(true));
    let (response, _) = runner
        .run_post_call(&context, GuardrailRequest::new(json!({"id": "response"})))
        .await
        .unwrap();
    assert_eq!(response.data["remote"], json!(true));
    drop(host);
}

#[tokio::test]
async fn remote_logger_batches_terminal_event_to_same_host() {
    let (host, client, _) = start_client().await;
    let logger = RemoteCustomLogger::new("callback-1".to_string(), client);
    let details = ModelCallDetails::new("model", "provider", CallType::Ocr);
    logger
        .async_log_success_event(
            &details,
            &CallbackValue::new("ocr", json!({"id": "response"})),
            CallbackTiming::new(1.0, 2.0),
        )
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), host.callback_notify.notified())
        .await
        .unwrap();
    assert_eq!(host.callback_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn remote_stream_transformer_uses_one_duplex_rpc() {
    let (_host, client, _) = start_client().await;
    let transformer = RemoteStreamTransformer::new("callback-1".to_string(), client, true);
    let output = transformer
        .transform(
            json!({"model": "test"}),
            AuthContext::default(),
            stream::iter(vec![Ok(json!({"value": "hello"}))]),
        )
        .collect::<Vec<_>>()
        .await;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].as_ref().unwrap()["transformed"], json!(true));
}

#[tokio::test]
async fn unavailable_stream_fails_open_without_losing_buffered_chunks() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let (client, activation) =
        PythonExtensionClient::connect(settings(format!("http://{address}")), manifest())
            .await
            .unwrap();
    assert!(matches!(activation, ActivationState::Degraded(_)));
    let transformer = RemoteStreamTransformer::new("callback-1".to_string(), client, true);
    let originals = (0..16)
        .map(|value| json!({"value": value}))
        .collect::<Vec<_>>();
    let output = tokio::time::timeout(
        Duration::from_secs(2),
        transformer
            .transform(
                json!({"model": "test"}),
                AuthContext::default(),
                stream::iter(originals.clone().into_iter().map(Ok)),
            )
            .collect::<Vec<_>>(),
    )
    .await
    .unwrap();
    assert_eq!(
        output.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
        originals
    );
}

#[tokio::test]
async fn upstream_stream_failure_is_preserved() {
    let (_host, client, _) = start_client().await;
    let transformer = RemoteStreamTransformer::new("callback-1".to_string(), client, true);
    let output = transformer
        .transform(
            json!({"model": "test"}),
            AuthContext::default(),
            stream::iter(vec![
                Ok(json!({"value": "hello"})),
                Err(litellm_core::CoreError::Network(
                    "upstream closed".to_string(),
                )),
            ]),
        )
        .collect::<Vec<_>>()
        .await;
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].as_ref().unwrap()["transformed"], json!(true));
    assert!(matches!(
        &output[1],
        Err(litellm_core::CoreError::Network(message)) if message == "upstream closed"
    ));
}

#[tokio::test]
async fn plugin_stream_failure_passes_through_original_chunks() {
    let (_host, client, _) = start_client().await;
    let transformer = RemoteStreamTransformer::new("callback-1".to_string(), client, true);
    let originals = vec![json!({"value": 1}), json!({"value": 2})];
    let output = transformer
        .transform(
            json!({"fail_stream": true}),
            AuthContext::default(),
            stream::iter(originals.clone().into_iter().map(Ok)),
        )
        .collect::<Vec<_>>()
        .await;
    assert_eq!(
        output.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
        originals
    );
}

#[tokio::test]
async fn dropping_transformed_stream_cancels_upstream_production() {
    let (_host, client, _) = start_client().await;
    let transformer = RemoteStreamTransformer::new("callback-1".to_string(), client, true);
    let consumed = Arc::new(AtomicUsize::new(0));
    let observed = consumed.clone();
    let input = stream::iter(0..10_000).map(move |value| {
        observed.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"value": value}))
    });
    let mut output = transformer.transform(json!({}), AuthContext::default(), input);
    assert!(output.next().await.is_some());
    drop(output);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(consumed.load(Ordering::SeqCst) < 10_000);
}

#[tokio::test]
async fn unavailable_host_fails_open_and_records_bypass() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let settings = settings(format!("http://{address}"));
    let (client, activation) = PythonExtensionClient::connect(settings, manifest())
        .await
        .unwrap();
    assert!(matches!(activation, ActivationState::Degraded(_)));
    let guardrail = RemoteCustomGuardrail::new(
        "remote".to_string(),
        "guardrail-1".to_string(),
        vec![GuardrailEventHook::PreCall],
        client.clone(),
    );
    let decision = guardrail
        .async_pre_call_hook(
            &GuardrailContext::new(CallType::Ocr),
            GuardrailRequest::new(json!({"model": "ocr"})),
        )
        .await
        .unwrap();
    assert!(matches!(
        decision,
        crate::integrations::custom_guardrail::GuardrailDecision::Allow(_)
    ));
    assert!(!client.bypass_counts().is_empty());
}

#[tokio::test]
async fn plugin_operation_errors_fail_open_and_record_bypasses() {
    let (host, client, _) = start_client().await;
    host.fail_operations.store(true, Ordering::SeqCst);
    let guardrail = RemoteCustomGuardrail::new(
        "remote".to_string(),
        "guardrail-1".to_string(),
        vec![GuardrailEventHook::PreCall],
        client.clone(),
    );
    let decision = guardrail
        .async_pre_call_hook(
            &GuardrailContext::new(CallType::Ocr),
            GuardrailRequest::new(json!({"model": "ocr"})),
        )
        .await
        .unwrap();
    assert!(matches!(
        decision,
        crate::integrations::custom_guardrail::GuardrailDecision::Allow(_)
    ));

    let logger = RemoteCustomLogger::new("callback-1".to_string(), client.clone());
    logger
        .async_log_success_event(
            &ModelCallDetails::new("model", "provider", CallType::Ocr),
            &CallbackValue::new("ocr", json!({"id": "response"})),
            CallbackTiming::new(1.0, 2.0),
        )
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), host.callback_notify.notified())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    let counts = client.bypass_counts();
    assert!(counts.keys().any(|(plugin, hook, _)| {
        plugin == "guardrail-1" && hook == &(HookPhase::PreCall as i32).to_string()
    }));
    assert!(
        counts
            .keys()
            .any(|(plugin, hook, _)| plugin == "callback-1" && hook == "callback")
    );
}

async fn start_client() -> (
    Arc<MockHost>,
    Arc<PythonExtensionClient>,
    tokio::task::JoinHandle<()>,
) {
    let host = Arc::new(MockHost::default());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let service = PythonExtensionHostServer::from_arc(host.clone());
    let server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(service)
            .serve(address)
            .await
            .unwrap();
    });
    let (client, activation) =
        PythonExtensionClient::connect(settings(format!("http://{address}")), manifest())
            .await
            .unwrap();
    assert!(matches!(activation, ActivationState::Active(_)));
    (host, client, server)
}

fn manifest() -> PythonExtensionManifest {
    PythonExtensionManifest {
        revision_id: "rust-test-revision".to_string(),
        extensions: vec![
            ManifestExtension {
                id: "guardrail-1".to_string(),
                kind: ManifestExtensionKind::Guardrail,
                entrypoint: "fixture.Guardrail".to_string(),
                constructor: json!({"kwargs": {"guardrail_name": "remote"}}),
            },
            ManifestExtension {
                id: "callback-1".to_string(),
                kind: ManifestExtensionKind::Callback,
                entrypoint: "fixture.callback".to_string(),
                constructor: json!({"callback_events": ["success", "failure"]}),
            },
        ],
    }
}

fn settings(endpoint: String) -> PythonExtensionSettings {
    PythonExtensionSettings {
        endpoint,
        token: TOKEN.to_string(),
        connect_timeout: Duration::from_millis(200),
        hook_timeout: Duration::from_millis(200),
        callback_queue_size: 8,
        callback_batch_size: 4,
    }
}

fn ok() -> OperationResult {
    OperationResult {
        ok: true,
        ..Default::default()
    }
}

fn operation_error() -> OperationResult {
    OperationResult {
        ok: false,
        error_code: ErrorCode::ExtensionFailed.into(),
        error_message: "plugin failed".to_string(),
    }
}

fn assert_token<T>(request: &Request<T>) {
    assert_eq!(
        request
            .metadata()
            .get("x-litellm-extension-token")
            .and_then(|value| value.to_str().ok()),
        Some(TOKEN)
    );
}
