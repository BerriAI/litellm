use std::future::{Ready, ready};
use std::sync::Mutex;
use std::time::SystemTime;

use litellm_core::Error;
use litellm_core::audio_transcription::{
    AudioFormat, AudioGuardrailRunner, AudioInput, AudioRoute, AudioRouteRequest, AudioServices,
};
use litellm_core::integrations::custom_logger::LogFuture;
use litellm_core::lifecycle::program::{Observations, Operation, ProgramOptions};
use litellm_core::lifecycle::{
    ActionResult, CallLifecycle, CallLifecycleContext, CallbackFuture, Clock,
    DeploymentFailureHooks, DeploymentPreHooks, DeploymentSuccessHooks, ModerationHooks, Outcome,
    PreCallHooks, TerminalDispatcher, TerminalRecord,
};
use litellm_core::providers::auth::AuthorizationServices;
use rstest::rstest;
use serde_json::Value;

#[derive(Default)]
struct Services {
    reject: Option<Operation>,
    audio_api_base: Option<String>,
    events: Mutex<Vec<Operation>>,
    terminals: Mutex<Vec<TerminalRecord>>,
}

impl Clock for Services {
    fn now(&self) -> f64 {
        1.0
    }
}

impl<Request: Send> PreCallHooks<Request> for Services {
    type PreCallFuture<'a> = Ready<ActionResult<Request, Error>>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: Request,
    ) -> Self::PreCallFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

impl<Request: Send> ModerationHooks<Request> for Services {
    type ModerationFuture<'a> = Ready<ActionResult<Request, Error>>;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: Request,
    ) -> Self::ModerationFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

impl DeploymentPreHooks<Value> for Services {
    fn async_pre_call_deployment_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: Value,
    ) -> CallbackFuture<'a, ActionResult<Value, Error>>
    where
        Value: 'a,
    {
        Box::pin(async move {
            self.events.lock().unwrap().push(Operation::DeploymentPre);
            if self.reject == Some(Operation::DeploymentPre) {
                return ActionResult::Reject(Error::InvalidRequest("rejected".into()));
            }
            ActionResult::Continue(request)
        })
    }
}

impl DeploymentSuccessHooks<Value> for Services {
    fn async_post_call_success_deployment_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        response: Value,
    ) -> CallbackFuture<'a, ActionResult<Value, Error>>
    where
        Value: 'a,
    {
        Box::pin(async move {
            self.events
                .lock()
                .unwrap()
                .push(Operation::DeploymentSuccess);
            if self.reject == Some(Operation::DeploymentSuccess) {
                return ActionResult::Reject(Error::InvalidRequest("rejected".into()));
            }
            if self.audio_api_base.is_some() {
                return ActionResult::Replace(serde_json::json!({"text": "replaced transcript"}));
            }
            ActionResult::Continue(response)
        })
    }
}

impl DeploymentFailureHooks for Services {
    fn async_post_call_failure_deployment_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        _: &'a Error,
    ) -> CallbackFuture<'a, Result<(), Error>> {
        Box::pin(async move {
            self.events
                .lock()
                .unwrap()
                .push(Operation::DeploymentFailure);
            Err(Error::InvalidRequest("observer failed".into()))
        })
    }
}

impl TerminalDispatcher for Services {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        Box::pin(async move {
            self.terminals.lock().unwrap().push(terminal.clone());
            Ok(())
        })
    }
}

fn context(route: &str) -> CallLifecycleContext {
    CallLifecycleContext::new(route, "test-model", "test-provider", "contract-call")
}

#[rstest]
#[case::success(None)]
#[case::provider_failure(Some(Operation::Send))]
#[case::pre_rejection(Some(Operation::DeploymentPre))]
#[case::success_rejection(Some(Operation::DeploymentSuccess))]
#[tokio::test]
async fn state_machine_and_executor_agree_on_deployment_notifications(
    #[case] reject: Option<Operation>,
) {
    let services = Services {
        reject,
        ..Services::default()
    };
    let result = CallLifecycle::asynchronous()
        .run(
            context("messages"),
            Value::Null,
            &services,
            &services,
            &services,
            |_| async {
                if reject == Some(Operation::Send) {
                    Err(Error::InvalidRequest("rejected".into()))
                } else {
                    Ok(Value::Null)
                }
            },
        )
        .await
        .into_result();
    if reject.is_some() {
        assert!(matches!(result, Err(Error::InvalidRequest(message)) if message == "rejected"));
    } else {
        assert!(result.is_ok());
    }
    assert_eq!(services.terminals.lock().unwrap().len(), 1);

    let mut planned = CallLifecycle::planned(ProgramOptions {
        asynchronous: true,
        internal_call: false,
    });
    let mut notifications = Vec::new();
    while !matches!(planned.operation(), Operation::Complete(_)) {
        let operation = planned.operation();
        if matches!(
            operation,
            Operation::DeploymentPre | Operation::DeploymentSuccess | Operation::DeploymentFailure
        ) {
            notifications.push(operation);
        }
        let ticket = planned.issue().unwrap();
        planned
            .complete_operation(
                ticket,
                if reject == Some(operation) || operation == Operation::DeploymentFailure {
                    Outcome::Failure
                } else {
                    Outcome::Success
                },
                Observations {
                    logger_available: true,
                    has_fallbacks: false,
                },
            )
            .unwrap();
    }
    assert_eq!(
        planned.operation(),
        Operation::Complete(if reject.is_some() {
            Outcome::Failure
        } else {
            Outcome::Success
        })
    );
    assert_eq!(notifications, *services.events.lock().unwrap());
}

impl AuthorizationServices for Services {
    fn environment(&self, _: &str) -> Option<String> {
        None
    }
    fn signing_time(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }

    #[cfg(feature = "bedrock-auth")]
    fn resolve_aws_credentials<'a>(
        &'a self,
        config: litellm_core::providers::bedrock::aws_base::AwsAuthConfig,
    ) -> litellm_core::providers::bedrock::aws_base::AwsCredentialFuture<'a> {
        litellm_core::providers::auth::shared_native_authorization_services()
            .resolve_aws_credentials(config)
    }
}

impl AudioServices for Services {
    fn guardrails(&self) -> AudioGuardrailRunner {
        AudioGuardrailRunner::new(vec![])
    }
}

#[tokio::test]
async fn audio_preparation_failure_notifies_supplied_deployment_hook() {
    let services = Services::default();
    let result = AudioRoute::execute(
        &services,
        AudioRouteRequest {
            model: "unsupported/model",
            audio: AudioInput {
                data: "AQI=".into(),
                format: AudioFormat::Wav,
                filename: None,
            },
            api_key: None,
            api_base: None,
            custom_llm_provider: Some("unsupported"),
            extra_headers: None,
            optional_params: Default::default(),
            timeout: None,
            request_metadata: Default::default(),
            litellm_call_id: Some("contract-call"),
        },
    )
    .await
    .into_result();

    assert!(matches!(result, Err(Error::InvalidProvider(_))));
    assert_eq!(services.terminals.lock().unwrap().len(), 1);
    assert_eq!(
        *services.events.lock().unwrap(),
        [Operation::DeploymentPre, Operation::DeploymentFailure]
    );
}

impl DeploymentPreHooks<litellm_core::audio_transcription::PreparedAudioTranscriptionRequest>
    for Services
{
    fn async_pre_call_deployment_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: litellm_core::audio_transcription::PreparedAudioTranscriptionRequest,
    ) -> CallbackFuture<
        'a,
        ActionResult<litellm_core::audio_transcription::PreparedAudioTranscriptionRequest, Error>,
    >
    where
        litellm_core::audio_transcription::PreparedAudioTranscriptionRequest: 'a,
    {
        Box::pin(async move {
            self.events.lock().unwrap().push(Operation::DeploymentPre);
            if self.reject == Some(Operation::DeploymentPre) {
                return ActionResult::Reject(Error::InvalidRequest("rejected".into()));
            }
            if let Some(api_base) = &self.audio_api_base {
                return ActionResult::Replace(
                    litellm_core::audio_transcription::PreparedAudioTranscriptionRequest {
                        api_base: Some(api_base.clone()),
                        audio: AudioInput {
                            data: "AwQ=".into(),
                            ..request.audio
                        },
                        ..request
                    },
                );
            }
            ActionResult::Continue(request)
        })
    }
}

#[cfg(feature = "bedrock-auth")]
#[rstest]
#[case::replacement(None)]
#[case::pre_rejection(Some(Operation::DeploymentPre))]
#[case::success_rejection(Some(Operation::DeploymentSuccess))]
#[tokio::test]
async fn audio_deployment_hooks_replace_payloads_and_rejections_preserve_error(
    #[case] reject: Option<Operation>,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api_base = format!("http://{}", listener.local_addr().unwrap());
    let services = Services {
        reject,
        audio_api_base: Some(api_base),
        ..Services::default()
    };
    let provider = tokio::spawn(async move {
        let accepted =
            tokio::time::timeout(std::time::Duration::from_millis(500), listener.accept()).await;
        if reject == Some(Operation::DeploymentPre) {
            assert!(accepted.is_err(), "rejected request reached provider");
            return;
        }
        let (mut socket, _) = accepted.unwrap().unwrap();
        let mut received = vec![];
        loop {
            let mut buffer = [0u8; 4096];
            let count = socket.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0);
            received.extend_from_slice(&buffer[..count]);
            let request = String::from_utf8_lossy(&received);
            if let Some((headers, body)) = request.split_once("\r\n\r\n") {
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.to_lowercase()
                            .strip_prefix("content-length: ")
                            .map(str::to_owned)
                    })
                    .unwrap()
                    .parse()
                    .unwrap();
                if body.len() >= length {
                    break;
                }
            }
        }
        let wire = String::from_utf8(received).unwrap();
        assert!(wire.contains("\"bytes\":\"AwQ=\""));
        let body = r#"{"output":{"message":{"content":[{"text":"provider transcript"}]}}}"#;
        socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
    });
    let result = AudioRoute::execute(
        &services,
        AudioRouteRequest {
            model: "bedrock/mistral.voxtral-mini-3b-2507",
            audio: AudioInput {
                data: "AQI=".into(),
                format: AudioFormat::Wav,
                filename: None,
            },
            api_key: None,
            api_base: Some("http://127.0.0.1:1"),
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: serde_json::Map::from_iter([
                ("aws_access_key_id".into(), serde_json::json!("test-key")),
                (
                    "aws_secret_access_key".into(),
                    serde_json::json!("test-secret"),
                ),
                ("aws_region_name".into(), serde_json::json!("us-east-1")),
            ]),
            timeout: Some(std::time::Duration::from_secs(1)),
            request_metadata: Default::default(),
            litellm_call_id: Some("audio-contract"),
        },
    )
    .await
    .into_result();
    if reject.is_some() {
        assert!(matches!(result, Err(Error::InvalidRequest(message)) if message == "rejected"));
    } else {
        assert_eq!(
            result.unwrap(),
            serde_json::json!({"text":"replaced transcript"})
        );
    }
    let terminals = services.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert_eq!(
        matches!(
            terminals[0].classification,
            litellm_core::lifecycle::TerminalClassification::Success
        ),
        reject.is_none()
    );
    if reject.is_none() {
        assert!(
            matches!(&terminals[0].projection, litellm_core::lifecycle::RouteProjection::Audio { value } if value == &serde_json::json!({"text":"replaced transcript"}))
        );
    }
    assert!(
        !services
            .events
            .lock()
            .unwrap()
            .contains(&Operation::DeploymentFailure)
    );
    drop(terminals);
    provider.await.unwrap();
}
