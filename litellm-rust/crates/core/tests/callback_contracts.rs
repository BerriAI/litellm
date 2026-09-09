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

impl<Request: Send> DeploymentPreHooks<Request> for Services {
    fn async_pre_call_deployment_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: Request,
    ) -> CallbackFuture<'a, ActionResult<Request, Error>>
    where
        Request: 'a,
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

impl<Response: Send> DeploymentSuccessHooks<Response> for Services {
    fn async_post_call_success_deployment_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        response: Response,
    ) -> CallbackFuture<'a, ActionResult<Response, Error>>
    where
        Response: 'a,
    {
        Box::pin(async move {
            self.events
                .lock()
                .unwrap()
                .push(Operation::DeploymentSuccess);
            if self.reject == Some(Operation::DeploymentSuccess) {
                return ActionResult::Reject(Error::InvalidRequest("rejected".into()));
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
    let result = CallLifecycle::default()
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
        planned
            .advance(
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
        _: litellm_core::providers::bedrock::aws_base::AwsAuthConfig,
    ) -> litellm_core::providers::bedrock::aws_base::AwsCredentialFuture<'a> {
        Box::pin(async { Err(Error::Auth("unexpected credential lookup".into())) })
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
