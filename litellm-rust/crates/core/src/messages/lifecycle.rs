use std::future::{Future, Ready, ready};
use std::sync::Arc;

use crate::Error;
use crate::integrations::custom_logger::{LogError, LogFuture};
use crate::integrations::types::Usage;
use crate::lifecycle::program::{ProgramOptions, actions_for};
use crate::lifecycle::{
    ActionBinding, ActionResult, CallLifecycle, CallLifecycleContext, Clock,
    DeploymentFailureHooks, DeploymentPreHooks, DeploymentSuccessHooks, ExecutedCall, Lifecycle,
    LifecycleRoute, ModerationHooks, Outcome, PreCallHooks, StreamingCall, StreamingObserver,
    TerminalDispatcher, TerminalRecord,
};

use super::handler::execute_messages_provider_call;
use super::types::{AnthropicMessagesResponse, MessagesRequest};

pub use crate::lifecycle::program::{Observations, Operation, Transition};

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub asynchronous: bool,
    pub internal_call: bool,
    pub call_id: Option<String>,
    pub trace_id: Option<String>,
}

#[derive(Debug)]
pub struct MessagesState {
    program: CallLifecycle,
}

#[derive(Debug)]
pub struct MessagesRoute;

impl LifecycleRoute for MessagesRoute {
    type Admission = ();
    type Options = Options;
    type Context = Observations;
    type Operation = Operation;
    type Observation = Observations;
    type Outcome = Outcome;
    type Transition = Transition;
    type Error = Error;
    type Decline = std::convert::Infallible;
    type State = MessagesState;

    fn admit(_: &(), options: Options) -> Result<Result<Self::State, Self::Decline>, Error> {
        Ok(Ok(MessagesState {
            program: CallLifecycle::planned(ProgramOptions {
                asynchronous: options.asynchronous,
                internal_call: options.internal_call,
            }),
        }))
    }

    fn operation(state: &Self::State) -> Operation {
        state.program.operation()
    }

    fn advance(
        state: &mut Self::State,
        outcome: Outcome,
        observations: Observations,
    ) -> Result<Transition, Error> {
        state
            .program
            .advance(outcome, observations)
            .ok_or_else(|| Error::InvalidRequest("messages lifecycle is already complete".into()))
    }

    fn actions_for(operation: Operation, _: &Observations) -> &'static [ActionBinding] {
        actions_for(operation)
    }
}

impl Lifecycle<MessagesRoute> {
    pub fn commitment(&self) -> crate::lifecycle::Commitment {
        self.state.program.commitment()
    }

    pub fn failure_stage(&self) -> Option<crate::lifecycle::FailureStage> {
        self.state.program.failure_stage()
    }
}

#[cfg(test)]
mod program_tests {
    use super::*;

    #[test]
    fn sync_and_async_sequences_build_then_run_pre_call() {
        for (asynchronous, expected) in [
            (
                false,
                vec![
                    Operation::Setup,
                    Operation::BuildRequest,
                    Operation::PreCall,
                    Operation::Send,
                    Operation::SyncSuccess,
                    Operation::Restore,
                ],
            ),
            (
                true,
                vec![
                    Operation::Setup,
                    Operation::DeploymentPre,
                    Operation::BuildRequest,
                    Operation::PreCall,
                    Operation::Send,
                    Operation::DeploymentSuccess,
                    Operation::AsyncSuccess,
                    Operation::SyncSuccessIfNeeded,
                    Operation::Restore,
                ],
            ),
        ] {
            let mut machine = machine(Options {
                asynchronous,
                ..Options::default()
            })
            .unwrap();
            for operation in expected {
                assert_eq!(machine.operation(), operation);
                machine
                    .advance(
                        Outcome::Success,
                        Observations {
                            logger_available: true,
                            has_fallbacks: false,
                        },
                    )
                    .unwrap();
            }
            assert_eq!(machine.operation(), Operation::Complete(Outcome::Success));
        }
    }
}

pub trait MessagesServices:
    PreCallHooks<MessagesRequest>
    + ModerationHooks<MessagesRequest>
    + DeploymentPreHooks<MessagesRequest>
    + DeploymentSuccessHooks<AnthropicMessagesResponse>
    + DeploymentFailureHooks
    + TerminalDispatcher
    + Clock
{
}

impl<T> MessagesServices for T where
    T: PreCallHooks<MessagesRequest>
        + ModerationHooks<MessagesRequest>
        + DeploymentPreHooks<MessagesRequest>
        + DeploymentSuccessHooks<AnthropicMessagesResponse>
        + DeploymentFailureHooks
        + TerminalDispatcher
        + Clock
{
}

#[derive(Default)]
pub struct NoopServices;

impl Clock for NoopServices {
    fn now(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or(0.0)
    }
}

impl PreCallHooks<MessagesRequest> for NoopServices {
    type PreCallFuture<'a> = Ready<ActionResult<MessagesRequest, Error>>;
    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

impl ModerationHooks<MessagesRequest> for NoopServices {
    type ModerationFuture<'a> = Ready<ActionResult<MessagesRequest, Error>>;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::ModerationFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

impl DeploymentPreHooks<MessagesRequest> for NoopServices {}
impl DeploymentSuccessHooks<AnthropicMessagesResponse> for NoopServices {}
impl DeploymentFailureHooks for NoopServices {}

impl TerminalDispatcher for NoopServices {
    fn dispatch<'a>(&'a self, _: &'a TerminalRecord) -> LogFuture<'a> {
        Box::pin(async { Ok::<(), LogError>(()) })
    }
}

pub async fn messages<S: MessagesServices>(
    services: &S,
    request: MessagesRequest,
    _options: Options,
    context: CallLifecycleContext,
) -> ExecutedCall<AnthropicMessagesResponse, Error> {
    CallLifecycle::default()
        .run_with_usage(
            (context, request),
            services,
            services,
            services,
            |request| async move { execute_messages_provider_call(request).await },
            anthropic_response_usage,
        )
        .await
}

pub(crate) async fn messages_with_provider<S, ProviderCall, ProviderFuture>(
    services: &S,
    request: MessagesRequest,
    context: CallLifecycleContext,
    provider_call: ProviderCall,
) -> ExecutedCall<AnthropicMessagesResponse, Error>
where
    S: MessagesServices,
    ProviderCall: FnOnce(MessagesRequest) -> ProviderFuture,
    ProviderFuture: Future<Output = Result<AnthropicMessagesResponse, Error>>,
{
    CallLifecycle::default()
        .run_with_usage(
            (context, request),
            services,
            services,
            services,
            provider_call,
            anthropic_response_usage,
        )
        .await
}

fn anthropic_response_usage(response: &AnthropicMessagesResponse) -> Option<Usage> {
    let usage = response.usage.as_ref()?;
    let prompt_tokens = usage.get("input_tokens")?.as_u64()?;
    let completion_tokens = usage.get("output_tokens")?.as_u64()?;
    Some(Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    })
}

pub(crate) async fn messages_stream_with<S, ProviderCall, ProviderFuture>(
    services: Arc<S>,
    request: MessagesRequest,
    _options: Options,
    context: CallLifecycleContext,
    provider_call: ProviderCall,
) -> Result<StreamingCall, Error>
where
    S: MessagesServices + 'static,
    ProviderCall: FnOnce(MessagesRequest) -> ProviderFuture,
    ProviderFuture: Future<Output = Result<crate::lifecycle::StreamingSource, Error>>,
{
    CallLifecycle::default()
        .run_streaming(
            context,
            request,
            services,
            Box::<AnthropicUsageObserver>::default(),
            provider_call,
        )
        .await
}

#[derive(Default)]
struct AnthropicUsageObserver {
    pending: Vec<u8>,
    usage: Usage,
}

impl AnthropicUsageObserver {
    fn observe_event(&mut self, event: &[u8]) {
        let Some(data) = event
            .split(|byte| *byte == b'\n')
            .find_map(|line| line.strip_prefix(b"data:"))
        else {
            return;
        };
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(data.trim_ascii_start()) else {
            return;
        };
        if !matches!(
            value.get("type").and_then(serde_json::Value::as_str),
            Some("message_start" | "message_delta")
        ) {
            return;
        }
        let Some(usage) = value.get("usage").or_else(|| {
            value
                .get("message")
                .and_then(|message| message.get("usage"))
        }) else {
            return;
        };
        if let Some(input_tokens) = usage
            .get("input_tokens")
            .and_then(serde_json::Value::as_u64)
        {
            self.usage.prompt_tokens = input_tokens;
        }
        if let Some(output_tokens) = usage
            .get("output_tokens")
            .and_then(serde_json::Value::as_u64)
        {
            self.usage.completion_tokens = output_tokens;
        }
        self.usage.total_tokens = self.usage.prompt_tokens + self.usage.completion_tokens;
    }
}

impl StreamingObserver for AnthropicUsageObserver {
    fn observe(&mut self, bytes: &[u8]) {
        self.pending.extend_from_slice(bytes);
        while let Some(end) = self.pending.windows(2).position(|window| window == b"\n\n") {
            let event = self.pending.drain(..end + 2).collect::<Vec<_>>();
            self.observe_event(&event);
        }
    }

    fn usage(&self) -> Usage {
        self.usage
    }

    fn projection(&self) -> serde_json::Value {
        serde_json::json!({"stream": true})
    }
}

pub fn machine(options: Options) -> Result<Lifecycle<MessagesRoute>, Error> {
    Lifecycle::admit(&(), options).map(|result| match result {
        Ok(machine) => machine,
        Err(never) => match never {},
    })
}
