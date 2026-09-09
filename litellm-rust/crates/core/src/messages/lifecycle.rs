use crate::messages::types::ProviderMessagesRequest;
use std::future::{Future, Ready, ready};
use std::sync::Arc;

use crate::Error;
use crate::integrations::custom_logger::{LogError, LogFuture};
use crate::integrations::types::Usage;
use crate::lifecycle::program::{ProgramOptions, actions_for};
use crate::lifecycle::{
    ActionBinding, ActionResult, CallLifecycle, CallLifecycleContext, Clock,
    DeploymentFailureHooks, DeploymentPreHooks, DeploymentSuccessHooks, ExecutedCall, Lifecycle,
    LifecycleRoute, ModerationHooks, Outcome, PreCallHooks, StreamingCall, TerminalDispatcher,
    TerminalRecord,
};

use super::handler::execute_provider_messages_request;
use super::request::build_provider_request;
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

    fn program(state: &Self::State) -> &CallLifecycle { &state.program }
    fn program_mut(state: &mut Self::State) -> &mut CallLifecycle { &mut state.program }

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
                    Operation::InputHooks,
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
                    Operation::InputHooks,
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
    + ModerationHooks<ProviderMessagesRequest>
    + DeploymentPreHooks<MessagesRequest>
    + DeploymentSuccessHooks<AnthropicMessagesResponse>
    + DeploymentFailureHooks
    + TerminalDispatcher
    + Clock
{
}

impl<T> MessagesServices for T where
    T: PreCallHooks<MessagesRequest>
        + ModerationHooks<ProviderMessagesRequest>
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

impl ModerationHooks<ProviderMessagesRequest> for NoopServices {
    type ModerationFuture<'a> = Ready<ActionResult<ProviderMessagesRequest, Error>>;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: ProviderMessagesRequest,
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

impl Options {
    pub(crate) fn context(&self, mut context: CallLifecycleContext) -> CallLifecycleContext {
        if let Some(call_id) = &self.call_id {
            context.litellm_call_id = call_id.clone();
        }
        if let Some(trace_id) = &self.trace_id {
            context.trace_id = Some(trace_id.clone());
        }
        context
    }

    fn apply(self, context: CallLifecycleContext) -> (CallLifecycle, CallLifecycleContext) {
        let context = self.context(context);
        (
            CallLifecycle::planned(ProgramOptions {
                asynchronous: self.asynchronous,
                internal_call: self.internal_call,
            }),
            context,
        )
    }
}

pub async fn messages<S: MessagesServices>(
    services: &S,
    request: MessagesRequest,
    options: Options,
    context: CallLifecycleContext,
) -> ExecutedCall<AnthropicMessagesResponse, Error> {
    messages_with_provider(
        services,
        request,
        options,
        context,
        |request| std::future::ready(build_provider_request(request)),
        execute_provider_messages_request,
    )
    .await
}

pub(crate) async fn messages_with_provider<
    S,
    Prepare,
    PrepareFuture,
    ProviderCall,
    ProviderFuture,
>(
    services: &S,
    request: MessagesRequest,
    options: Options,
    context: CallLifecycleContext,
    prepare: Prepare,
    provider_call: ProviderCall,
) -> ExecutedCall<AnthropicMessagesResponse, Error>
where
    S: MessagesServices,
    Prepare: FnOnce(MessagesRequest) -> PrepareFuture + Send,
    PrepareFuture: Future<Output = Result<ProviderMessagesRequest, Error>> + Send,
    ProviderCall: FnOnce(ProviderMessagesRequest) -> ProviderFuture + Send,
    ProviderFuture: Future<Output = Result<AnthropicMessagesResponse, Error>> + Send,
{
    let (program, context) = options.apply(context);
    program
        .run_prepared_with_usage(
            (context, request),
            services,
            services,
            services,
            prepare,
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

pub(crate) async fn messages_stream_with<S, Prepare, PrepareFuture, ProviderCall, ProviderFuture>(
    services: Arc<S>,
    request: MessagesRequest,
    options: Options,
    context: CallLifecycleContext,
    prepare: Prepare,
    provider_call: ProviderCall,
) -> Result<StreamingCall, Error>
where
    S: MessagesServices + crate::lifecycle::StreamDrain + 'static,
    Prepare: FnOnce(MessagesRequest) -> PrepareFuture + Send,
    PrepareFuture: Future<Output = Result<ProviderMessagesRequest, Error>> + Send,
    ProviderCall: FnOnce(ProviderMessagesRequest) -> ProviderFuture + Send,
    ProviderFuture: Future<Output = Result<crate::lifecycle::StreamingSource, Error>> + Send,
{
    let (program, context) = options.apply(context);
    program
        .run_streaming_prepared(
            context,
            request,
            services,
            Box::new(crate::sse::SseObserver::new(
                super::streaming::AnthropicMessagesObserver::default(),
            )),
            prepare,
            provider_call,
        )
        .await
}

pub fn machine(options: Options) -> Result<Lifecycle<MessagesRoute>, Error> {
    Lifecycle::admit(&(), options).map(|result| match result {
        Ok(machine) => machine,
        Err(never) => match never {},
    })
}
