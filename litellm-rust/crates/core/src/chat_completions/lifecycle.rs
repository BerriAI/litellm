use crate::Error;
use crate::integrations::custom_logger::{LogError, LogFuture};
use crate::integrations::types::Usage;
use crate::lifecycle::program::{CallProgram, ProgramOptions, actions_for};
use crate::lifecycle::{
    ActionBinding, ActionResult, CallLifecycle, CallLifecycleContext, Clock, ExecutedCall,
    Lifecycle, LifecycleRoute, ModerationHooks, Outcome, PreCallHooks, SystemClock,
    TerminalDispatcher, TerminalRecord,
};

use super::handler::execute_chat_completions_provider_call_with_transport;
use super::types::SettledChatRequest;
use super::types::{ChatCompletionsResponse, ResolvedChatCompletionsRequest};

use super::chat_completions_decline_reason;

use serde_json::{Map, Value};

pub use crate::lifecycle::program::{Observations, Operation, Transition};

#[derive(Clone, Debug)]
pub struct Admission {
    pub model: String,
    pub messages: Value,
    pub optional_params: Map<String, Value>,
    pub custom_llm_provider: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub asynchronous: bool,
    pub internal_call: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Decline(&'static str);

impl Decline {
    pub fn reason(&self) -> &'static str {
        self.0
    }
}

#[derive(Debug)]
pub struct ChatCompletionsState {
    program: CallProgram,
}

#[derive(Debug)]
pub struct ChatCompletionsRoute;

impl LifecycleRoute for ChatCompletionsRoute {
    type Admission = Admission;
    type Options = Options;
    type Context = Observations;
    type Operation = Operation;
    type Observation = Observations;
    type Outcome = Outcome;
    type Transition = Transition;
    type Error = Error;
    type Decline = Decline;
    type State = ChatCompletionsState;

    fn admit(
        admission: &Admission,
        options: Options,
    ) -> Result<Result<Self::State, Decline>, Error> {
        if let Some(reason) = chat_completions_decline_reason(
            &admission.model,
            admission.custom_llm_provider.as_deref(),
            admission.messages.clone(),
            &admission.optional_params,
        ) {
            return Ok(Err(Decline(reason)));
        }
        Ok(Ok(ChatCompletionsState {
            program: CallProgram::new(ProgramOptions {
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
        state.program.advance(outcome, observations).ok_or_else(|| {
            Error::InvalidRequest("chat completions lifecycle is already complete".into())
        })
    }

    fn actions_for(operation: Operation, _: &Observations) -> &'static [ActionBinding] {
        actions_for(operation)
    }
}

impl Lifecycle<ChatCompletionsRoute> {
    pub fn commitment(&self) -> crate::lifecycle::Commitment {
        self.state.program.commitment()
    }

    pub fn failure_stage(&self) -> Option<crate::lifecycle::FailureStage> {
        self.state.program.failure_stage()
    }
}

pub fn machine(
    admission: &Admission,
    options: Options,
) -> Result<Result<Lifecycle<ChatCompletionsRoute>, Decline>, Error> {
    Lifecycle::admit(admission, options)
}

pub trait ChatCompletionsSession:
    for<'request> PreCallHooks<ResolvedChatCompletionsRequest<'request>>
    + for<'request> ModerationHooks<ResolvedChatCompletionsRequest<'request>>
    + TerminalDispatcher
    + Clock
{
}

impl<T> ChatCompletionsSession for T where
    T: for<'request> PreCallHooks<ResolvedChatCompletionsRequest<'request>>
        + for<'request> ModerationHooks<ResolvedChatCompletionsRequest<'request>>
        + TerminalDispatcher
        + Clock
{
}

pub async fn execute<'request, S, T, A>(
    application: &A,
    transport: &T,
    session: &S,
    request: ResolvedChatCompletionsRequest<'request>,
    context: CallLifecycleContext,
) -> ExecutedCall<ChatCompletionsResponse, Error>
where
    S: ChatCompletionsSession,
    T: crate::runtime::HttpTransport,
    A: crate::providers::auth::ChatAuthorizationServices,
{
    CallLifecycle
        .run_with_usage(
            (context, request),
            session,
            session,
            session,
            |request| async move {
                execute_chat_completions_provider_call_with_transport(
                    application,
                    transport,
                    request,
                )
                .await
            },
            response_usage,
        )
        .await
}

fn response_usage(response: &ChatCompletionsResponse) -> Option<Usage> {
    Some(Usage {
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        total_tokens: response.usage.total_tokens,
    })
}

struct UndispatchedSession;

impl Clock for UndispatchedSession {
    fn now(&self) -> f64 {
        SystemClock.now()
    }
}

impl PreCallHooks<SettledChatRequest> for UndispatchedSession {
    type PreCallFuture<'a> = std::future::Ready<ActionResult<SettledChatRequest, Error>>;
    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: SettledChatRequest,
    ) -> Self::PreCallFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
}

impl ModerationHooks<SettledChatRequest> for UndispatchedSession {
    type ModerationFuture<'a> = std::future::Ready<ActionResult<SettledChatRequest, Error>>;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: SettledChatRequest,
    ) -> Self::ModerationFuture<'a> {
        std::future::ready(ActionResult::Continue(request))
    }
}

impl TerminalDispatcher for UndispatchedSession {
    fn dispatch<'a>(&'a self, _: &'a TerminalRecord) -> LogFuture<'a> {
        Box::pin(async { Ok::<(), LogError>(()) })
    }
}

pub(crate) async fn execute_settled(
    request: SettledChatRequest,
    context: CallLifecycleContext,
) -> ExecutedCall<ChatCompletionsResponse, Error> {
    CallLifecycle
        .run_with_usage(
            (context, request),
            &UndispatchedSession,
            &UndispatchedSession,
            &UndispatchedSession,
            |request| async move { super::handler::execute_settled_request(request).await },
            response_usage,
        )
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admission() -> Admission {
        Admission {
            model: "claude-sonnet-4-5".into(),
            messages: serde_json::json!([{"role": "user", "content": "hi"}]),
            optional_params: Map::from_iter([("max_tokens".into(), Value::from(16))]),
            custom_llm_provider: Some("anthropic".into()),
        }
    }

    #[test]
    fn admission_declines_before_the_lifecycle_starts() {
        let mut unsupported = admission();
        unsupported.messages = serde_json::json!([]);
        assert!(matches!(
            machine(&unsupported, Options::default()),
            Ok(Err(_))
        ));
        assert!(matches!(
            machine(&admission(), Options::default()),
            Ok(Ok(_))
        ));
    }

    #[test]
    fn sync_and_async_success_sequences_are_selected_by_core() {
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
            let mut machine = machine(
                &admission(),
                Options {
                    asynchronous,
                    ..Options::default()
                },
            )
            .unwrap()
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
