use crate::Error;
use crate::integrations::types::Usage;
use crate::lifecycle::program::ProgramOptions;
use crate::lifecycle::{
    CallLifecycle, CallLifecycleContext, Clock, DeploymentFailureHooks, DeploymentPreHooks,
    DeploymentSuccessHooks, ExecutedCall, Lifecycle, LifecycleRoute, ModerationHooks, PreCallHooks,
    TerminalDispatcher,
};

use super::handler::execute_chat_completions_provider_call_with_transport;
use super::types::SettledChatRequest;
use super::types::{
    ChatCompletionsRequest, ChatCompletionsResponse, ResolvedChatCompletionsRequest,
};

use super::chat_completions_decline_reason;

use serde_json::{Map, Value};

pub use crate::lifecycle::program::{Observations, Operation, Transition};

#[derive(Clone, Debug)]
pub struct Admission {
    pub model: String,
    pub messages: Vec<super::types::ChatMessage>,
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
    program: CallLifecycle,
}

#[derive(Debug)]
pub struct ChatCompletionsRoute;

impl crate::lifecycle::machine::sealed::Sealed for ChatCompletionsRoute {}

impl LifecycleRoute for ChatCompletionsRoute {
    type Admission = Admission;
    type Options = Options;
    type Decline = Decline;
    type State = ChatCompletionsState;

    fn program(state: &Self::State) -> &CallLifecycle {
        &state.program
    }
    fn program_mut(state: &mut Self::State) -> &mut CallLifecycle {
        &mut state.program
    }

    fn admit(
        admission: &Admission,
        options: Options,
    ) -> Result<Result<Self::State, Decline>, Error> {
        if let Some(reason) = chat_completions_decline_reason(
            &admission.model,
            admission.custom_llm_provider.as_deref(),
            &admission.messages,
            &admission.optional_params,
        ) {
            return Ok(Err(Decline(reason)));
        }
        Ok(Ok(ChatCompletionsState {
            program: CallLifecycle::planned(ProgramOptions {
                asynchronous: options.asynchronous,
                internal_call: options.internal_call,
            }),
        }))
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
    for<'request> PreCallHooks<ChatCompletionsRequest<'request>>
    + for<'request> ModerationHooks<ResolvedChatCompletionsRequest<'request>>
    + for<'request> DeploymentPreHooks<ChatCompletionsRequest<'request>>
    + DeploymentSuccessHooks<ChatCompletionsResponse>
    + DeploymentFailureHooks
    + TerminalDispatcher
    + Clock
{
}

impl<T> ChatCompletionsSession for T where
    T: for<'request> PreCallHooks<ChatCompletionsRequest<'request>>
        + for<'request> ModerationHooks<ResolvedChatCompletionsRequest<'request>>
        + for<'request> DeploymentPreHooks<ChatCompletionsRequest<'request>>
        + DeploymentSuccessHooks<ChatCompletionsResponse>
        + DeploymentFailureHooks
        + TerminalDispatcher
        + Clock
{
}

pub async fn execute<'request, S, T, A>(
    application: &A,
    transport: &T,
    session: &S,
    request: ChatCompletionsRequest<'request>,
    context: CallLifecycleContext,
) -> ExecutedCall<ChatCompletionsResponse, Error>
where
    S: ChatCompletionsSession,
    T: crate::runtime::HttpTransport,
    A: crate::providers::auth::ChatAuthorizationServices,
{
    CallLifecycle::asynchronous()
        .run_prepared_with_usage(
            (context, request),
            (session, session, session),
            |request| std::future::ready(super::request::resolve_request(request)),
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

pub(crate) async fn execute_settled(
    request: SettledChatRequest,
    context: CallLifecycleContext,
) -> ExecutedCall<ChatCompletionsResponse, Error> {
    let start_time = crate::lifecycle::SystemClock.now();
    let result = super::handler::execute_settled_request(request).await;
    let mut context = context;
    if let Ok(response) = &result
        && let Some(usage) = response_usage(response)
    {
        context.usage = usage;
        context.provider_usage = crate::lifecycle::terminal::UsageObservation::Final(usage);
    }
    crate::lifecycle::execution::provider_result(context, start_time, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::Outcome;

    fn admission() -> Admission {
        Admission {
            model: "claude-sonnet-4-5".into(),
            messages: serde_json::from_value(serde_json::json!([
                {"role": "user", "content": "hi"}
            ]))
            .expect("messages"),
            optional_params: Map::from_iter([("max_tokens".into(), Value::from(16))]),
            custom_llm_provider: Some("anthropic".into()),
        }
    }

    #[test]
    fn admission_declines_before_the_lifecycle_starts() {
        let mut unsupported = admission();
        unsupported.messages = Vec::new();
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
