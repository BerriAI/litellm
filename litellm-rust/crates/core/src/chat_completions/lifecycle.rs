use crate::Error;
use crate::lifecycle::{
    ActionBinding, ActionKind, Delivery, ErrorDisposition, FailurePolicy, Lifecycle,
    LifecycleRoute, Outcome, Owner, ResultPolicy,
};

use super::chat_completions_decline_reason;

use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Setup,
    DeploymentPre,
    Prepare,
    Send,
    DeploymentSuccess,
    DeploymentFailure,
    SyncSuccess,
    AsyncSuccess,
    SyncSuccessIfNeeded,
    SyncFailure,
    AsyncFailure,
    Restore,
    Complete(Outcome),
}

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

#[derive(Clone, Copy, Debug, Default)]
pub struct Observations {
    pub logger_available: bool,
    pub has_fallbacks: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub error: ErrorDisposition,
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
    operation: Operation,
    outcome: Outcome,
    asynchronous: bool,
    internal_call: bool,
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
            operation: Operation::Setup,
            outcome: Outcome::Success,
            asynchronous: options.asynchronous,
            internal_call: options.internal_call,
        }))
    }

    fn operation(state: &Self::State) -> Operation {
        state.operation
    }

    fn advance(
        state: &mut Self::State,
        outcome: Outcome,
        observations: Observations,
    ) -> Result<Transition, Error> {
        use Operation::*;

        if matches!(state.operation, Complete(_)) {
            return Err(Error::InvalidRequest(
                "chat completions lifecycle is already complete".into(),
            ));
        }
        let failure =
            if observations.logger_available && !(state.asynchronous && state.internal_call) {
                SyncFailure
            } else {
                Restore
            };
        let error = if outcome != Outcome::Success && state.operation != DeploymentFailure {
            state.outcome = outcome;
            ErrorDisposition::Replace
        } else {
            ErrorDisposition::Preserve
        };
        state.operation = match (state.operation, outcome) {
            (Restore, _) => Complete(state.outcome),
            (DeploymentFailure, _) => failure,
            (_, Outcome::Abort) => Restore,
            (SyncFailure | AsyncFailure, Outcome::Failure) => Restore,
            (Prepare | Send, Outcome::Failure) if state.asynchronous => DeploymentFailure,
            (_, Outcome::Failure) => failure,
            (Setup, Outcome::Success) if state.asynchronous => DeploymentPre,
            (Setup | DeploymentPre, Outcome::Success) => Prepare,
            (Prepare, Outcome::Success) => Send,
            (Send, Outcome::Success) if state.asynchronous => DeploymentSuccess,
            (Send, Outcome::Success) => SyncSuccess,
            (DeploymentSuccess, Outcome::Success) => {
                if state.internal_call || observations.has_fallbacks {
                    SyncSuccessIfNeeded
                } else {
                    AsyncSuccess
                }
            }
            (AsyncSuccess, Outcome::Success) => SyncSuccessIfNeeded,
            (SyncFailure, Outcome::Success) if state.asynchronous => AsyncFailure,
            (SyncSuccess | SyncSuccessIfNeeded | SyncFailure | AsyncFailure, Outcome::Success) => {
                Restore
            }
            (Complete(_), _) => unreachable!(),
        };
        Ok(Transition { error })
    }

    fn actions_for(operation: Operation, _: &Observations) -> &'static [ActionBinding] {
        match operation {
            Operation::Prepare | Operation::Send => &PROVIDER_ACTION,
            Operation::SyncFailure | Operation::AsyncFailure | Operation::DeploymentFailure => {
                &FAILURE_ACTION
            }
            Operation::Restore => &RESTORE_ACTION,
            Operation::Complete(_) => &[],
            _ => &CALLBACK_ACTION,
        }
    }
}

const PROVIDER_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::ProviderCall,
    delivery: Delivery::InlineAwaited,
    on_result: ResultPolicy::Replace,
    on_error: FailurePolicy::Propagate,
    owner: Owner::Core,
}];
const CALLBACK_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::TerminalSuccess,
    delivery: Delivery::InlineAwaited,
    on_result: ResultPolicy::Continue,
    on_error: FailurePolicy::RecordAndContinue,
    owner: Owner::Route,
}];
const FAILURE_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::TerminalFailure,
    delivery: Delivery::InlineAwaited,
    on_result: ResultPolicy::Continue,
    on_error: FailurePolicy::PreserveOriginalFailure,
    owner: Owner::Route,
}];
const RESTORE_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::Restore,
    delivery: Delivery::InlineDirect,
    on_result: ResultPolicy::Continue,
    on_error: FailurePolicy::Propagate,
    owner: Owner::Core,
}];

pub fn machine(
    admission: &Admission,
    options: Options,
) -> Result<Result<Lifecycle<ChatCompletionsRoute>, Decline>, Error> {
    Lifecycle::admit(admission, options)
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
                    Operation::Prepare,
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
                    Operation::Prepare,
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
