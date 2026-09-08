use crate::Error;
use crate::lifecycle::program::{CallProgram, ProgramOptions, actions_for};
use crate::lifecycle::{ActionBinding, Lifecycle, LifecycleRoute, Outcome};

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
                pre_call: false,
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
