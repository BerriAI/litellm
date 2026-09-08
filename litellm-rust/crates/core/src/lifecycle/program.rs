use super::{
    ActionBinding, ActionKind, Delivery, ErrorDisposition, FailurePolicy, Outcome, Owner,
    ResultPolicy,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Setup,
    DeploymentPre,
    Prepare,
    PreCall,
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

#[derive(Clone, Copy, Debug, Default)]
pub struct Observations {
    pub logger_available: bool,
    pub has_fallbacks: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Commitment {
    Replayable,
    ProviderStarted,
    ResponseReceived,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureStage {
    BeforeProvider,
    ProviderCall,
    AfterProviderResponse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub operation: Operation,
    pub error: ErrorDisposition,
    pub commitment: Commitment,
    pub failure_stage: Option<FailureStage>,
}

#[derive(Clone, Copy, Debug)]
pub struct ProgramOptions {
    pub asynchronous: bool,
    pub internal_call: bool,
    pub pre_call: bool,
}

#[derive(Debug)]
pub struct CallProgram {
    operation: Operation,
    outcome: Outcome,
    commitment: Commitment,
    failure_stage: Option<FailureStage>,
    options: ProgramOptions,
}

impl CallProgram {
    pub fn new(options: ProgramOptions) -> Self {
        Self {
            operation: Operation::Setup,
            outcome: Outcome::Success,
            commitment: Commitment::Replayable,
            failure_stage: None,
            options,
        }
    }

    pub fn operation(&self) -> Operation {
        self.operation
    }

    pub fn commitment(&self) -> Commitment {
        self.commitment
    }

    pub fn failure_stage(&self) -> Option<FailureStage> {
        self.failure_stage
    }

    pub fn advance(&mut self, outcome: Outcome, observations: Observations) -> Option<Transition> {
        use Operation::*;

        if matches!(self.operation, Complete(_)) {
            return None;
        }
        let current = self.operation;
        let failure = if observations.logger_available
            && !(self.options.asynchronous && self.options.internal_call)
        {
            SyncFailure
        } else {
            Restore
        };
        let error = if outcome != Outcome::Success && current != DeploymentFailure {
            self.outcome = outcome;
            ErrorDisposition::Replace
        } else {
            ErrorDisposition::Preserve
        };
        let failure_stage =
            (outcome != Outcome::Success).then(|| self.classify_failure_stage(current));
        if error == ErrorDisposition::Replace {
            self.failure_stage = failure_stage;
        }
        if current == Send {
            self.commitment = if outcome == Outcome::Success {
                Commitment::ResponseReceived
            } else {
                Commitment::ProviderStarted
            };
        }
        self.operation = match (current, outcome) {
            (Restore, _) => Complete(self.outcome),
            (DeploymentFailure, _) => failure,
            (_, Outcome::Abort) => Restore,
            (SyncFailure | AsyncFailure, Outcome::Failure) => Restore,
            (Prepare | PreCall | Send, Outcome::Failure) if self.options.asynchronous => {
                DeploymentFailure
            }
            (_, Outcome::Failure) => failure,
            (Setup, Outcome::Success) if self.options.asynchronous => DeploymentPre,
            (Setup | DeploymentPre, Outcome::Success) => Prepare,
            (Prepare, Outcome::Success) if self.options.pre_call => PreCall,
            (Prepare | PreCall, Outcome::Success) => Send,
            (Send, Outcome::Success) if self.options.asynchronous => DeploymentSuccess,
            (Send, Outcome::Success) => SyncSuccess,
            (DeploymentSuccess, Outcome::Success) => {
                if self.options.internal_call || observations.has_fallbacks {
                    SyncSuccessIfNeeded
                } else {
                    AsyncSuccess
                }
            }
            (AsyncSuccess, Outcome::Success) => SyncSuccessIfNeeded,
            (SyncFailure, Outcome::Success) if self.options.asynchronous => AsyncFailure,
            (SyncSuccess | SyncSuccessIfNeeded | SyncFailure | AsyncFailure, Outcome::Success) => {
                Restore
            }
            (Complete(_), _) => unreachable!(),
        };
        Some(Transition {
            operation: self.operation,
            error,
            commitment: self.commitment,
            failure_stage: self.failure_stage,
        })
    }

    fn classify_failure_stage(&self, operation: Operation) -> FailureStage {
        match (self.commitment, operation) {
            (Commitment::Replayable, Operation::Send) => FailureStage::ProviderCall,
            (Commitment::Replayable, _) => FailureStage::BeforeProvider,
            (Commitment::ProviderStarted, _) => FailureStage::ProviderCall,
            (Commitment::ResponseReceived, _) => FailureStage::AfterProviderResponse,
        }
    }
}

pub fn actions_for(operation: Operation) -> &'static [ActionBinding] {
    match operation {
        Operation::Prepare | Operation::Send => &PROVIDER_ACTION,
        Operation::PreCall => &PRE_CALL_ACTION,
        Operation::SyncFailure | Operation::AsyncFailure | Operation::DeploymentFailure => {
            &FAILURE_ACTION
        }
        Operation::Restore => &RESTORE_ACTION,
        Operation::Complete(_) => &[],
        _ => &CALLBACK_ACTION,
    }
}

const PROVIDER_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::ProviderCall,
    delivery: Delivery::InlineAwaited,
    on_result: ResultPolicy::Replace,
    on_error: FailurePolicy::Propagate,
    owner: Owner::Core,
}];

const PRE_CALL_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::RequestPolicy,
    delivery: Delivery::InlineDirect,
    on_result: ResultPolicy::Continue,
    on_error: FailurePolicy::Propagate,
    owner: Owner::Route,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn observations() -> Observations {
        Observations {
            logger_available: true,
            has_fallbacks: false,
        }
    }

    #[test]
    fn commitment_classifies_failures_without_host_inference() {
        let mut before = CallProgram::new(ProgramOptions {
            asynchronous: false,
            internal_call: false,
            pre_call: false,
        });
        let failure = before.advance(Outcome::Failure, observations()).unwrap();
        assert_eq!(failure.commitment, Commitment::Replayable);
        assert_eq!(failure.failure_stage, Some(FailureStage::BeforeProvider));

        let mut provider = CallProgram::new(ProgramOptions {
            asynchronous: false,
            internal_call: false,
            pre_call: false,
        });
        provider.advance(Outcome::Success, observations()).unwrap();
        provider.advance(Outcome::Success, observations()).unwrap();
        let failure = provider.advance(Outcome::Failure, observations()).unwrap();
        assert_eq!(failure.commitment, Commitment::ProviderStarted);
        assert_eq!(failure.failure_stage, Some(FailureStage::ProviderCall));

        let mut after = CallProgram::new(ProgramOptions {
            asynchronous: false,
            internal_call: false,
            pre_call: false,
        });
        after.advance(Outcome::Success, observations()).unwrap();
        after.advance(Outcome::Success, observations()).unwrap();
        after.advance(Outcome::Success, observations()).unwrap();
        let failure = after.advance(Outcome::Failure, observations()).unwrap();
        assert_eq!(failure.commitment, Commitment::ResponseReceived);
        assert_eq!(
            failure.failure_stage,
            Some(FailureStage::AfterProviderResponse)
        );
    }
}
