use super::{
    ActionBinding, ActionKind, Delivery, ErrorDisposition, FailurePolicy, Outcome, Owner,
    ResultPolicy,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Setup,
    DeploymentPre,
    BuildRequest,
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
}

#[derive(Debug)]
pub struct CallLifecycle {
    operation: Operation,
    outcome: Outcome,
    commitment: Commitment,
    failure_stage: Option<FailureStage>,
    options: ProgramOptions,
}

impl CallLifecycle {
    pub fn planned(options: ProgramOptions) -> Self {
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
            (BuildRequest | PreCall | Send, Outcome::Failure) if self.options.asynchronous => {
                DeploymentFailure
            }
            (_, Outcome::Failure) => failure,
            (Setup, Outcome::Success) if self.options.asynchronous => DeploymentPre,
            (Setup | DeploymentPre, Outcome::Success) => BuildRequest,
            (BuildRequest, Outcome::Success) => PreCall,
            (PreCall, Outcome::Success) => Send,
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

impl Default for CallLifecycle {
    fn default() -> Self {
        Self::planned(ProgramOptions {
            asynchronous: false,
            internal_call: false,
        })
    }
}

impl Operation {
    pub fn is_awaited(self, asynchronous: bool) -> bool {
        matches!(
            self,
            Self::DeploymentPre
                | Self::DeploymentSuccess
                | Self::DeploymentFailure
                | Self::AsyncFailure
        ) || (asynchronous && self == Self::Send)
    }
}

pub fn actions_for(operation: Operation) -> &'static [ActionBinding] {
    match operation {
        Operation::BuildRequest => &REQUEST_BUILD_ACTION,
        Operation::Send => &PROVIDER_ACTION,
        Operation::PreCall => &PRE_CALL_ACTION,
        Operation::SyncFailure | Operation::AsyncFailure | Operation::DeploymentFailure => {
            &FAILURE_ACTION
        }
        Operation::Restore => &RESTORE_ACTION,
        Operation::Complete(_) => &[],
        _ => &CALLBACK_ACTION,
    }
}

const REQUEST_BUILD_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::RequestBuild,
    delivery: Delivery::InlineDirect,
    on_result: ResultPolicy::Replace,
    on_error: FailurePolicy::Propagate,
    owner: Owner::Core,
}];

const PROVIDER_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::ProviderCall,
    delivery: Delivery::InlineAwaited,
    on_result: ResultPolicy::Replace,
    on_error: FailurePolicy::Propagate,
    owner: Owner::Core,
}];

const PRE_CALL_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::PreparedCallLogging,
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
        let mut before = CallLifecycle::planned(ProgramOptions {
            asynchronous: false,
            internal_call: false,
        });
        let failure = before.advance(Outcome::Failure, observations()).unwrap();
        assert_eq!(failure.commitment, Commitment::Replayable);
        assert_eq!(failure.failure_stage, Some(FailureStage::BeforeProvider));

        let mut provider = CallLifecycle::planned(ProgramOptions {
            asynchronous: false,
            internal_call: false,
        });
        provider.advance(Outcome::Success, observations()).unwrap();
        provider.advance(Outcome::Success, observations()).unwrap();
        provider.advance(Outcome::Success, observations()).unwrap();
        let failure = provider.advance(Outcome::Failure, observations()).unwrap();
        assert_eq!(failure.commitment, Commitment::ProviderStarted);
        assert_eq!(failure.failure_stage, Some(FailureStage::ProviderCall));

        let mut after = CallLifecycle::planned(ProgramOptions {
            asynchronous: false,
            internal_call: false,
        });
        after.advance(Outcome::Success, observations()).unwrap();
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

    #[test]
    fn request_build_and_pre_call_failures_are_replayable() {
        for success_count in [1, 2] {
            let mut program = CallLifecycle::planned(ProgramOptions {
                asynchronous: false,
                internal_call: false,
            });
            for _ in 0..success_count {
                program.advance(Outcome::Success, observations()).unwrap();
            }
            let failure = program.advance(Outcome::Failure, observations()).unwrap();
            assert_eq!(failure.commitment, Commitment::Replayable);
            assert_eq!(failure.failure_stage, Some(FailureStage::BeforeProvider));
        }
    }

    #[test]
    fn request_build_has_its_own_action_kind() {
        assert_eq!(
            actions_for(Operation::BuildRequest)[0].kind,
            ActionKind::RequestBuild
        );
        assert_eq!(
            actions_for(Operation::Send)[0].kind,
            ActionKind::ProviderCall
        );
    }
}
