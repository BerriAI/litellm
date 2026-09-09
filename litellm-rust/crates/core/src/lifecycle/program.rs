use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    ActionBinding, ActionKind, Delivery, ErrorDisposition, FailurePolicy, Outcome, Owner,
    ResultPolicy,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Setup,
    DeploymentPre,
    InputHooks,
    BuildRequest,
    PreCall,
    Send,
    StreamComplete,
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
pub struct OperationTicket {
    owner: u64,
    generation: u64,
    operation: Operation,
}

impl OperationTicket {
    pub fn operation(&self) -> Operation { self.operation }
}

#[derive(Debug)]
pub struct ProviderPermit {
    _private: (),
}

#[derive(Debug)]
pub struct CallLifecycle {
    owner: u64,
    generation: u64,
    issued: bool,
    provider_issued: bool,
    operation: Operation,
    outcome: Outcome,
    commitment: Commitment,
    failure_stage: Option<FailureStage>,
    options: ProgramOptions,
}

impl CallLifecycle {
    pub fn planned(options: ProgramOptions) -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        Self {
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            generation: 0,
            issued: false,
            provider_issued: false,
            operation: Operation::Setup,
            outcome: Outcome::Success,
            commitment: Commitment::Replayable,
            failure_stage: None,
            options,
        }
    }

    pub fn asynchronous() -> Self {
        Self::planned(ProgramOptions {
            asynchronous: true,
            internal_call: false,
        })
    }

    pub fn operation(&self) -> Operation {
        self.operation
    }

    pub fn issue(&mut self) -> Result<OperationTicket, crate::Error> {
        if self.issued || matches!(self.operation, Operation::Complete(_)) {
            return Err(crate::Error::InvalidRequest("lifecycle operation is already issued or complete".into()));
        }
        self.issued = true;
        Ok(OperationTicket { owner: self.owner, generation: self.generation, operation: self.operation })
    }

    fn owns(&self, ticket: &OperationTicket) -> bool {
        self.issued && ticket.owner == self.owner && ticket.generation == self.generation && ticket.operation == self.operation
    }

    pub fn complete_operation(&mut self, ticket: OperationTicket, outcome: Outcome, observations: Observations) -> Result<Transition, crate::Error> {
        if !self.owns(&ticket) {
            return Err(crate::Error::InvalidRequest("stale or foreign lifecycle operation".into()));
        }
        self.issued = false;
        self.advance(outcome, observations).ok_or_else(|| crate::Error::InvalidRequest("lifecycle is complete".into()))
    }

    pub fn provider_permit(&mut self, ticket: &OperationTicket) -> Result<ProviderPermit, crate::Error> {
        if !self.owns(ticket) || ticket.operation != Operation::Send || self.provider_issued {
            return Err(crate::Error::InvalidRequest("provider execution requires the current send operation".into()));
        }
        self.provider_issued = true;
        self.begin_provider();
        Ok(ProviderPermit { _private: () })
    }

    pub(crate) fn begin_provider(&mut self) {
        self.commitment = Commitment::ProviderStarted;
    }

    pub(crate) fn transfer_stream(&mut self) {
        assert_eq!(self.operation, Operation::Send);
        self.commitment = Commitment::ProviderStarted;
        self.operation = Operation::StreamComplete;
    }

    pub fn commitment(&self) -> Commitment {
        self.commitment
    }

    pub fn failure_stage(&self) -> Option<FailureStage> {
        self.failure_stage
    }

    pub(crate) fn advance(&mut self, outcome: Outcome, observations: Observations) -> Option<Transition> {
        use Operation::*;

        if matches!(self.operation, Complete(_)) {
            return None;
        }
        self.generation += 1;
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
        if matches!(current, Send | StreamComplete) {
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
            (operation, Outcome::Failure)
                if self.options.asynchronous && operation.notifies_deployment_failure() =>
            {
                DeploymentFailure
            }
            (_, Outcome::Failure) => failure,
            (Setup, Outcome::Success) if self.options.asynchronous => DeploymentPre,
            (Setup | DeploymentPre, Outcome::Success) => InputHooks,
            (InputHooks, Outcome::Success) => BuildRequest,
            (BuildRequest, Outcome::Success) => PreCall,
            (PreCall, Outcome::Success) => Send,
            (Send, Outcome::Success) if self.options.asynchronous => DeploymentSuccess,
            (Send, Outcome::Success) => SyncSuccess,
            (StreamComplete, Outcome::Success) if !self.options.asynchronous => SyncSuccess,
            (StreamComplete, Outcome::Success)
                if self.options.internal_call || observations.has_fallbacks =>
            {
                SyncSuccessIfNeeded
            }
            (StreamComplete, Outcome::Success) => AsyncSuccess,
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

    pub(crate) fn delivers_terminal(&self, observations: Observations) -> bool {
        observations.logger_available
            && match self.operation {
                Operation::SyncSuccess | Operation::AsyncSuccess | Operation::SyncFailure => true,
                Operation::SyncSuccessIfNeeded => {
                    self.options.internal_call || observations.has_fallbacks
                }
                _ => false,
            }
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
    pub fn notifies_deployment_failure(self) -> bool {
        matches!(
            self,
            Self::InputHooks | Self::BuildRequest | Self::PreCall | Self::Send
        )
    }

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
        Operation::InputHooks => &INPUT_HOOK_ACTION,
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

const INPUT_HOOK_ACTION: [ActionBinding; 1] = [ActionBinding {
    kind: ActionKind::InputHooks,
    delivery: Delivery::InlineAwaited,
    on_result: ResultPolicy::Replace,
    on_error: FailurePolicy::Propagate,
    owner: Owner::Core,
}];

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
        for success_count in [1, 2, 3] {
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
