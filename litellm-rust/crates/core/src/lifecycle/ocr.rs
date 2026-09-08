use crate::Error;
use crate::ocr::{OcrRequest, prepare};

use super::{
    ActionBinding, ActionKind, Delivery, ErrorDisposition, FailurePolicy, LifecycleRoute, Outcome,
    Owner, ResultPolicy,
};

#[derive(Debug)]
pub enum NativeOutcome<T> {
    Completed(T),
    Declined(Decline),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Decline(&'static str);

impl Decline {
    pub fn reason(&self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CredentialMethod {
    #[default]
    Configured,
    Acquisition,
}

#[derive(Default)]
pub struct Options {
    pub asynchronous: bool,
    pub internal_call: bool,
    pub call_id: Option<String>,
    pub trace_id: Option<String>,
    pub credential_method: CredentialMethod,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Identity {
    pub requested_model: String,
    pub call_id: String,
    pub trace_id: Option<String>,
    pub generated_call_id: bool,
}

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

#[derive(Clone, Copy, Debug, Default)]
pub struct Observations {
    pub logger_available: bool,
    pub has_fallbacks: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub operation: Operation,
    pub error: ErrorDisposition,
}

#[derive(Debug)]
pub struct OcrState {
    operation: Operation,
    outcome: Outcome,
    asynchronous: bool,
    internal_call: bool,
    identity: Identity,
}

#[derive(Debug)]
pub struct OcrRoute;

pub type Lifecycle = super::Lifecycle<OcrRoute>;

impl Lifecycle {
    pub fn new(admission: &OcrRequest, options: Options) -> Result<NativeOutcome<Self>, Error> {
        <super::Lifecycle<OcrRoute>>::admit(admission, options).map(|admission| match admission {
            Ok(lifecycle) => NativeOutcome::Completed(lifecycle),
            Err(decline) => NativeOutcome::Declined(decline),
        })
    }

    pub fn identity(&self) -> &Identity {
        &self.state.identity
    }
}

impl LifecycleRoute for OcrRoute {
    type Admission = OcrRequest;
    type Options = Options;
    type Context = Observations;
    type Operation = Operation;
    type Observation = Observations;
    type Outcome = Outcome;
    type Transition = Transition;
    type Error = Error;
    type Decline = Decline;
    type State = OcrState;

    fn admit(
        admission: &Self::Admission,
        options: Self::Options,
    ) -> Result<Result<Self::State, Self::Decline>, Self::Error> {
        match prepare::admission_capabilities(admission) {
            Err(Error::Unsupported(reason)) => return Ok(Err(Decline(reason))),
            Err(error) => return Err(error),
            Ok(()) => {}
        }
        if options.credential_method == CredentialMethod::Acquisition {
            return Ok(Err(Decline("OCR credential acquisition")));
        }
        let generated_call_id = options.call_id.is_none();
        let call_id = options.call_id.unwrap_or_else(generate_call_id);
        Ok(Ok(OcrState {
            operation: Operation::Setup,
            outcome: Outcome::Success,
            asynchronous: options.asynchronous,
            internal_call: options.internal_call,
            identity: Identity {
                requested_model: admission.model.clone(),
                call_id,
                trace_id: options.trace_id,
                generated_call_id,
            },
        }))
    }

    fn operation(state: &Self::State) -> Self::Operation {
        state.operation
    }

    fn advance(
        state: &mut Self::State,
        outcome: Self::Outcome,
        observations: Self::Observation,
    ) -> Result<Self::Transition, Self::Error> {
        use Operation::*;

        if matches!(state.operation, Complete(_)) {
            return Err(Error::InvalidRequest(
                "OCR lifecycle is already complete".into(),
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
        Ok(Transition {
            operation: state.operation,
            error,
        })
    }

    fn actions_for(
        operation: Self::Operation,
        _context: &Self::Context,
    ) -> &'static [ActionBinding] {
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

fn generate_call_id() -> String {
    let id = (rand::random::<u128>() & !(0xf000_u128 << 64 | 0xc000_u128 << 48))
        | (0x4000_u128 << 64 | 0x8000_u128 << 48);
    let hex = format!("{id:032x}");
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}
