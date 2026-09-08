use std::future::{Ready, ready};

use crate::Error;
use crate::integrations::custom_logger::{LogError, LogFuture};
use crate::lifecycle::{
    ActionBinding, ActionKind, ActionResult, CallLifecycle, CallLifecycleContext, Clock, Delivery,
    ErrorDisposition, ExecutedCall, FailurePolicy, Lifecycle, LifecycleRoute, Outcome, Owner,
    RequestPolicy, ResultPolicy, TerminalDispatcher, TerminalRecord,
};

use super::handler::execute_messages_provider_call;
use super::types::{AnthropicMessagesResponse, MessagesRequest};

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

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub asynchronous: bool,
    pub internal_call: bool,
    pub call_id: Option<String>,
    pub trace_id: Option<String>,
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
pub struct MessagesState {
    operation: Operation,
    outcome: Outcome,
    asynchronous: bool,
    internal_call: bool,
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
                "messages lifecycle is already complete".into(),
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

pub trait MessagesServices:
    RequestPolicy<MessagesRequest, MessagesRequest> + TerminalDispatcher + Clock
{
}

impl<T> MessagesServices for T where
    T: RequestPolicy<MessagesRequest, MessagesRequest> + TerminalDispatcher + Clock
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

impl RequestPolicy<MessagesRequest, MessagesRequest> for NoopServices {
    type PreCallFuture<'a> = Ready<ActionResult<MessagesRequest, Error>>;
    type DuringCallFuture<'a> = Ready<ActionResult<MessagesRequest, Error>>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        ready(ActionResult::Continue(request))
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::DuringCallFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

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
    CallLifecycle
        .run(
            context,
            request,
            services,
            services,
            services,
            |request| async move { execute_messages_provider_call(request).await },
        )
        .await
}

pub fn machine(options: Options) -> Result<Lifecycle<MessagesRoute>, Error> {
    Lifecycle::admit(&(), options).map(|result| match result {
        Ok(machine) => machine,
        Err(never) => match never {},
    })
}
