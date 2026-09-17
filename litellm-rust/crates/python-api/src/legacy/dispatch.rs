//! Legacy Python callback dispatch policy retained for migration parity.
//! `fallbacks` expires when router owns fallbacks; sync-target capability facts expire with
//! sync/async callback unification; deferred release is a host-owned candidate.

use std::collections::HashSet;

use super::vocabulary::{
    CallbackId, CallbackInvocation, CallbackMethod, Delivery, InvocationOutcome, LoggedMarker,
};

use litellm_bridge::protocol::HostPhase;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallbackCapabilities {
    pub object_sync_event: bool,
    pub sync_for_async: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackFamily {
    RequestPreCall,
    RequestPostCall,
    DeploymentPreCall,
    DeploymentPostCall,
    DeploymentFailure,
    SyncSuccess,
    AsyncSuccess,
    SyncFailure,
    AsyncFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseGate {
    Immediate,
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dispatch {
    pub family: CallbackFamily,
    pub delivery: Delivery,
    pub gate: ReleaseGate,
}

impl Dispatch {
    pub const fn immediate(family: CallbackFamily, delivery: Delivery) -> Self {
        Self {
            family,
            delivery,
            gate: ReleaseGate::Immediate,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetErrorPolicy {
    Contain,
    Propagate,
}

impl CallbackFamily {
    pub const fn dispatch_method(self) -> CallbackMethod {
        match self {
            Self::RequestPreCall => CallbackMethod::LogPreApiCall,
            Self::RequestPostCall => CallbackMethod::LogPostApiCall,
            Self::DeploymentPreCall => CallbackMethod::PreCallDeploymentHook,
            Self::DeploymentPostCall => CallbackMethod::PostCallSuccessDeploymentHook,
            Self::DeploymentFailure => CallbackMethod::PostCallFailureDeploymentHook,
            Self::SyncSuccess => CallbackMethod::LogSuccessEvent,
            Self::AsyncSuccess => CallbackMethod::AsyncLogSuccessEvent,
            Self::SyncFailure => CallbackMethod::LogFailureEvent,
            Self::AsyncFailure => CallbackMethod::AsyncLogFailureEvent,
        }
    }

    pub const fn hook_method(self) -> Option<CallbackMethod> {
        match self {
            Self::SyncSuccess => Some(CallbackMethod::LoggingHook),
            Self::AsyncSuccess => Some(CallbackMethod::AsyncLoggingHook),
            _ => None,
        }
    }

    pub const fn marker(self) -> Option<LoggedMarker> {
        match self {
            Self::SyncSuccess => Some(LoggedMarker::SyncSuccess),
            Self::AsyncSuccess => Some(LoggedMarker::AsyncSuccess),
            Self::SyncFailure => Some(LoggedMarker::SyncFailure),
            Self::AsyncFailure => Some(LoggedMarker::AsyncFailure),
            _ => None,
        }
    }

    pub const fn prepares_logging(self) -> bool {
        self.marker().is_some()
    }

    pub const fn error_policy(self) -> TargetErrorPolicy {
        match self {
            Self::DeploymentPreCall | Self::DeploymentPostCall => TargetErrorPolicy::Propagate,
            _ => TargetErrorPolicy::Contain,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchStep {
    PrepareLogging,
    Invoke(CallbackInvocation),
    MarkLogged(LoggedMarker),
    Complete { aborted: bool },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuccessFacts {
    pub asynchronous: bool,
    pub internal: bool,
    pub fallbacks: bool,
    pub deferred: bool,
    pub sync_target_capabilities: Vec<CallbackCapabilities>,
}

pub fn targets(
    family: CallbackFamily,
    global: &[CallbackId],
    dynamic: Option<&[CallbackId]>,
) -> Vec<CallbackId> {
    match family {
        CallbackFamily::RequestPreCall | CallbackFamily::RequestPostCall => global
            .iter()
            .chain(dynamic.unwrap_or_default())
            .copied()
            .collect(),
        CallbackFamily::DeploymentPreCall
        | CallbackFamily::DeploymentPostCall
        | CallbackFamily::DeploymentFailure => global.to_vec(),
        CallbackFamily::SyncSuccess
        | CallbackFamily::AsyncSuccess
        | CallbackFamily::SyncFailure
        | CallbackFamily::AsyncFailure => match dynamic {
            Some(dynamic) => {
                let mut seen = HashSet::new();
                dynamic
                    .iter()
                    .chain(global)
                    .copied()
                    .filter(|id| seen.insert(*id))
                    .collect()
            }
            None => global.to_vec(),
        },
    }
}

pub fn plan_success(facts: &SuccessFacts) -> Vec<Dispatch> {
    if !facts.asynchronous {
        return vec![Dispatch::immediate(
            CallbackFamily::SyncSuccess,
            Delivery::Worker,
        )];
    }
    let background = (!facts.internal && !facts.fallbacks).then_some(Dispatch {
        family: CallbackFamily::AsyncSuccess,
        delivery: Delivery::Background,
        gate: if facts.deferred {
            ReleaseGate::Deferred
        } else {
            ReleaseGate::Immediate
        },
    });
    let worker = facts
        .sync_target_capabilities
        .iter()
        .any(|capabilities| capabilities.sync_for_async)
        .then_some(Dispatch::immediate(
            CallbackFamily::SyncSuccess,
            Delivery::Worker,
        ));
    background.into_iter().chain(worker).collect()
}

pub fn plan_failure(phase: HostPhase, asynchronous: bool, internal: bool) -> Option<Dispatch> {
    if asynchronous && internal {
        return None;
    }
    match phase {
        HostPhase::Failure => Some(Dispatch::immediate(
            CallbackFamily::SyncFailure,
            Delivery::Inline,
        )),
        HostPhase::AsyncFailure => Some(Dispatch::immediate(
            CallbackFamily::AsyncFailure,
            Delivery::Await,
        )),
        _ => None,
    }
}

pub fn plan_request(family: CallbackFamily) -> Dispatch {
    let delivery = match family {
        CallbackFamily::RequestPreCall | CallbackFamily::RequestPostCall => Delivery::Inline,
        _ => Delivery::Await,
    };
    Dispatch::immediate(family, delivery)
}

pub fn object_target_eligible(
    sync_request: bool,
    method: CallbackMethod,
    capabilities: CallbackCapabilities,
) -> bool {
    if matches!(
        method,
        CallbackMethod::LogSuccessEvent | CallbackMethod::LogFailureEvent
    ) && capabilities.object_sync_event
    {
        return sync_request;
    }
    true
}

pub trait DispatchFacts {
    fn eligible(&mut self, target: CallbackId, method: CallbackMethod) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Position {
    Prepare,
    Hook(usize),
    Mark,
    Dispatch(usize),
    Complete { aborted: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CursorFacts {
    pub already_logged: bool,
    pub stream: bool,
    pub sync_request: bool,
}

pub struct DispatchCursor {
    dispatch: Dispatch,
    targets: Vec<CallbackId>,
    facts: CursorFacts,
    position: Position,
}

impl DispatchCursor {
    pub fn start(dispatch: Dispatch, targets: Vec<CallbackId>, facts: CursorFacts) -> Self {
        let position = if dispatch.family.marker().is_some() && facts.already_logged {
            Position::Complete { aborted: false }
        } else if dispatch.family.prepares_logging() {
            Position::Prepare
        } else {
            Position::Dispatch(0)
        };
        Self {
            dispatch,
            targets,
            facts,
            position,
        }
    }

    pub fn object_target_eligible(
        &self,
        method: CallbackMethod,
        capabilities: CallbackCapabilities,
    ) -> bool {
        object_target_eligible(self.facts.sync_request, method, capabilities)
    }

    pub const fn family(&self) -> CallbackFamily {
        self.dispatch.family
    }

    pub const fn delivery(&self) -> Delivery {
        self.dispatch.delivery
    }

    pub fn targets(&self) -> &[CallbackId] {
        &self.targets
    }

    pub fn accept(&mut self, outcome: InvocationOutcome) {
        if outcome == InvocationOutcome::Failed
            && self.dispatch.family.error_policy() == TargetErrorPolicy::Propagate
        {
            self.position = Position::Complete { aborted: true };
        }
    }

    pub fn next(&mut self, facts: &mut dyn DispatchFacts) -> DispatchStep {
        loop {
            match self.position {
                Position::Prepare => {
                    self.position = if self.dispatch.family.hook_method().is_some() {
                        Position::Hook(0)
                    } else {
                        Position::Mark
                    };
                    return DispatchStep::PrepareLogging;
                }
                Position::Hook(index) => {
                    let Some(method) = self.dispatch.family.hook_method() else {
                        self.position = Position::Mark;
                        continue;
                    };
                    let Some(target) = self.targets.get(index).copied() else {
                        self.position = Position::Mark;
                        continue;
                    };
                    self.position = Position::Hook(index + 1);
                    if facts.eligible(target, method) {
                        return DispatchStep::Invoke(self.invocation(target, method));
                    }
                }
                Position::Mark => {
                    self.position = Position::Dispatch(0);
                    if let Some(marker) = self.dispatch.family.marker()
                        && !self.facts.stream
                    {
                        return DispatchStep::MarkLogged(marker);
                    }
                }
                Position::Dispatch(index) => {
                    let Some(target) = self.targets.get(index).copied() else {
                        self.position = Position::Complete { aborted: false };
                        continue;
                    };
                    self.position = Position::Dispatch(index + 1);
                    let method = self.dispatch.family.dispatch_method();
                    if facts.eligible(target, method) {
                        return DispatchStep::Invoke(self.invocation(target, method));
                    }
                }
                Position::Complete { aborted } => return DispatchStep::Complete { aborted },
            }
        }
    }

    const fn invocation(&self, target: CallbackId, method: CallbackMethod) -> CallbackInvocation {
        CallbackInvocation {
            target,
            method,
            delivery: self.dispatch.delivery,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AllEligible;

    impl DispatchFacts for AllEligible {
        fn eligible(&mut self, _: CallbackId, _: CallbackMethod) -> bool {
            true
        }
    }

    struct Gate<F: FnMut(CallbackId, CallbackMethod) -> bool>(F);

    impl<F: FnMut(CallbackId, CallbackMethod) -> bool> DispatchFacts for Gate<F> {
        fn eligible(&mut self, target: CallbackId, method: CallbackMethod) -> bool {
            (self.0)(target, method)
        }
    }

    fn ids(values: &[u64]) -> Vec<CallbackId> {
        values.iter().copied().map(CallbackId).collect()
    }

    fn drain(cursor: &mut DispatchCursor, facts: &mut dyn DispatchFacts) -> Vec<DispatchStep> {
        let mut steps = Vec::new();
        loop {
            let step = cursor.next(facts);
            steps.push(step);
            match step {
                DispatchStep::Complete { .. } => return steps,
                DispatchStep::Invoke(_) => cursor.accept(InvocationOutcome::Completed),
                _ => {}
            }
        }
    }

    fn start(
        family: CallbackFamily,
        targets: Vec<CallbackId>,
        already_logged: bool,
        stream: bool,
    ) -> DispatchCursor {
        let delivery = match family {
            CallbackFamily::SyncSuccess => Delivery::Worker,
            CallbackFamily::AsyncSuccess => Delivery::Background,
            CallbackFamily::SyncFailure
            | CallbackFamily::RequestPreCall
            | CallbackFamily::RequestPostCall => Delivery::Inline,
            _ => Delivery::Await,
        };
        DispatchCursor::start(
            Dispatch::immediate(family, delivery),
            targets,
            CursorFacts {
                already_logged,
                stream,
                sync_request: true,
            },
        )
    }

    #[test]
    fn ordering_transcribes_each_callback_family() {
        assert_eq!(
            targets(
                CallbackFamily::SyncSuccess,
                &ids(&[3, 1, 4]),
                Some(&ids(&[1, 2]))
            ),
            ids(&[1, 2, 3, 4])
        );
        assert_eq!(
            targets(CallbackFamily::AsyncFailure, &ids(&[3, 3, 1]), None),
            ids(&[3, 3, 1])
        );
        assert_eq!(
            targets(
                CallbackFamily::RequestPreCall,
                &ids(&[1, 2]),
                Some(&ids(&[2, 3]))
            ),
            ids(&[1, 2, 2, 3])
        );
    }

    #[test]
    fn terminal_cursor_runs_hooks_marks_then_dispatches() {
        let steps = drain(
            &mut start(CallbackFamily::SyncSuccess, ids(&[1, 2]), false, false),
            &mut AllEligible,
        );
        assert_eq!(
            steps,
            vec![
                DispatchStep::PrepareLogging,
                DispatchStep::Invoke(CallbackInvocation {
                    target: CallbackId(1),
                    method: CallbackMethod::LoggingHook,
                    delivery: Delivery::Worker
                }),
                DispatchStep::Invoke(CallbackInvocation {
                    target: CallbackId(2),
                    method: CallbackMethod::LoggingHook,
                    delivery: Delivery::Worker
                }),
                DispatchStep::MarkLogged(LoggedMarker::SyncSuccess),
                DispatchStep::Invoke(CallbackInvocation {
                    target: CallbackId(1),
                    method: CallbackMethod::LogSuccessEvent,
                    delivery: Delivery::Worker
                }),
                DispatchStep::Invoke(CallbackInvocation {
                    target: CallbackId(2),
                    method: CallbackMethod::LogSuccessEvent,
                    delivery: Delivery::Worker
                }),
                DispatchStep::Complete { aborted: false },
            ]
        );
    }

    #[test]
    fn cursor_skips_already_logged_terminal_families_and_stream_markers() {
        assert_eq!(
            start(CallbackFamily::AsyncSuccess, ids(&[1]), true, false).next(&mut AllEligible),
            DispatchStep::Complete { aborted: false }
        );
        let steps = drain(
            &mut start(CallbackFamily::SyncSuccess, ids(&[1]), false, true),
            &mut AllEligible,
        );
        assert!(
            !steps
                .iter()
                .any(|step| matches!(step, DispatchStep::MarkLogged(_)))
        );
    }

    #[test]
    fn cursor_applies_eligibility_per_method_and_propagation_policy() {
        let mut cursor = start(CallbackFamily::SyncSuccess, ids(&[1, 2]), false, false);
        let mut gate = Gate(|target, method| {
            !(target == CallbackId(1) && method == CallbackMethod::LoggingHook)
                && !(target == CallbackId(2) && method == CallbackMethod::LogSuccessEvent)
        });
        let invoked: Vec<_> = drain(&mut cursor, &mut gate)
            .into_iter()
            .filter_map(|step| match step {
                DispatchStep::Invoke(invocation) => Some((invocation.target, invocation.method)),
                _ => None,
            })
            .collect();
        assert_eq!(
            invoked,
            [
                (CallbackId(2), CallbackMethod::LoggingHook),
                (CallbackId(1), CallbackMethod::LogSuccessEvent)
            ]
        );

        let mut deployment = start(
            CallbackFamily::DeploymentPreCall,
            ids(&[1, 2]),
            false,
            false,
        );
        assert!(matches!(
            deployment.next(&mut AllEligible),
            DispatchStep::Invoke(_)
        ));
        deployment.accept(InvocationOutcome::Failed);
        assert_eq!(
            deployment.next(&mut AllEligible),
            DispatchStep::Complete { aborted: true }
        );
    }

    #[test]
    fn success_planning_preserves_legacy_conditions() {
        assert_eq!(
            plan_success(&SuccessFacts {
                asynchronous: false,
                internal: true,
                fallbacks: true,
                deferred: true,
                sync_target_capabilities: vec![]
            }),
            [Dispatch::immediate(
                CallbackFamily::SyncSuccess,
                Delivery::Worker
            )]
        );
        assert_eq!(
            plan_success(&SuccessFacts {
                asynchronous: true,
                internal: false,
                fallbacks: false,
                deferred: true,
                sync_target_capabilities: vec![CallbackCapabilities {
                    object_sync_event: true,
                    sync_for_async: true,
                }]
            }),
            [
                Dispatch {
                    family: CallbackFamily::AsyncSuccess,
                    delivery: Delivery::Background,
                    gate: ReleaseGate::Deferred
                },
                Dispatch::immediate(CallbackFamily::SyncSuccess, Delivery::Worker),
            ]
        );
        assert!(
            plan_success(&SuccessFacts {
                asynchronous: true,
                internal: false,
                fallbacks: true,
                deferred: false,
                sync_target_capabilities: vec![
                    CallbackCapabilities {
                        object_sync_event: false,
                        sync_for_async: false,
                    },
                    CallbackCapabilities {
                        object_sync_event: true,
                        sync_for_async: false,
                    }
                ]
            })
            .is_empty()
        );
    }

    #[test]
    fn failure_and_request_plans_preserve_delivery() {
        assert_eq!(
            plan_failure(HostPhase::Failure, false, true),
            Some(Dispatch::immediate(
                CallbackFamily::SyncFailure,
                Delivery::Inline
            ))
        );
        assert_eq!(
            plan_failure(HostPhase::AsyncFailure, true, false),
            Some(Dispatch::immediate(
                CallbackFamily::AsyncFailure,
                Delivery::Await
            ))
        );
        assert_eq!(plan_failure(HostPhase::Failure, true, true), None);
        assert_eq!(
            plan_request(CallbackFamily::RequestPreCall).delivery,
            Delivery::Inline
        );
        assert_eq!(
            plan_request(CallbackFamily::DeploymentPreCall).delivery,
            Delivery::Await
        );
    }

    #[test]
    fn sync_leaf_methods_skip_object_targets_for_async_requests() {
        assert!(!object_target_eligible(
            false,
            CallbackMethod::LogSuccessEvent,
            CallbackCapabilities {
                object_sync_event: true,
                sync_for_async: false,
            }
        ));
        assert!(!object_target_eligible(
            false,
            CallbackMethod::LogFailureEvent,
            CallbackCapabilities {
                object_sync_event: true,
                sync_for_async: true,
            }
        ));
        assert!(object_target_eligible(
            false,
            CallbackMethod::LogSuccessEvent,
            CallbackCapabilities {
                object_sync_event: false,
                sync_for_async: true,
            }
        ));
    }
}
