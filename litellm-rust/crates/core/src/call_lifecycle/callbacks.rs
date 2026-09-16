use super::host::HostPhase;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CallbackId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    Inline,
    Await,
    Worker,
    Background,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackKind {
    CustomLogger,
    Callable { internal: bool },
    Named { known: bool },
    Opaque,
}

impl CallbackKind {
    fn runs_sync_handler_for_async_call(self) -> bool {
        match self {
            Self::Callable { internal } => !internal,
            Self::Named { known } => !known,
            Self::CustomLogger | Self::Opaque => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackMethod {
    LogPreApiCall,
    LogPostApiCall,
    PreCallDeploymentHook,
    PostCallSuccessDeploymentHook,
    PostCallFailureDeploymentHook,
    LoggingHook,
    AsyncLoggingHook,
    LogSuccessEvent,
    AsyncLogSuccessEvent,
    LogFailureEvent,
    AsyncLogFailureEvent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallbackInvocation {
    pub target: CallbackId,
    pub method: CallbackMethod,
    pub delivery: Delivery,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetErrorPolicy {
    Contain,
    Propagate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoggedMarker {
    SyncSuccess,
    AsyncSuccess,
    SyncFailure,
    AsyncFailure,
}

impl LoggedMarker {
    pub const fn key(self) -> &'static str {
        match self {
            Self::SyncSuccess => "has_logged_sync_success",
            Self::AsyncSuccess => "has_logged_async_success",
            Self::SyncFailure => "has_logged_sync_failure",
            Self::AsyncFailure => "has_logged_async_failure",
        }
    }
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

    pub fn targets(self, global: &[CallbackId], dynamic: Option<&[CallbackId]>) -> Vec<CallbackId> {
        match self {
            Self::RequestPreCall | Self::RequestPostCall => global
                .iter()
                .chain(dynamic.unwrap_or_default())
                .copied()
                .collect(),
            Self::DeploymentPreCall | Self::DeploymentPostCall | Self::DeploymentFailure => {
                global.to_vec()
            }
            Self::SyncSuccess | Self::AsyncSuccess | Self::SyncFailure | Self::AsyncFailure => {
                let Some(dynamic) = dynamic else {
                    return global.to_vec();
                };
                let mut seen = std::collections::HashSet::new();
                dynamic
                    .iter()
                    .chain(global)
                    .copied()
                    .filter(|id| seen.insert(*id))
                    .collect()
            }
        }
    }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuccessFacts {
    pub asynchronous: bool,
    pub internal: bool,
    pub fallbacks: bool,
    pub deferred: bool,
    pub sync_target_kinds: Vec<CallbackKind>,
}

pub fn plan_success(facts: &SuccessFacts) -> Vec<Dispatch> {
    if !facts.asynchronous {
        return vec![Dispatch {
            family: CallbackFamily::SyncSuccess,
            delivery: Delivery::Worker,
            gate: ReleaseGate::Immediate,
        }];
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
        .sync_target_kinds
        .iter()
        .any(|kind| kind.runs_sync_handler_for_async_call())
        .then_some(Dispatch {
            family: CallbackFamily::SyncSuccess,
            delivery: Delivery::Worker,
            gate: ReleaseGate::Immediate,
        });
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
    kind: CallbackKind,
) -> bool {
    match (method, kind) {
        (
            CallbackMethod::LogSuccessEvent | CallbackMethod::LogFailureEvent,
            CallbackKind::CustomLogger | CallbackKind::Callable { .. },
        ) => sync_request,
        _ => true,
    }
}

pub trait DispatchFacts {
    fn eligible(&mut self, target: CallbackId, method: CallbackMethod) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchStep {
    PrepareLogging,
    Invoke(CallbackInvocation),
    MarkLogged(LoggedMarker),
    Complete { aborted: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvocationOutcome {
    Completed,
    Failed,
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
        let family = dispatch.family;
        let position = if family.marker().is_some() && facts.already_logged {
            Position::Complete { aborted: false }
        } else if family.prepares_logging() {
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

    pub fn object_target_eligible(&self, method: CallbackMethod, kind: CallbackKind) -> bool {
        object_target_eligible(self.facts.sync_request, method, kind)
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
                    self.position = self.after_prepare();
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
                    match self.dispatch.family.marker() {
                        Some(marker) if !self.facts.stream => {
                            return DispatchStep::MarkLogged(marker);
                        }
                        _ => continue,
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

    fn after_prepare(&self) -> Position {
        if self.dispatch.family.hook_method().is_some() {
            Position::Hook(0)
        } else {
            Position::Mark
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

    fn ids(values: &[u64]) -> Vec<CallbackId> {
        values.iter().copied().map(CallbackId).collect()
    }

    fn delivery_for(family: CallbackFamily) -> Delivery {
        match family {
            CallbackFamily::SyncSuccess => Delivery::Worker,
            CallbackFamily::AsyncSuccess => Delivery::Background,
            CallbackFamily::SyncFailure
            | CallbackFamily::RequestPreCall
            | CallbackFamily::RequestPostCall => Delivery::Inline,
            _ => Delivery::Await,
        }
    }

    fn start(
        family: CallbackFamily,
        targets: Vec<CallbackId>,
        already_logged: bool,
        stream: bool,
    ) -> DispatchCursor {
        DispatchCursor::start(
            Dispatch::immediate(family, delivery_for(family)),
            targets,
            CursorFacts {
                already_logged,
                stream,
                sync_request: true,
            },
        )
    }

    #[test]
    fn terminal_families_order_dynamic_before_global_and_keep_first_duplicate() {
        let combined = CallbackFamily::SyncSuccess.targets(&ids(&[3, 1, 4]), Some(&ids(&[1, 2])));
        assert_eq!(combined, ids(&[1, 2, 3, 4]));
    }

    #[test]
    fn terminal_families_copy_global_without_dedup_when_dynamic_is_absent() {
        let combined = CallbackFamily::AsyncFailure.targets(&ids(&[3, 3, 1]), None);
        assert_eq!(combined, ids(&[3, 3, 1]));
    }

    #[test]
    fn request_families_order_global_before_dynamic_without_dedup() {
        let combined = CallbackFamily::RequestPreCall.targets(&ids(&[1, 2]), Some(&ids(&[2, 3])));
        assert_eq!(combined, ids(&[1, 2, 2, 3]));
    }

    #[test]
    fn success_runs_every_hook_before_any_dispatch_and_marks_between_passes() {
        let mut cursor = start(CallbackFamily::SyncSuccess, ids(&[1, 2]), false, false);
        let steps = drain(&mut cursor, &mut AllEligible);
        let invocation = |target, method| {
            DispatchStep::Invoke(CallbackInvocation {
                target: CallbackId(target),
                method,
                delivery: Delivery::Worker,
            })
        };
        assert_eq!(
            steps,
            vec![
                DispatchStep::PrepareLogging,
                invocation(1, CallbackMethod::LoggingHook),
                invocation(2, CallbackMethod::LoggingHook),
                DispatchStep::MarkLogged(LoggedMarker::SyncSuccess),
                invocation(1, CallbackMethod::LogSuccessEvent),
                invocation(2, CallbackMethod::LogSuccessEvent),
                DispatchStep::Complete { aborted: false },
            ]
        );
    }

    #[test]
    fn async_success_uses_async_leaf_methods_and_background_delivery() {
        let mut cursor = start(CallbackFamily::AsyncSuccess, ids(&[7]), false, false);
        let steps = drain(&mut cursor, &mut AllEligible);
        let methods: Vec<_> = steps
            .iter()
            .filter_map(|step| match step {
                DispatchStep::Invoke(invocation) => {
                    assert_eq!(invocation.delivery, Delivery::Background);
                    Some(invocation.method)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            methods,
            [
                CallbackMethod::AsyncLoggingHook,
                CallbackMethod::AsyncLogSuccessEvent
            ]
        );
    }

    #[test]
    fn failure_and_request_families_have_no_hook_pass() {
        for (family, delivery, method) in [
            (
                CallbackFamily::SyncFailure,
                Delivery::Inline,
                CallbackMethod::LogFailureEvent,
            ),
            (
                CallbackFamily::AsyncFailure,
                Delivery::Await,
                CallbackMethod::AsyncLogFailureEvent,
            ),
        ] {
            let mut cursor = start(family, ids(&[1, 2]), false, false);
            let steps = drain(&mut cursor, &mut AllEligible);
            assert_eq!(steps[0], DispatchStep::PrepareLogging);
            assert!(matches!(steps[1], DispatchStep::MarkLogged(_)));
            assert_eq!(
                &steps[2..],
                &[
                    DispatchStep::Invoke(CallbackInvocation {
                        target: CallbackId(1),
                        method,
                        delivery
                    }),
                    DispatchStep::Invoke(CallbackInvocation {
                        target: CallbackId(2),
                        method,
                        delivery
                    }),
                    DispatchStep::Complete { aborted: false },
                ]
            );
        }
        let mut cursor = start(CallbackFamily::RequestPreCall, ids(&[1]), false, false);
        let steps = drain(&mut cursor, &mut AllEligible);
        assert_eq!(
            steps,
            vec![
                DispatchStep::Invoke(CallbackInvocation {
                    target: CallbackId(1),
                    method: CallbackMethod::LogPreApiCall,
                    delivery: Delivery::Inline,
                }),
                DispatchStep::Complete { aborted: false },
            ]
        );
    }

    #[test]
    fn already_logged_marker_skips_the_whole_terminal_family_but_not_request_families() {
        let mut cursor = start(CallbackFamily::AsyncSuccess, ids(&[1]), true, false);
        assert_eq!(
            cursor.next(&mut AllEligible),
            DispatchStep::Complete { aborted: false }
        );
        let mut cursor = start(CallbackFamily::RequestPostCall, ids(&[1]), true, false);
        assert!(matches!(
            cursor.next(&mut AllEligible),
            DispatchStep::Invoke(_)
        ));
    }

    #[test]
    fn streaming_skips_the_marker_write_but_still_dispatches() {
        let mut cursor = start(CallbackFamily::SyncSuccess, ids(&[1]), false, true);
        let steps = drain(&mut cursor, &mut AllEligible);
        assert!(
            !steps
                .iter()
                .any(|step| matches!(step, DispatchStep::MarkLogged(_)))
        );
        assert_eq!(
            steps
                .iter()
                .filter(|step| matches!(step, DispatchStep::Invoke(_)))
                .count(),
            2
        );
    }

    #[test]
    fn ineligible_targets_are_skipped_per_method_without_affecting_others() {
        let mut cursor = start(CallbackFamily::SyncSuccess, ids(&[1, 2]), false, false);
        let mut facts = Gate(|target, method| {
            !(target == CallbackId(1) && method == CallbackMethod::LoggingHook)
                && !(target == CallbackId(2) && method == CallbackMethod::LogSuccessEvent)
        });
        let invoked: Vec<_> = drain(&mut cursor, &mut facts)
            .into_iter()
            .filter_map(|step| match step {
                DispatchStep::Invoke(invocation) => Some((invocation.target.0, invocation.method)),
                _ => None,
            })
            .collect();
        assert_eq!(
            invoked,
            [
                (2, CallbackMethod::LoggingHook),
                (1, CallbackMethod::LogSuccessEvent)
            ]
        );
    }

    #[test]
    fn contained_failures_continue_and_propagating_failures_abort() {
        let mut cursor = start(CallbackFamily::SyncFailure, ids(&[1, 2]), false, false);
        assert_eq!(cursor.next(&mut AllEligible), DispatchStep::PrepareLogging);
        assert!(matches!(
            cursor.next(&mut AllEligible),
            DispatchStep::MarkLogged(_)
        ));
        assert!(matches!(
            cursor.next(&mut AllEligible),
            DispatchStep::Invoke(_)
        ));
        cursor.accept(InvocationOutcome::Failed);
        assert!(matches!(
            cursor.next(&mut AllEligible),
            DispatchStep::Invoke(CallbackInvocation {
                target: CallbackId(2),
                ..
            })
        ));

        let mut cursor = start(
            CallbackFamily::DeploymentPreCall,
            ids(&[1, 2]),
            false,
            false,
        );
        assert!(matches!(
            cursor.next(&mut AllEligible),
            DispatchStep::Invoke(_)
        ));
        cursor.accept(InvocationOutcome::Failed);
        assert_eq!(
            cursor.next(&mut AllEligible),
            DispatchStep::Complete { aborted: true }
        );
    }

    #[test]
    fn sync_sdk_success_selects_one_worker_dispatch() {
        let plan = plan_success(&SuccessFacts {
            asynchronous: false,
            internal: true,
            fallbacks: true,
            deferred: true,
            sync_target_kinds: vec![],
        });
        assert_eq!(
            plan,
            [Dispatch {
                family: CallbackFamily::SyncSuccess,
                delivery: Delivery::Worker,
                gate: ReleaseGate::Immediate,
            }]
        );
    }

    #[test]
    fn async_sdk_success_enqueues_background_then_worker_only_for_external_sync_targets() {
        let base = SuccessFacts {
            asynchronous: true,
            internal: false,
            fallbacks: false,
            deferred: false,
            sync_target_kinds: vec![
                CallbackKind::CustomLogger,
                CallbackKind::Named { known: true },
            ],
        };
        assert_eq!(
            plan_success(&base),
            [Dispatch {
                family: CallbackFamily::AsyncSuccess,
                delivery: Delivery::Background,
                gate: ReleaseGate::Immediate,
            }]
        );
        let with_external = SuccessFacts {
            sync_target_kinds: vec![
                CallbackKind::CustomLogger,
                CallbackKind::Callable { internal: false },
            ],
            deferred: true,
            ..base.clone()
        };
        assert_eq!(
            plan_success(&with_external),
            [
                Dispatch {
                    family: CallbackFamily::AsyncSuccess,
                    delivery: Delivery::Background,
                    gate: ReleaseGate::Deferred,
                },
                Dispatch {
                    family: CallbackFamily::SyncSuccess,
                    delivery: Delivery::Worker,
                    gate: ReleaseGate::Immediate,
                },
            ]
        );
        let internal_or_fallback = SuccessFacts {
            internal: true,
            sync_target_kinds: vec![CallbackKind::Named { known: false }],
            ..base
        };
        assert_eq!(
            plan_success(&internal_or_fallback),
            [Dispatch {
                family: CallbackFamily::SyncSuccess,
                delivery: Delivery::Worker,
                gate: ReleaseGate::Immediate,
            }]
        );
    }

    #[test]
    fn opaque_and_internal_targets_never_trigger_the_worker_for_async_calls() {
        let plan = plan_success(&SuccessFacts {
            asynchronous: true,
            internal: false,
            fallbacks: true,
            deferred: false,
            sync_target_kinds: vec![
                CallbackKind::Opaque,
                CallbackKind::Callable { internal: true },
            ],
        });
        assert!(plan.is_empty());
    }

    #[test]
    fn failure_families_follow_the_phase_and_skip_internal_async_calls() {
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
        assert_eq!(plan_failure(HostPhase::Success, false, false), None);
    }

    #[test]
    fn delivery_is_selected_by_the_plan_and_carried_on_every_invocation() {
        let mut cursor = DispatchCursor::start(
            Dispatch::immediate(CallbackFamily::SyncFailure, Delivery::Worker),
            ids(&[1]),
            CursorFacts {
                already_logged: false,
                stream: false,
                sync_request: true,
            },
        );
        let deliveries: Vec<_> = drain(&mut cursor, &mut AllEligible)
            .into_iter()
            .filter_map(|step| match step {
                DispatchStep::Invoke(invocation) => Some(invocation.delivery),
                _ => None,
            })
            .collect();
        assert_eq!(deliveries, [Delivery::Worker]);
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
    fn sync_leaf_methods_skip_object_targets_on_async_requests() {
        let asynchronous = DispatchCursor::start(
            Dispatch::immediate(CallbackFamily::SyncSuccess, Delivery::Worker),
            ids(&[1]),
            CursorFacts {
                already_logged: false,
                stream: false,
                sync_request: false,
            },
        );
        for kind in [
            CallbackKind::CustomLogger,
            CallbackKind::Callable { internal: false },
        ] {
            assert!(!asynchronous.object_target_eligible(CallbackMethod::LogSuccessEvent, kind));
            assert!(!asynchronous.object_target_eligible(CallbackMethod::LogFailureEvent, kind));
            assert!(asynchronous.object_target_eligible(CallbackMethod::LoggingHook, kind));
            assert!(
                asynchronous.object_target_eligible(CallbackMethod::AsyncLogSuccessEvent, kind)
            );
        }
        assert!(asynchronous.object_target_eligible(
            CallbackMethod::LogSuccessEvent,
            CallbackKind::Named { known: false }
        ));
        let synchronous = start(CallbackFamily::SyncSuccess, ids(&[1]), false, false);
        assert!(
            synchronous.object_target_eligible(
                CallbackMethod::LogSuccessEvent,
                CallbackKind::CustomLogger
            )
        );
    }
}
