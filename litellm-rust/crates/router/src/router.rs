use std::time::{Duration, Instant};

use litellm_callbacks::event::{AttemptInfo, CallEvent};
use litellm_callbacks::host::{HostOp, HostResult};
use litellm_callbacks::machine::{HostFailure, Interrupted, Machine, MachineStep, Step};
use litellm_callbacks::route::{LayeredOp, LayeredResult, Route};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::attempt::{Attempt, AttemptContext, AttemptDisposition, AttemptFactory};
use crate::clock::Clock;
use crate::plan::{DeploymentId, LogicalCallId, RoutePlan};
use crate::report::{AttemptRecord, CallFailure, CallReport};
use crate::routing::{Picker, PlanSource, Routed, RoutingOp, RoutingResult};
use crate::signals::{Candidate, Load};

type ErrorOf<F> = <<<F as AttemptFactory>::Attempt as Machine>::Route as Route>::Error;
type AttemptComplete<F> = <<F as AttemptFactory>::Attempt as Machine>::Complete;
type Report<F> = CallReport<AttemptComplete<F>, ErrorOf<F>>;
type RouteOf<F> = Routed<<F as AttemptFactory>::Attempt>;
type StepOf<F> = Result<MachineStep<RouteOf<F>, Report<F>>, ErrorOf<F>>;

/// The whole crate is this machine. It mints the [`LogicalCallId`], walks groups and
/// retries per plan, picks deployments locally or through the host, forwards every attempt
/// op to the host, and completes with exactly one [`CallReport`].
pub struct Router<F: AttemptFactory, C: Clock> {
    plan: Option<RoutePlan>,
    picker: Picker,
    factory: F,
    clock: C,
    rng: StdRng,
    id: LogicalCallId,
    origin: Instant,
    state: State<F::Attempt, ErrorOf<F>>,
    cursor: Cursor,
    attempts: Vec<AttemptRecord>,
    last_error: Option<ErrorOf<F>>,
}

enum State<A, E> {
    Idle,
    Planning,
    Picking {
        group: u32,
    },
    Starting {
        attempt: A,
        context: AttemptContext,
    },
    Attempting {
        attempt: A,
        context: AttemptContext,
        started: Duration,
        fresh: bool,
    },
    Failing {
        error: E,
        disposition: AttemptDisposition,
    },
    Sleeping(Duration),
    Done,
}

/// Where the loop is in the plan: current group, deployments already rerouted away from
/// in that group, and retries spent on the current deployment.
#[derive(Debug, Default)]
struct Cursor {
    group: usize,
    rejected: Vec<DeploymentId>,
    current: Option<DeploymentId>,
    retries: u32,
    attempt_index: u32,
}

enum Next {
    Deployment(DeploymentId, u32),
    Ask(u32, Vec<Candidate>),
    Exhausted,
}

impl<F: AttemptFactory, C: Clock> Router<F, C> {
    pub fn new(plan: PlanSource, picker: Picker, factory: F, clock: C, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let id = LogicalCallId {
            call_id: hex128(&mut rng),
            trace_id: hex128(&mut rng),
        };
        Self {
            plan: match plan {
                PlanSource::Plan(plan) => Some(plan),
                PlanSource::Host => None,
            },
            picker,
            factory,
            origin: clock.now(),
            clock,
            rng,
            id,
            state: State::Idle,
            cursor: Cursor::default(),
            attempts: Vec::new(),
            last_error: None,
        }
    }

    pub fn id(&self) -> &LogicalCallId {
        &self.id
    }

    fn plan(&self) -> &RoutePlan {
        self.plan
            .as_ref()
            .expect("the loop only runs once the plan is resolved")
    }

    fn elapsed(&self) -> Duration {
        self.clock.now().duration_since(self.origin)
    }

    fn budget_left(&self) -> bool {
        let attempts_left = self
            .plan()
            .attempt_budget
            .is_none_or(|budget| self.cursor.attempt_index < budget);
        let time_left = self
            .plan()
            .timeout
            .is_none_or(|timeout| self.elapsed() < timeout);
        attempts_left && time_left
    }

    fn candidates(&self) -> Vec<Candidate> {
        self.plan().groups[self.cursor.group]
            .iter()
            .filter(|deployment| !self.cursor.rejected.contains(&deployment.id))
            .map(|deployment| Candidate {
                id: deployment.id,
                weight: deployment.weight,
                load: match &self.picker {
                    Picker::Local { signals, .. } => signals.load(deployment.id),
                    Picker::Host => Load::available(),
                },
            })
            .filter(|candidate| candidate.load.available)
            .collect()
    }

    fn next_deployment(&mut self) -> Next {
        if let Some(current) = self.cursor.current
            && self.cursor.retries <= self.plan().retry.max_retries
        {
            return Next::Deployment(current, self.cursor.group as u32);
        }
        while self.cursor.group < self.plan().groups.len() {
            let group = self.cursor.group as u32;
            let candidates = self.candidates();
            match &self.picker {
                Picker::Local { picker, .. } => {
                    if let Some(deployment) = picker.pick(&candidates, &mut self.rng) {
                        self.select(deployment);
                        return Next::Deployment(deployment, group);
                    }
                }
                Picker::Host if !candidates.is_empty() => return Next::Ask(group, candidates),
                Picker::Host => {}
            }
            self.advance_group();
        }
        Next::Exhausted
    }

    fn select(&mut self, deployment: DeploymentId) {
        self.cursor.current = Some(deployment);
        self.cursor.retries = 0;
    }

    fn advance_group(&mut self) {
        self.cursor.group += 1;
        self.cursor.rejected.clear();
        self.cursor.current = None;
    }

    fn backoff(&mut self, retry_after: Option<Duration>) -> Duration {
        let policy = &self.plan().retry;
        let exponent = self.cursor.retries.saturating_sub(1).min(16);
        let base = policy
            .initial_backoff
            .checked_mul(1 << exponent)
            .unwrap_or(policy.max_backoff);
        let honored = match retry_after {
            Some(after) if policy.respect_retry_after => base.max(after),
            _ => base,
        };
        let capped = honored.min(policy.max_backoff);
        if !policy.jitter {
            return capped;
        }
        let fraction = 0.5 + (self.rng.next_u32() as f64 / u32::MAX as f64) * 0.5;
        capped.mul_f64(fraction)
    }

    fn record(
        &mut self,
        context: &AttemptContext,
        started: Duration,
        failed: Option<AttemptDisposition>,
    ) {
        self.attempts.push(AttemptRecord {
            deployment: context.deployment,
            group_index: context.group_index,
            started,
            ended: self.elapsed(),
            failed,
        });
    }

    fn report(
        &mut self,
        outcome: Result<AttemptComplete<F>, CallFailure<ErrorOf<F>>>,
    ) -> Report<F> {
        self.state = State::Done;
        CallReport {
            id: self.id.clone(),
            outcome,
            attempts: std::mem::take(&mut self.attempts),
            selected: self.cursor.current,
        }
    }

    fn complete(
        &mut self,
        outcome: Result<AttemptComplete<F>, CallFailure<ErrorOf<F>>>,
    ) -> StepOf<F> {
        Ok(MachineStep::Complete(self.report(outcome)))
    }

    fn ask(&mut self, op: RoutingOp) -> StepOf<F> {
        Ok(MachineStep::Host(HostOp::Route(LayeredOp::Outer(op))))
    }

    fn start_attempt(&mut self, deployment: DeploymentId, group_index: u32) {
        let context = AttemptContext {
            id: self.id.clone(),
            attempt_index: self.cursor.attempt_index,
            deployment,
            group_index,
        };
        self.cursor.attempt_index += 1;
        let attempt = self.factory.start(&context);
        self.state = State::Starting { attempt, context };
    }

    fn info(context: &AttemptContext) -> AttemptInfo {
        AttemptInfo {
            index: context.attempt_index,
            group: context.group_index,
            deployment: context.deployment.0,
        }
    }

    fn failed_event(context: &AttemptContext, disposition: &AttemptDisposition) -> CallEvent {
        let (retryable, cooldown) = match disposition {
            AttemptDisposition::Retryable { .. } => (true, None),
            AttemptDisposition::Reroute { cooldown } => (false, *cooldown),
            AttemptDisposition::Fatal => (false, None),
        };
        CallEvent::AttemptFailed {
            attempt: Self::info(context),
            retryable,
            cooldown,
        }
    }

    fn after_failure(&mut self, disposition: AttemptDisposition) {
        match disposition {
            AttemptDisposition::Retryable { retry_after } => {
                self.cursor.retries += 1;
                if self.cursor.retries <= self.plan().retry.max_retries {
                    let sleep = self.backoff(retry_after);
                    self.state = State::Sleeping(sleep);
                    return;
                }
                self.reject_current();
            }
            AttemptDisposition::Reroute { .. } => self.reject_current(),
            AttemptDisposition::Fatal => {}
        }
        self.state = State::Idle;
    }

    fn reject_current(&mut self) {
        if let Some(current) = self.cursor.current.take() {
            self.cursor.rejected.push(current);
        }
        self.cursor.retries = 0;
    }

    fn routing_answer(result: Option<HostResult<RouteOf<F>>>) -> Option<RoutingResult> {
        match result {
            Some(HostResult::Route(LayeredResult::Outer(answer))) => Some(answer),
            _ => None,
        }
    }

    fn picked(&mut self, group: u32, choice: Option<DeploymentId>) -> Result<(), &'static str> {
        let Some(deployment) = choice else {
            self.advance_group();
            self.state = State::Idle;
            return Ok(());
        };
        if !self
            .candidates()
            .iter()
            .any(|candidate| candidate.id == deployment)
        {
            return Err("the host picked a deployment that was not a candidate");
        }
        self.select(deployment);
        self.start_attempt(deployment, group);
        Ok(())
    }

    async fn step(&mut self, mut result: Option<HostResult<RouteOf<F>>>) -> StepOf<F> {
        loop {
            match std::mem::replace(&mut self.state, State::Done) {
                State::Done => {
                    let last = self.last_error.take();
                    return self.complete(Err(CallFailure::Budget { last }));
                }
                State::Planning => match Self::routing_answer(result.take()) {
                    Some(RoutingResult::Plan(plan)) => {
                        self.plan = Some(plan);
                        self.state = State::Idle;
                    }
                    _ => {
                        return self.complete(Err(CallFailure::Protocol(
                            "resolve_plan must be answered with a plan",
                        )));
                    }
                },
                State::Picking { group } => match Self::routing_answer(result.take()) {
                    Some(RoutingResult::Picked(choice)) => {
                        if let Err(violation) = self.picked(group, choice) {
                            return self.complete(Err(CallFailure::Protocol(violation)));
                        }
                    }
                    _ => {
                        return self.complete(Err(CallFailure::Protocol(
                            "pick must be answered with a picked deployment",
                        )));
                    }
                },
                State::Sleeping(duration) => {
                    self.clock.sleep(duration).await;
                    self.state = State::Idle;
                }
                State::Idle => {
                    if self.plan.is_none() {
                        self.state = State::Planning;
                        return self.ask(RoutingOp::ResolvePlan);
                    }
                    if !self.budget_left() {
                        let last = self.last_error.take();
                        return self.complete(Err(CallFailure::Budget { last }));
                    }
                    match self.next_deployment() {
                        Next::Deployment(deployment, group) => {
                            self.start_attempt(deployment, group)
                        }
                        Next::Ask(group, candidates) => {
                            self.state = State::Picking { group };
                            return self.ask(RoutingOp::Pick { group, candidates });
                        }
                        Next::Exhausted => {
                            let failure = match self.last_error.take() {
                                Some(last) => CallFailure::Exhausted { last },
                                None => CallFailure::Budget { last: None },
                            };
                            return self.complete(Err(failure));
                        }
                    }
                }
                State::Starting { attempt, context } => {
                    let event = CallEvent::AttemptStarted {
                        attempt: Self::info(&context),
                    };
                    self.state = State::Attempting {
                        attempt,
                        started: self.elapsed(),
                        context,
                        fresh: true,
                    };
                    return Ok(MachineStep::Host(HostOp::Emit(event)));
                }
                State::Attempting {
                    mut attempt,
                    context,
                    started,
                    fresh,
                } => {
                    let input = match (fresh, result.take()) {
                        (true, _) | (false, None) => None,
                        (false, Some(answer)) => match answer.map_result(|answer| match answer {
                            LayeredResult::Inner(inner) => Some(inner),
                            LayeredResult::Outer(_) => None,
                        }) {
                            Some(inner) => Some(inner),
                            None => {
                                return self.complete(Err(CallFailure::Protocol(
                                    "an attempt op was answered with a routing result",
                                )));
                            }
                        },
                    };
                    match attempt.resume(input).await {
                        Ok(MachineStep::Host(op)) => {
                            self.state = State::Attempting {
                                attempt,
                                context,
                                started,
                                fresh: false,
                            };
                            return Ok(MachineStep::Host(op.map_op(LayeredOp::Inner)));
                        }
                        Ok(MachineStep::Complete(response)) => {
                            self.record(&context, started, None);
                            return self.complete(Ok(response));
                        }
                        Err(error) => {
                            let disposition = F::Attempt::disposition(&error);
                            self.record(&context, started, Some(disposition.clone()));
                            let event = Self::failed_event(&context, &disposition);
                            self.state = State::Failing { error, disposition };
                            return Ok(MachineStep::Host(HostOp::Emit(event)));
                        }
                    }
                }
                State::Failing { error, disposition } => {
                    result = None;
                    if disposition == AttemptDisposition::Fatal {
                        return self.complete(Err(CallFailure::Fatal { error }));
                    }
                    self.last_error = Some(error);
                    self.after_failure(disposition);
                }
            }
        }
    }

    async fn interrupt_attempt(
        &mut self,
        failure: HostFailure<ErrorOf<F>>,
    ) -> Result<Report<F>, ErrorOf<F>> {
        let (mut attempt, context, started) = match std::mem::replace(&mut self.state, State::Done)
        {
            State::Attempting {
                attempt,
                context,
                started,
                ..
            } => (attempt, context, started),
            State::Starting { attempt, context } => (attempt, context, self.elapsed()),
            _ => {
                return Ok(self.report(Err(CallFailure::Interrupted {
                    error: failure.into_error(),
                })));
            }
        };
        match attempt.interrupt(failure).await {
            Ok(response) => {
                self.record(&context, started, None);
                Ok(self.report(Ok(response)))
            }
            Err(error) => {
                self.record(&context, started, Some(F::Attempt::disposition(&error)));
                Ok(self.report(Err(CallFailure::Interrupted { error })))
            }
        }
    }
}

impl<F, C> Machine for Router<F, C>
where
    F: AttemptFactory + 'static,
    C: Clock + 'static,
{
    type Route = RouteOf<F>;
    type Complete = Report<F>;

    fn resume(&mut self, result: Option<HostResult<Self::Route>>) -> Step<'_, Self> {
        Box::pin(self.step(result))
    }

    fn interrupt(
        &mut self,
        failure: HostFailure<<Self::Route as Route>::Error>,
    ) -> Interrupted<'_, Self> {
        Box::pin(self.interrupt_attempt(failure))
    }
}

fn hex128(rng: &mut StdRng) -> String {
    format!("{:016x}{:016x}", rng.next_u64(), rng.next_u64())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::attempt::AttemptError;
    use crate::pick::RoundRobin;
    use crate::plan::{Deployment, RetryPolicy};
    use crate::signals::{NoSignals, Signals};

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Outcome {
        Ok(&'static str),
        Err(AttemptDisposition),
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ScriptedError(AttemptDisposition);

    struct ScriptedRoute;

    impl Route for ScriptedRoute {
        type Response = &'static str;
        type Error = ScriptedError;
        type Op = &'static str;
        type OpResult = ();
    }

    struct ScriptedAttempt {
        ops: VecDeque<&'static str>,
        outcome: Outcome,
    }

    impl Machine for ScriptedAttempt {
        type Route = ScriptedRoute;
        type Complete = &'static str;

        fn resume(&mut self, _: Option<HostResult<ScriptedRoute>>) -> Step<'_, Self> {
            Box::pin(async move {
                if let Some(op) = self.ops.pop_front() {
                    return Ok(MachineStep::Host(HostOp::Route(op)));
                }
                match self.outcome.clone() {
                    Outcome::Ok(value) => Ok(MachineStep::Complete(value)),
                    Outcome::Err(disposition) => Err(ScriptedError(disposition)),
                }
            })
        }

        fn interrupt(&mut self, failure: HostFailure<ScriptedError>) -> Interrupted<'_, Self> {
            Box::pin(async move { Err(failure.into_error()) })
        }
    }

    impl AttemptError for ScriptedError {
        fn disposition(&self) -> AttemptDisposition {
            self.0.clone()
        }
    }

    type Scripted = (Vec<&'static str>, Outcome);

    #[derive(Clone)]
    struct Script {
        outcomes: Arc<Mutex<VecDeque<Scripted>>>,
        contexts: Arc<Mutex<Vec<AttemptContext>>>,
    }

    impl Script {
        fn new(outcomes: Vec<Scripted>) -> Self {
            Self {
                outcomes: Arc::new(Mutex::new(outcomes.into())),
                contexts: Arc::default(),
            }
        }
    }

    impl AttemptFactory for Script {
        type Attempt = ScriptedAttempt;

        fn start(&self, context: &AttemptContext) -> ScriptedAttempt {
            self.contexts.lock().unwrap().push(context.clone());
            let (ops, outcome) = self
                .outcomes
                .lock()
                .unwrap()
                .pop_front()
                .expect("script has an outcome for every attempt");
            ScriptedAttempt {
                ops: ops.into(),
                outcome,
            }
        }
    }

    #[derive(Clone, Default)]
    struct FakeClock {
        now: Arc<Mutex<Duration>>,
        sleeps: Arc<Mutex<Vec<Duration>>>,
    }

    impl Clock for FakeClock {
        fn now(&self) -> Instant {
            static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
            *ORIGIN.get_or_init(Instant::now) + *self.now.lock().unwrap()
        }

        fn sleep(&self, duration: Duration) -> impl Future<Output = ()> + Send {
            self.sleeps.lock().unwrap().push(duration);
            *self.now.lock().unwrap() += duration;
            async {}
        }
    }

    fn deployments(ids: &[u64]) -> Vec<Deployment> {
        ids.iter()
            .copied()
            .map(|id| Deployment::new(DeploymentId(id)))
            .collect()
    }

    fn policy(max_retries: u32) -> RetryPolicy {
        RetryPolicy {
            max_retries,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(350),
            jitter: false,
            respect_retry_after: true,
        }
    }

    fn retryable(retry_after: Option<u64>) -> Outcome {
        Outcome::Err(AttemptDisposition::Retryable {
            retry_after: retry_after.map(Duration::from_millis),
        })
    }

    fn reroute() -> Outcome {
        Outcome::Err(AttemptDisposition::Reroute { cooldown: None })
    }

    fn local() -> Picker {
        Picker::local(RoundRobin::default(), NoSignals)
    }

    async fn drain<C: Clock + 'static>(
        router: &mut Router<Script, C>,
    ) -> (Vec<&'static str>, Report<Script>) {
        let mut ops = Vec::new();
        let mut result = None;
        loop {
            match router
                .resume(result.take())
                .await
                .expect("router never errors")
            {
                MachineStep::Host(HostOp::Route(LayeredOp::Inner(op))) => {
                    ops.push(op);
                    result = Some(HostResult::Route(LayeredResult::Inner(())));
                }
                MachineStep::Host(HostOp::Route(LayeredOp::Outer(op))) => {
                    panic!("a local picker never asks the host: {op:?}")
                }
                MachineStep::Host(HostOp::Emit(_)) => result = Some(HostResult::Emitted),
                MachineStep::Host(_) => panic!("scripted attempts only yield route ops"),
                MachineStep::Complete(report) => return (ops, report),
            }
        }
    }

    #[tokio::test]
    async fn forwards_every_attempt_op_and_reports_success_on_attempt_k() {
        let script = Script::new(vec![
            (vec!["project", "send"], retryable(None)),
            (vec!["project", "send"], Outcome::Ok("done")),
        ]);
        let mut router = Router::new(
            RoutePlan::single(deployments(&[1]), policy(3)).into(),
            local(),
            script.clone(),
            FakeClock::default(),
            7,
        );
        let (ops, report) = drain(&mut router).await;
        assert_eq!(ops, ["project", "send", "project", "send"]);
        assert_eq!(report.outcome, Ok("done"));
        assert_eq!(report.attempts.len(), 2);
        assert!(report.attempts[0].failed.is_some());
        assert!(report.attempts[1].failed.is_none());
        assert_eq!(report.selected, Some(DeploymentId(1)));
        let contexts = script.contexts.lock().unwrap();
        assert!(contexts.iter().all(|context| context.id == report.id));
        assert_eq!(
            contexts.iter().map(|c| c.attempt_index).collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[tokio::test]
    async fn backoff_doubles_honors_retry_after_and_caps_at_max() {
        let clock = FakeClock::default();
        let script = Script::new(vec![
            (vec![], retryable(None)),
            (vec![], retryable(Some(300))),
            (vec![], retryable(None)),
            (vec![], Outcome::Ok("done")),
        ]);
        let mut router = Router::new(
            RoutePlan::single(deployments(&[1]), policy(3)).into(),
            local(),
            script,
            clock.clone(),
            7,
        );
        let (_, report) = drain(&mut router).await;
        assert_eq!(report.outcome, Ok("done"));
        assert_eq!(
            *clock.sleeps.lock().unwrap(),
            [
                Duration::from_millis(100),
                Duration::from_millis(300),
                Duration::from_millis(350)
            ]
        );
    }

    #[tokio::test]
    async fn reroute_skips_remaining_retries_and_falls_back_group_by_group() {
        let script = Script::new(vec![
            (vec![], reroute()),
            (vec![], reroute()),
            (vec![], Outcome::Ok("fallback")),
        ]);
        let plan = RoutePlan {
            groups: vec![deployments(&[1, 2]), deployments(&[9])],
            retry: policy(5),
            attempt_budget: None,
            timeout: None,
        };
        let mut router = Router::new(
            plan.into(),
            local(),
            script.clone(),
            FakeClock::default(),
            7,
        );
        let (_, report) = drain(&mut router).await;
        assert_eq!(report.outcome, Ok("fallback"));
        let contexts = script.contexts.lock().unwrap();
        let visited: Vec<_> = contexts
            .iter()
            .map(|c| (c.deployment, c.group_index))
            .collect();
        assert_eq!(
            visited,
            [
                (DeploymentId(1), 0),
                (DeploymentId(2), 0),
                (DeploymentId(9), 1)
            ]
        );
    }

    #[tokio::test]
    async fn fatal_stops_immediately_with_budget_left() {
        let script = Script::new(vec![(vec![], Outcome::Err(AttemptDisposition::Fatal))]);
        let mut router = Router::new(
            RoutePlan::single(deployments(&[1, 2]), policy(5)).into(),
            local(),
            script,
            FakeClock::default(),
            7,
        );
        let (_, report) = drain(&mut router).await;
        assert_eq!(
            report.outcome,
            Err(CallFailure::Fatal {
                error: ScriptedError(AttemptDisposition::Fatal)
            })
        );
        assert_eq!(report.attempts.len(), 1);
    }

    #[tokio::test]
    async fn exhausted_candidates_report_the_last_error() {
        let script = Script::new(vec![(vec![], retryable(None)), (vec![], reroute())]);
        let mut router = Router::new(
            RoutePlan::single(deployments(&[1]), policy(1)).into(),
            local(),
            script,
            FakeClock::default(),
            7,
        );
        let (_, report) = drain(&mut router).await;
        assert_eq!(
            report.outcome,
            Err(CallFailure::Exhausted {
                last: ScriptedError(AttemptDisposition::Reroute { cooldown: None })
            })
        );
        assert_eq!(report.attempts.len(), 2);
    }

    #[tokio::test]
    async fn attempt_budget_stops_before_the_next_attempt_starts() {
        let script = Script::new(vec![(vec![], retryable(None)), (vec![], retryable(None))]);
        let plan = RoutePlan {
            groups: vec![deployments(&[1])],
            retry: policy(10),
            attempt_budget: Some(2),
            timeout: None,
        };
        let mut router = Router::new(plan.into(), local(), script, FakeClock::default(), 7);
        let (_, report) = drain(&mut router).await;
        assert!(matches!(
            report.outcome,
            Err(CallFailure::Budget { last: Some(_) })
        ));
        assert_eq!(report.attempts.len(), 2);
    }

    #[tokio::test]
    async fn empty_plan_reports_budget_without_running_anything() {
        let script = Script::new(vec![]);
        let mut router = Router::new(
            RoutePlan::single(vec![], policy(1)).into(),
            local(),
            script,
            FakeClock::default(),
            7,
        );
        let (ops, report) = drain(&mut router).await;
        assert!(ops.is_empty());
        assert_eq!(report.outcome, Err(CallFailure::Budget { last: None }));
        assert!(report.attempts.is_empty());
        assert_eq!(report.selected, None);
    }

    #[tokio::test]
    async fn interrupt_mid_attempt_ends_the_logical_call() {
        let script = Script::new(vec![(vec!["send", "never"], Outcome::Ok("unreachable"))]);
        let mut router = Router::new(
            RoutePlan::single(deployments(&[1, 2]), policy(5)).into(),
            local(),
            script.clone(),
            FakeClock::default(),
            7,
        );
        let started = router.resume(None).await.unwrap();
        assert!(matches!(
            started,
            MachineStep::Host(HostOp::Emit(CallEvent::AttemptStarted { .. }))
        ));
        let first = router.resume(Some(HostResult::Emitted)).await.unwrap();
        assert!(matches!(
            first,
            MachineStep::Host(HostOp::Route(LayeredOp::Inner("send")))
        ));
        let cancelled = ScriptedError(AttemptDisposition::Reroute { cooldown: None });
        let report = router
            .interrupt(HostFailure::Cancelled(cancelled.clone()))
            .await
            .unwrap();
        assert_eq!(
            report.outcome,
            Err(CallFailure::Interrupted { error: cancelled })
        );
        assert_eq!(script.contexts.lock().unwrap().len(), 1);
    }

    struct CoolingDown(DeploymentId);

    impl Signals for CoolingDown {
        fn load(&self, deployment: DeploymentId) -> Load {
            Load {
                available: deployment != self.0,
                ..Load::available()
            }
        }
    }

    #[tokio::test]
    async fn cooled_down_deployments_are_never_attempted() {
        let script = Script::new(vec![(vec![], Outcome::Ok("done"))]);
        let mut router = Router::new(
            RoutePlan::single(deployments(&[1, 2]), policy(0)).into(),
            Picker::local(RoundRobin::default(), CoolingDown(DeploymentId(1))),
            script.clone(),
            FakeClock::default(),
            7,
        );
        let (_, report) = drain(&mut router).await;
        assert_eq!(report.selected, Some(DeploymentId(2)));
        assert_eq!(
            script.contexts.lock().unwrap()[0].deployment,
            DeploymentId(2)
        );
    }

    #[test]
    fn logical_call_ids_are_unique_per_router() {
        let make = |seed| {
            Router::new(
                RoutePlan::single(deployments(&[1]), policy(0)).into(),
                local(),
                Script::new(vec![]),
                FakeClock::default(),
                seed,
            )
            .id()
            .clone()
        };
        assert_ne!(make(1), make(2));
        assert_ne!(make(1).call_id, make(1).trace_id);
    }
}
