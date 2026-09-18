use std::time::{Duration, Instant};

use litellm_callbacks::event::{AttemptInfo, CallEvent};
use litellm_callbacks::failure::FailureClass;
use litellm_callbacks::host::{HostOp, HostResult};
use litellm_callbacks::machine::{HostFailure, Interrupted, Machine, MachineStep, Step};
use litellm_callbacks::route::{LayeredOp, LayeredResult, Route};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::attempt::{Attempt, AttemptContext, AttemptFactory};
use crate::clock::Clock;
use crate::decide::{ChainsConfigured, Decision, Situation, decide};
use crate::plan::{Deployment, DeploymentId, Fallbacks, RoutePlan};
use crate::report::{AttemptRecord, CallFailure, CallReport};
use crate::routing::{Picker, PlanSource, Routed, RoutingOp, RoutingResult};
use crate::signals::{Candidate, Load};

type ErrorOf<F> = <<<F as AttemptFactory>::Attempt as Machine>::Route as Route>::Error;
type AttemptComplete<F> = <<F as AttemptFactory>::Attempt as Machine>::Complete;
type Report<F> = CallReport<AttemptComplete<F>, ErrorOf<F>>;
type RouteOf<F> = Routed<<F as AttemptFactory>::Attempt>;
type StepOf<F> = Result<MachineStep<RouteOf<F>, Report<F>>, ErrorOf<F>>;

/// The whole crate is this machine. It walks groups and retries per plan, picks
/// deployments locally or through the host, forwards every attempt op and chunk to the
/// host, reports each failure before deciding on it, and completes with exactly one
/// [`CallReport`].
pub struct Router<F: AttemptFactory, C: Clock> {
    plan: Option<RoutePlan>,
    picker: Picker,
    factory: F,
    clock: C,
    rng: StdRng,
    trace_id: String,
    origin: Instant,
    state: State<F::Attempt, ErrorOf<F>>,
    cursor: Cursor,
    attempts: Vec<AttemptRecord>,
    last_error: Option<ErrorOf<F>>,
}

enum State<A, E> {
    Idle,
    Planning,
    Picking,
    Starting {
        attempt: A,
        context: AttemptContext,
    },
    Attempting {
        attempt: A,
        context: AttemptContext,
        started: Duration,
        fresh: bool,
        chunks: u32,
    },
    /// The failure op is out; the decision is made once the host has recorded it.
    Recording {
        error: E,
        class: FailureClass,
        retry_after: Option<Duration>,
        context: AttemptContext,
        started: Duration,
        ended: Duration,
        chunks: u32,
    },
    Sleeping,
    Advancing,
    Done,
}

/// Where the loop is: the current group, what it must not pick again in it, what it has
/// tried overall, and the retries spent so far in this group.
#[derive(Debug, Default)]
struct Cursor {
    group: Vec<Deployment>,
    depth: u32,
    skipped: Vec<DeploymentId>,
    tried: Vec<DeploymentId>,
    chain_class: Option<FailureClass>,
    retries: u32,
    attempt_index: u32,
    last: Option<DeploymentId>,
    last_class: Option<FailureClass>,
}

enum Next {
    Deployment(DeploymentId),
    Ask(Vec<Candidate>),
    Exhausted,
}

impl<F: AttemptFactory, C: Clock> Router<F, C> {
    pub fn new(
        plan: PlanSource,
        picker: Picker,
        factory: F,
        clock: C,
        seed: u64,
        trace_id: Option<String>,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let trace_id = trace_id.unwrap_or_else(|| hex128(&mut rng));
        let (plan, group) = match plan {
            PlanSource::Plan(plan) => {
                let group = plan.primary.clone();
                (Some(*plan), group)
            }
            PlanSource::Host => (None, Vec::new()),
        };
        Self {
            plan,
            picker,
            factory,
            origin: clock.now(),
            clock,
            rng,
            trace_id,
            state: State::Idle,
            cursor: Cursor {
                group,
                ..Cursor::default()
            },
            attempts: Vec::new(),
            last_error: None,
        }
    }

    pub fn trace_id(&self) -> &str {
        &self.trace_id
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
        self.cursor
            .group
            .iter()
            .filter(|deployment| !self.cursor.skipped.contains(&deployment.id))
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
        let candidates = self.candidates();
        match &self.picker {
            Picker::Local { picker, .. } => match picker.pick(&candidates, &mut self.rng) {
                Some(deployment) => Next::Deployment(deployment),
                None => Next::Exhausted,
            },
            Picker::Host if candidates.is_empty() => Next::Exhausted,
            Picker::Host => Next::Ask(candidates),
        }
    }

    fn enter_group(&mut self, group: Vec<Deployment>) {
        self.cursor.group = group;
        self.cursor.depth += 1;
        self.cursor.skipped.clear();
        self.cursor.retries = 0;
        self.state = State::Idle;
    }

    /// The current group is done with. Enters the next one in process, asks the host for
    /// it, or ends the call; `None` means the loop simply continues in the new group.
    fn leave_group(&mut self, class: FailureClass) -> Option<StepOf<F>> {
        let class = *self.cursor.chain_class.get_or_insert(class);
        let depth = self.cursor.depth;
        match &self.plan().fallbacks {
            Fallbacks::Disabled => Some(self.exhausted()),
            Fallbacks::Chains(chains) => match chains.next(class, depth) {
                Some(group) => {
                    self.enter_group(group);
                    None
                }
                None => Some(self.exhausted()),
            },
            Fallbacks::Host => {
                self.state = State::Advancing;
                Some(self.ask(RoutingOp::NextGroup {
                    class,
                    depth,
                    tried: self.cursor.tried.clone(),
                }))
            }
        }
    }

    fn exhausted(&mut self) -> StepOf<F> {
        match self.last_error.take() {
            Some(last) => self.complete(Err(CallFailure::Exhausted { last })),
            None => self.complete(Err(CallFailure::Budget { last: None })),
        }
    }

    fn jitter(&mut self) -> Duration {
        let fraction = f64::from(self.rng.next_u32()) / f64::from(u32::MAX);
        self.plan().retry.jitter.mul_f64(fraction)
    }

    fn situation(
        &self,
        class: FailureClass,
        retry_after: Option<Duration>,
        failed: Deployment,
    ) -> Situation {
        Situation {
            class,
            retry_after,
            retries_used: self.cursor.retries,
            failed,
            group_size: self.cursor.group.len(),
            available: self.candidates().len(),
            chains: match &self.plan().fallbacks {
                Fallbacks::Disabled => ChainsConfigured::NONE,
                Fallbacks::Chains(chains) => chains.configured(),
                Fallbacks::Host => ChainsConfigured::ALL,
            },
        }
    }

    fn record(
        &mut self,
        context: &AttemptContext,
        started: Duration,
        ended: Duration,
        failure: Option<(FailureClass, Decision)>,
    ) {
        self.attempts.push(AttemptRecord {
            deployment: context.deployment,
            group_index: context.group_index,
            started,
            ended,
            failure,
        });
    }

    fn report(
        &mut self,
        outcome: Result<AttemptComplete<F>, CallFailure<ErrorOf<F>>>,
    ) -> Report<F> {
        self.state = State::Done;
        CallReport {
            trace_id: self.trace_id.clone(),
            outcome,
            attempts: std::mem::take(&mut self.attempts),
            selected: self.cursor.last,
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

    fn start_attempt(&mut self, deployment: DeploymentId) {
        let context = AttemptContext {
            trace_id: self.trace_id.clone(),
            attempt_index: self.cursor.attempt_index,
            deployment,
            group_index: self.cursor.depth,
        };
        self.cursor.attempt_index += 1;
        self.cursor.tried.push(deployment);
        self.cursor.last = Some(deployment);
        let attempt = self.factory.start(&context);
        self.state = State::Starting { attempt, context };
    }

    fn info(&self, context: &AttemptContext) -> AttemptInfo {
        AttemptInfo {
            trace_id: self.trace_id.clone(),
            index: context.attempt_index,
            group: context.group_index,
            deployment: context.deployment.0,
        }
    }

    fn deployment(&self, id: DeploymentId) -> Deployment {
        self.cursor
            .group
            .iter()
            .copied()
            .find(|deployment| deployment.id == id)
            .unwrap_or(Deployment::new(id))
    }

    /// Carries out a decision; `None` means the loop continues in the same or a new group.
    fn apply(
        &mut self,
        decision: Decision,
        class: FailureClass,
        error: ErrorOf<F>,
        failed: DeploymentId,
    ) -> Option<StepOf<F>> {
        self.cursor.last_class = Some(class);
        match decision {
            Decision::Retry {
                skip_failed,
                backoff,
            } => {
                self.last_error = Some(error);
                self.cursor.retries += 1;
                if skip_failed {
                    self.cursor.skipped.push(failed);
                }
                self.state = State::Sleeping;
                Some(self.ask(RoutingOp::Backoff {
                    attempt: self.cursor.attempt_index - 1,
                    duration: backoff,
                }))
            }
            Decision::Fallback => {
                self.last_error = Some(error);
                self.leave_group(class)
            }
            Decision::Stop => Some(self.complete(Err(CallFailure::Exhausted { last: error }))),
        }
    }

    fn routing_answer(result: Option<HostResult<RouteOf<F>>>) -> Option<RoutingResult> {
        match result {
            Some(HostResult::Route(LayeredResult::Outer(answer))) => Some(answer),
            _ => None,
        }
    }

    fn picked(&mut self, choice: Option<DeploymentId>) -> Result<Option<StepOf<F>>, &'static str> {
        let Some(deployment) = choice else {
            let class = self.cursor.last_class.unwrap_or(FailureClass::RateLimited);
            return Ok(self.leave_group(class));
        };
        if !self
            .candidates()
            .iter()
            .any(|candidate| candidate.id == deployment)
        {
            return Err("the host picked a deployment that was not a candidate");
        }
        self.start_attempt(deployment);
        Ok(None)
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
                        self.cursor.group = plan.primary.clone();
                        self.plan = Some(*plan);
                        self.state = State::Idle;
                    }
                    _ => {
                        return self.complete(Err(CallFailure::Protocol(
                            "resolve_plan must be answered with a plan",
                        )));
                    }
                },
                State::Picking => match Self::routing_answer(result.take()) {
                    Some(RoutingResult::Picked(choice)) => match self.picked(choice) {
                        Ok(Some(step)) => return step,
                        Ok(None) => {}
                        Err(violation) => {
                            return self.complete(Err(CallFailure::Protocol(violation)));
                        }
                    },
                    _ => {
                        return self.complete(Err(CallFailure::Protocol(
                            "pick must be answered with a picked deployment",
                        )));
                    }
                },
                State::Sleeping => match Self::routing_answer(result.take()) {
                    Some(RoutingResult::Slept) => self.state = State::Idle,
                    _ => {
                        return self.complete(Err(CallFailure::Protocol(
                            "backoff must be answered with slept",
                        )));
                    }
                },
                State::Advancing => match Self::routing_answer(result.take()) {
                    Some(RoutingResult::Group(Some(group))) => self.enter_group(group),
                    Some(RoutingResult::Group(None)) => return self.exhausted(),
                    _ => {
                        return self.complete(Err(CallFailure::Protocol(
                            "next_group must be answered with a group",
                        )));
                    }
                },
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
                        Next::Deployment(deployment) => self.start_attempt(deployment),
                        Next::Ask(candidates) => {
                            self.state = State::Picking;
                            let group = self.cursor.depth;
                            return self.ask(RoutingOp::Pick { group, candidates });
                        }
                        Next::Exhausted => {
                            let class = self.cursor.last_class.unwrap_or(FailureClass::RateLimited);
                            if let Some(step) = self.leave_group(class) {
                                return step;
                            }
                        }
                    }
                }
                State::Starting { attempt, context } => {
                    let event = CallEvent::AttemptStarted {
                        attempt: self.info(&context),
                    };
                    self.state = State::Attempting {
                        attempt,
                        started: self.elapsed(),
                        context,
                        fresh: true,
                        chunks: 0,
                    };
                    return Ok(MachineStep::Host(HostOp::Emit(event)));
                }
                State::Attempting {
                    mut attempt,
                    context,
                    started,
                    fresh,
                    chunks,
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
                                chunks,
                            };
                            return Ok(MachineStep::Host(op.map_op(LayeredOp::Inner)));
                        }
                        Ok(MachineStep::Yield(chunk)) => {
                            self.state = State::Attempting {
                                attempt,
                                context,
                                started,
                                fresh: false,
                                chunks: chunks + 1,
                            };
                            return Ok(MachineStep::Yield(chunk));
                        }
                        Ok(MachineStep::Complete(response)) => {
                            let ended = self.elapsed();
                            self.record(&context, started, ended, None);
                            return self.complete(Ok(response));
                        }
                        Err(error) => {
                            let class = F::Attempt::class(&error);
                            let retry_after = F::Attempt::retry_after(&error);
                            let op = HostOp::AttemptFailed {
                                attempt: self.info(&context),
                                class,
                                error: error.clone(),
                            };
                            self.state = State::Recording {
                                error,
                                class,
                                retry_after,
                                context,
                                started,
                                ended: self.elapsed(),
                                chunks,
                            };
                            return Ok(MachineStep::Host(op));
                        }
                    }
                }
                State::Recording {
                    error,
                    class,
                    retry_after,
                    context,
                    started,
                    ended,
                    chunks,
                } => {
                    if !matches!(result.take(), Some(HostResult::Recorded)) {
                        return self.complete(Err(CallFailure::Protocol(
                            "an attempt failure must be recorded before the loop continues",
                        )));
                    }
                    if chunks > 0 {
                        self.record(&context, started, ended, Some((class, Decision::Stop)));
                        return self.complete(Err(CallFailure::MidStream { error, chunks }));
                    }
                    let failed = self.deployment(context.deployment);
                    let situation = self.situation(class, retry_after, failed);
                    let jitter = self.jitter();
                    let decision = decide(&situation, &self.plan().retry, jitter);
                    self.record(&context, started, ended, Some((class, decision)));
                    if let Some(step) = self.apply(decision, class, error, failed.id) {
                        return step;
                    }
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
                let ended = self.elapsed();
                self.record(&context, started, ended, None);
                Ok(self.report(Ok(response)))
            }
            Err(error) => {
                let ended = self.elapsed();
                let class = F::Attempt::class(&error);
                self.record(&context, started, ended, Some((class, Decision::Stop)));
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

    fn succeeded(report: &Self::Complete) -> bool {
        report.outcome.is_ok()
    }

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
    use litellm_callbacks::failure::Classified;
    use litellm_callbacks::machine::{HostFailure, Interrupted, Step};

    use super::*;
    use crate::pick::RoundRobin;
    use crate::plan::RetryPolicy;
    use crate::signals::NoSignals;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Never;

    impl Classified for Never {
        fn class(&self) -> FailureClass {
            FailureClass::BadRequest
        }
    }

    struct Unit;

    impl Route for Unit {
        type Response = ();
        type Error = Never;
        type Op = ();
        type OpResult = ();
        type Chunk = std::convert::Infallible;
    }

    struct Idle;

    impl Machine for Idle {
        type Route = Unit;
        type Complete = ();

        fn resume(&mut self, _: Option<HostResult<Unit>>) -> Step<'_, Self> {
            Box::pin(async { Ok(MachineStep::Complete(())) })
        }

        fn interrupt(&mut self, failure: HostFailure<Never>) -> Interrupted<'_, Self> {
            Box::pin(async move { Err(failure.into_error()) })
        }
    }

    struct Factory;

    impl AttemptFactory for Factory {
        type Attempt = Idle;

        fn start(&self, _: &AttemptContext) -> Idle {
            Idle
        }
    }

    fn router(seed: u64, trace_id: Option<&str>) -> Router<Factory, crate::clock::SystemClock> {
        Router::new(
            RoutePlan::single(Vec::new(), RetryPolicy::none()).into(),
            Picker::local(RoundRobin::default(), NoSignals),
            Factory,
            crate::clock::SystemClock,
            seed,
            trace_id.map(String::from),
        )
    }

    #[test]
    fn a_trace_id_is_minted_from_the_seed_only_when_the_host_supplies_none() {
        assert_eq!(router(1, None).trace_id(), router(1, None).trace_id());
        assert_ne!(router(1, None).trace_id(), router(2, None).trace_id());
        assert_eq!(router(1, Some("proxy-trace")).trace_id(), "proxy-trace");
    }
}
