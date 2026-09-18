//! One harness for every router test binary: scripted attempts, a scenario host that
//! answers the routing ops from a script and keeps a cooldown table, and the trace
//! vocabulary the cases are written in.
#![allow(dead_code)] // shared by several test binaries, each of which uses a subset

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use litellm_callbacks::event::AttemptInfo;
use litellm_callbacks::failure::{Classified, FailureClass};
use litellm_callbacks::layer::Stack;
use litellm_callbacks::machine::Machine;
use litellm_callbacks::route::{Layered, LayeredOp, LayeredResult};
use litellm_callbacks_test::{
    Answers, Observed, RecordingHost, Script, Scripted, TestRoute, Trace, drain,
};
use litellm_router::{
    Attempt, AttemptContext, AttemptFactory, CallFailure, CallReport, Candidate, Clock, Decision,
    Deployment, DeploymentId, FallbackChains, Fallbacks, Load, Picker, PlanSource, Retries,
    RetryPolicy, RoundRobin, RoutePlan, RouterLayer, Routing, RoutingOp, RoutingResult, Signals,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestError {
    pub class: FailureClass,
    pub retry_after: Option<Duration>,
}

impl Classified for TestError {
    fn class(&self) -> FailureClass {
        self.class
    }

    fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }
}

pub fn failure(class: FailureClass) -> TestError {
    TestError {
        class,
        retry_after: None,
    }
}

/// An attempt that fails with this class after its one `send` op.
pub fn err(class: FailureClass) -> Script<TestError> {
    Script::err(&["send"], failure(class))
}

/// A rate limit carrying a `Retry-After` of this many seconds.
pub fn rate_limited_after(seconds: u64) -> Script<TestError> {
    Script::err(
        &["send"],
        TestError {
            class: FailureClass::RateLimited,
            retry_after: Some(Duration::from_secs(seconds)),
        },
    )
}

pub fn ok() -> Script<TestError> {
    Script::ok(&["send"], "ok")
}

/// Starts scripted attempts in order and records every context it was asked for.
#[derive(Clone)]
pub struct Attempts {
    scripts: Arc<Mutex<Vec<Script<TestError>>>>,
    pub started: Arc<Mutex<Vec<AttemptContext>>>,
}

impl Attempts {
    pub fn new(scripts: Vec<Script<TestError>>) -> Self {
        Self {
            scripts: Arc::new(Mutex::new(scripts)),
            started: Arc::default(),
        }
    }
}

impl AttemptFactory for Attempts {
    type Attempt = Scripted<TestError>;

    fn start(&self, context: &AttemptContext) -> Scripted<TestError> {
        self.started.lock().unwrap().push(context.clone());
        let mut scripts = self.scripts.lock().unwrap();
        assert!(
            !scripts.is_empty(),
            "the router started more attempts than were scripted"
        );
        Scripted::new(scripts.remove(0))
    }
}

/// A clock the scenario host advances by every backoff it is asked to wait.
#[derive(Clone, Default)]
pub struct FakeClock(Arc<Mutex<Duration>>);

impl FakeClock {
    pub fn advance(&self, duration: Duration) {
        *self.0.lock().unwrap() += duration;
    }

    pub fn elapsed(&self) -> Duration {
        *self.0.lock().unwrap()
    }
}

impl Clock for FakeClock {
    fn now(&self) -> Instant {
        static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        *ORIGIN.get_or_init(Instant::now) + *self.0.lock().unwrap()
    }
}

/// The cooldown table the router's signals read and the scenario host writes.
#[derive(Clone, Default)]
pub struct Cooldowns(Arc<Mutex<Vec<DeploymentId>>>);

impl Cooldowns {
    pub fn cool(&self, deployment: DeploymentId) {
        self.0.lock().unwrap().push(deployment);
    }
}

impl Signals for Cooldowns {
    fn load(&self, deployment: DeploymentId) -> Load {
        Load {
            available: !self.0.lock().unwrap().contains(&deployment),
            ..Load::available()
        }
    }
}

pub fn deployments(ids: &[u64]) -> Vec<Deployment> {
    ids.iter()
        .map(|id| Deployment::new(DeploymentId(*id)))
        .collect()
}

/// Python's retry timing without the random jitter, so waits are exact in a trace.
pub fn python_retries(retries: Retries) -> RetryPolicy {
    RetryPolicy {
        jitter: Duration::ZERO,
        ..RetryPolicy::python(retries)
    }
}

/// One group with the router's default retry count and no fallbacks.
pub fn plan(ids: &[u64], retries: u32) -> RoutePlan {
    RoutePlan::single(deployments(ids), python_retries(Retries::Default(retries)))
}

pub fn chains(generic: &[&[u64]]) -> FallbackChains {
    FallbackChains::generic(generic.iter().map(|group| deployments(group)).collect())
}

pub fn with_fallbacks(plan: RoutePlan, chains: FallbackChains) -> RoutePlan {
    RoutePlan {
        fallbacks: Fallbacks::Chains(chains),
        ..plan
    }
}

pub fn ids(candidates: &[Candidate]) -> String {
    let ids: Vec<String> = candidates.iter().map(|c| c.id.0.to_string()).collect();
    format!("[{}]", ids.join(","))
}

fn deployment_ids(deployments: &[DeploymentId]) -> String {
    let ids: Vec<String> = deployments.iter().map(|d| d.0.to_string()).collect();
    format!("[{}]", ids.join(","))
}

/// The routing half of the trace vocabulary: `plan`, `pick:<group>:[ids]`,
/// `next_group:<class>@<depth>:[tried]`, `sleep:<ms>`.
pub fn record(trace: &mut Trace, op: &RoutingOp) {
    trace.0.push(Observed::Route(match op {
        RoutingOp::ResolvePlan => "plan".into(),
        RoutingOp::Pick { group, candidates } => format!("pick:{group}:{}", ids(candidates)),
        RoutingOp::NextGroup {
            class,
            depth,
            tried,
        } => format!("next_group:{class}@{depth}:{}", deployment_ids(tried)),
        RoutingOp::Backoff { duration, .. } => format!("sleep:{}", duration.as_millis()),
    }));
}

pub type PickFn = fn(&[Candidate]) -> Option<DeploymentId>;

pub fn first(candidates: &[Candidate]) -> Option<DeploymentId> {
    candidates.first().map(|c| c.id)
}

/// The host side of one scenario: what it answers when the router asks for a plan, a pick
/// or the next group, which failures cool a deployment down, and how long it sleeps.
pub struct ScenarioHost {
    pub inner: RecordingHost<TestError>,
    pub plan: Option<RoutePlan>,
    pub pick: PickFn,
    pub groups: VecDeque<Option<Vec<Deployment>>>,
    pub cooldowns: Cooldowns,
    pub cool_on: Vec<FailureClass>,
    pub clock: FakeClock,
}

impl ScenarioHost {
    fn routing(&mut self, op: RoutingOp) -> Result<RoutingResult, TestError> {
        record(&mut self.inner.trace, &op);
        Ok(match op {
            RoutingOp::ResolvePlan => RoutingResult::Plan(Box::new(
                self.plan
                    .clone()
                    .expect("the scenario gave the host no plan to answer with"),
            )),
            RoutingOp::Pick { candidates, .. } => RoutingResult::Picked((self.pick)(&candidates)),
            RoutingOp::NextGroup { .. } => RoutingResult::Group(
                self.groups
                    .pop_front()
                    .expect("the scenario gave the host no next group to answer with"),
            ),
            RoutingOp::Backoff { duration, .. } => {
                self.clock.advance(duration);
                RoutingResult::Slept
            }
        })
    }
}

impl<A> Answers<Layered<Routing<A>, TestRoute<TestError>>> for ScenarioHost
where
    A: Attempt + Machine<Route = TestRoute<TestError>> + 'static,
{
    fn trace(&mut self) -> &mut Trace {
        &mut self.inner.trace
    }

    fn route(
        &mut self,
        op: LayeredOp<RoutingOp, String>,
    ) -> Result<LayeredResult<RoutingResult, String>, TestError> {
        match op {
            LayeredOp::Outer(op) => self.routing(op).map(LayeredResult::Outer),
            LayeredOp::Inner(op) => self.inner.route(op).map(LayeredResult::Inner),
        }
    }

    fn attempt_failed(&mut self, attempt: &AttemptInfo, class: FailureClass) {
        if self.cool_on.contains(&class) {
            self.cooldowns.cool(DeploymentId(attempt.deployment));
        }
    }
}

/// Everything one case states about the world: the plan, the attempts in the order the
/// router will start them, and the host's script.
pub struct Scenario {
    pub plan: PlanSource,
    pub attempts: Vec<Script<TestError>>,
    /// The host picks (through `Pick` ops) instead of the local round robin.
    pub host_picks: bool,
    pub pick: PickFn,
    /// The host's answer to `ResolvePlan`, for `PlanSource::Host`.
    pub host_plan: Option<RoutePlan>,
    /// The host's answers to `NextGroup`, in order, for `Fallbacks::Host`.
    pub groups: Vec<Option<Vec<Deployment>>>,
    /// Failure classes that cool the failed deployment down when the host records them.
    pub cool_on: Vec<FailureClass>,
    /// Deployments already cooling down when the call starts.
    pub precooled: Vec<u64>,
}

pub fn scenario(plan: impl Into<PlanSource>, attempts: Vec<Script<TestError>>) -> Scenario {
    Scenario {
        plan: plan.into(),
        attempts,
        host_picks: false,
        pick: first,
        host_plan: None,
        groups: Vec::new(),
        cool_on: Vec::new(),
        precooled: Vec::new(),
    }
}

pub struct Run {
    pub trace: Trace,
    pub report: CallReport<String, TestError>,
    pub started: Vec<AttemptContext>,
    pub elapsed: Duration,
}

impl Run {
    /// Every attempt as `(deployment, group)`, in order.
    pub fn visited(&self) -> Vec<(u64, u32)> {
        self.started
            .iter()
            .map(|context| (context.deployment.0, context.group_index))
            .collect()
    }

    /// What the loop decided after each attempt: `retry`, `retry-skip`, `fallback`, `stop`
    /// or `ok`.
    pub fn decisions(&self) -> Vec<&'static str> {
        self.report
            .attempts
            .iter()
            .map(|attempt| match attempt.failure {
                None => "ok",
                Some((
                    _,
                    Decision::Retry {
                        skip_failed: false, ..
                    },
                )) => "retry",
                Some((
                    _,
                    Decision::Retry {
                        skip_failed: true, ..
                    },
                )) => "retry-skip",
                Some((_, Decision::Fallback)) => "fallback",
                Some((_, Decision::Stop)) => "stop",
            })
            .collect()
    }
}

pub async fn run(scenario: Scenario) -> Run {
    let cooldowns = Cooldowns::default();
    for id in &scenario.precooled {
        cooldowns.cool(DeploymentId(*id));
    }
    let picker = if scenario.host_picks {
        Picker::Host
    } else {
        Picker::local(RoundRobin::default(), cooldowns.clone())
    };
    let clock = FakeClock::default();
    let attempts = Attempts::new(scenario.attempts);
    let mut stack = Stack::new(attempts.clone())
        .layer(RouterLayer::new(scenario.plan, picker, clock.clone(), 7))
        .build();
    let mut host = ScenarioHost {
        inner: RecordingHost::echo(),
        plan: scenario.host_plan,
        pick: scenario.pick,
        groups: scenario.groups.into(),
        cooldowns,
        cool_on: scenario.cool_on,
        clock: clock.clone(),
    };
    let report = drain(&mut stack, &mut host)
        .await
        .expect("a router completes with a report, never an error");
    let started = attempts.started.lock().unwrap().clone();
    Run {
        trace: host.inner.trace,
        report,
        started,
        elapsed: clock.elapsed(),
    }
}

/// The outcome a case expects, without the error's identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Exhausted,
    Budget,
    MidStream(u32),
    Protocol,
}

pub fn outcome(report: &CallReport<String, TestError>) -> Outcome {
    match &report.outcome {
        Ok(_) => Outcome::Ok,
        Err(CallFailure::Exhausted { .. }) => Outcome::Exhausted,
        Err(CallFailure::Budget { .. }) => Outcome::Budget,
        Err(CallFailure::MidStream { chunks, .. }) => Outcome::MidStream(*chunks),
        Err(CallFailure::Protocol(_)) => Outcome::Protocol,
        Err(CallFailure::Interrupted { .. }) => panic!("no scenario interrupts the call"),
    }
}
