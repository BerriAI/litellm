//! Behavior of the assembled stack: Router over per-attempt Cache over the route machine.
//! Each test states the attempts, drives the stack through a recording host that emits the
//! terminal event, and asserts the exact op trace and the report. The router's own ops (plan resolution,
//! picking) reach the host through the outer half of the layered route.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use litellm_cache::{
    BaseCache, CacheBackend, CacheConnectionResult, CacheConnectionStatus, CacheControls,
    CacheEntry, CacheFuture, CacheKwargs, CacheLayer, Cached,
};
use litellm_callbacks::layer::Stack;
use litellm_callbacks_test::{
    Answer, Observed, RecordingHost, Script, Scripted, TestRoute, Trace, WithOuter, assert_trace,
    drain,
};
use litellm_router::{
    AttemptContext, AttemptDisposition, AttemptError, AttemptFactory, CallFailure, CallReport,
    Clock, Deployment, DeploymentId, Layered, Machine, NoSignals, PerAttempt, Picker, PlanSource,
    RetryPolicy, RoundRobin, RoutePlan, RouterLayer, Routing, RoutingOp, RoutingResult,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestError(AttemptDisposition);

impl AttemptError for TestError {
    fn disposition(&self) -> AttemptDisposition {
        self.0.clone()
    }
}

fn retryable() -> TestError {
    TestError(AttemptDisposition::Retryable { retry_after: None })
}

fn fatal() -> TestError {
    TestError(AttemptDisposition::Fatal)
}

fn reroute() -> TestError {
    TestError(AttemptDisposition::Reroute { cooldown: None })
}

type Outer = fn(&mut Trace, RoutingOp) -> Result<RoutingResult, TestError>;

fn never_asked(_: &mut Trace, op: RoutingOp) -> Result<RoutingResult, TestError> {
    panic!("local routing never asks the host: {op:?}")
}

/// A host for a stack whose routing stays in process: the outer half must never be asked.
fn local<H>(inner: H) -> WithOuter<Outer, H> {
    WithOuter {
        outer: never_asked,
        inner,
    }
}

/// Records a routing op the way the inner host records route ops, so one trace shows both.
fn record(trace: &mut Trace, op: &RoutingOp) {
    trace.0.push(Observed::Route(match op {
        RoutingOp::ResolvePlan => "resolve_plan".into(),
        RoutingOp::Pick { group, candidates } => {
            let ids: Vec<String> = candidates.iter().map(|c| c.id.0.to_string()).collect();
            format!("pick:{group}:[{}]", ids.join(","))
        }
    }));
}

#[derive(Clone)]
struct Attempts {
    scripts: Arc<Mutex<Vec<Script<TestError>>>>,
    started: Arc<Mutex<Vec<AttemptContext>>>,
}

impl Attempts {
    fn new(scripts: Vec<Script<TestError>>) -> Self {
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
            "router started more attempts than scripted"
        );
        Scripted::new(scripts.remove(0))
    }
}

#[derive(Clone, Default)]
struct FakeClock(Arc<Mutex<Duration>>);

impl Clock for FakeClock {
    fn now(&self) -> Instant {
        static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        *ORIGIN.get_or_init(Instant::now) + *self.0.lock().unwrap()
    }

    fn sleep(&self, duration: Duration) -> impl Future<Output = ()> + Send {
        *self.0.lock().unwrap() += duration;
        async {}
    }
}

#[derive(Default)]
struct Memory {
    entries: Mutex<HashMap<String, CacheEntry>>,
    gets: Mutex<u32>,
}

impl BaseCache for Memory {
    type Value = CacheEntry;

    fn set_cache(
        &self,
        key: &str,
        value: CacheEntry,
        _: CacheKwargs,
    ) -> Result<(), litellm_cache::Error> {
        self.entries.lock().unwrap().insert(key.into(), value);
        Ok(())
    }

    fn get_cache(
        &self,
        key: &str,
        _: &CacheKwargs,
    ) -> Result<Option<CacheEntry>, litellm_cache::Error> {
        *self.gets.lock().unwrap() += 1;
        Ok(self.entries.lock().unwrap().get(key).cloned())
    }

    fn delete_cache(&self, key: &str) -> Result<(), litellm_cache::Error> {
        self.entries.lock().unwrap().remove(key);
        Ok(())
    }

    fn flush_cache(&self) -> Result<(), litellm_cache::Error> {
        self.entries.lock().unwrap().clear();
        Ok(())
    }

    fn disconnect(&self) -> CacheFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn test_connection(&self) -> CacheFuture<'_, CacheConnectionResult> {
        Box::pin(async {
            Ok(CacheConnectionResult {
                status: CacheConnectionStatus::Success,
                message: String::new(),
                error: None,
            })
        })
    }
}

fn caching() -> CacheControls {
    CacheControls {
        supported_call_type: true,
        configured: true,
        native_backend: true,
        default_on: true,
        caching: None,
        no_cache: false,
        no_store: false,
        use_cache: false,
    }
}

fn plan(deployments: &[u64], max_retries: u32) -> RoutePlan {
    RoutePlan::single(
        deployments
            .iter()
            .map(|id| Deployment::new(DeploymentId(*id)))
            .collect(),
        RetryPolicy {
            max_retries,
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(40),
            jitter: false,
            respect_retry_after: false,
        },
    )
}

struct Harness {
    attempts: Attempts,
    memory: Arc<Memory>,
    clock: FakeClock,
}

impl Harness {
    fn new(scripts: Vec<Script<TestError>>) -> Self {
        Self {
            attempts: Attempts::new(scripts),
            memory: Arc::default(),
            clock: FakeClock::default(),
        }
    }

    fn stack(
        &self,
        plan: impl Into<PlanSource>,
        picker: Picker,
    ) -> impl Machine<
        Route = Layered<Routing<Cached<Scripted<TestError>>>, TestRoute<TestError>>,
        Complete = CallReport<String, TestError>,
    > {
        let backend: CacheBackend = self.memory.clone();
        Stack::new(self.attempts.clone())
            .layer(PerAttempt(CacheLayer::new(
                backend,
                "key".into(),
                caching(),
            )))
            .layer(RouterLayer::new(plan, picker, self.clock.clone(), 1))
            .build()
    }

    fn local_stack(
        &self,
        plan: RoutePlan,
    ) -> impl Machine<
        Route = Layered<Routing<Cached<Scripted<TestError>>>, TestRoute<TestError>>,
        Complete = CallReport<String, TestError>,
    > {
        self.stack(plan, Picker::local(RoundRobin::default(), NoSignals))
    }
}

#[tokio::test]
async fn success_after_a_retry_yields_every_op_then_one_terminal_and_stores_once() {
    let harness = Harness::new(vec![
        Script::err(&["project", "send"], retryable()),
        Script::ok(&["project", "send"], "response"),
    ]);
    let mut stack = harness.local_stack(plan(&[1], 3));
    let mut host = local(RecordingHost::echo());

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_trace!(
        host.inner.trace,
        [
            "emit:attempt_started",
            "route:project",
            "route:send",
            "emit:attempt_failed",
            "emit:attempt_started",
            "route:project",
            "route:send",
            "emit:succeeded"
        ]
    );
    host.inner.trace.assert_one_terminal();
    assert_eq!(
        host.inner.trace.emitted("attempt_started"),
        report.attempts.len()
    );
    assert_eq!(report.outcome, Ok("response".into()));
    assert_eq!(report.attempts.len(), 2);
    assert_eq!(report.selected, Some(DeploymentId(1)));
    assert_eq!(harness.memory.entries.lock().unwrap().len(), 1);
    assert_eq!(
        *harness.memory.gets.lock().unwrap(),
        2,
        "one lookup per attempt"
    );
    let started = harness.attempts.started.lock().unwrap();
    assert!(started.iter().all(|context| context.id == report.id));
}

#[tokio::test]
async fn cache_hit_completes_without_route_ops_and_still_emits_one_terminal() {
    let harness = Harness::new(vec![Script::ok(&["never"], "unused")]);
    harness.memory.entries.lock().unwrap().insert(
        "key".into(),
        CacheEntry {
            timestamp: litellm_callbacks::event::epoch_seconds(),
            response: serde_json::json!("cached"),
        },
    );
    let mut stack = harness.local_stack(plan(&[1], 3));
    let mut host = local(RecordingHost::echo());

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_trace!(host.inner.trace, ["emit:attempt_started", "emit:succeeded"]);
    assert_eq!(report.outcome, Ok("cached".into()));
    assert_eq!(
        report.attempts.len(),
        1,
        "a hit is one attempt with no provider op"
    );
    assert!(report.attempts[0].failed.is_none());
}

#[tokio::test]
async fn exhausted_retries_emit_failed_once_and_report_the_last_error() {
    let harness = Harness::new(vec![
        Script::err(&["send"], retryable()),
        Script::err(&["send"], retryable()),
    ]);
    let mut stack = harness.local_stack(plan(&[1], 1));
    let mut host = local(RecordingHost::echo());

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_trace!(
        host.inner.trace,
        [
            "emit:attempt_started",
            "route:send",
            "emit:attempt_failed",
            "emit:attempt_started",
            "route:send",
            "emit:attempt_failed",
            "emit:failed"
        ]
    );
    host.inner.trace.assert_one_terminal();
    assert_eq!(host.inner.trace.emitted("attempt_failed"), 2);
    assert_eq!(
        report.outcome,
        Err(CallFailure::Exhausted { last: retryable() })
    );
    assert!(
        harness.memory.entries.lock().unwrap().is_empty(),
        "failures are not cached"
    );
    assert_eq!(*harness.clock.0.lock().unwrap(), Duration::from_millis(10));
}

#[tokio::test]
async fn fatal_stops_after_the_first_attempt_even_with_budget_left() {
    let harness = Harness::new(vec![Script::err(&["send"], fatal())]);
    let mut stack = harness.local_stack(plan(&[1, 2], 5));
    let mut host = local(RecordingHost::echo());

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_trace!(
        host.inner.trace,
        [
            "emit:attempt_started",
            "route:send",
            "emit:attempt_failed",
            "emit:failed"
        ]
    );
    assert_eq!(report.outcome, Err(CallFailure::Fatal { error: fatal() }));
    assert_eq!(harness.attempts.started.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn host_failure_mid_attempt_interrupts_the_whole_call_without_a_retry() {
    let harness = Harness::new(vec![Script::ok(&["project", "send"], "unreachable")]);
    let mut stack = harness.local_stack(plan(&[1, 2], 5));
    let mut host = local(RecordingHost::with(|op| {
        if op == "send" {
            Answer::Fail(retryable())
        } else {
            Answer::Value(op.to_string())
        }
    }));

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_trace!(
        host.inner.trace,
        [
            "emit:attempt_started",
            "route:project",
            "route:send",
            "emit:failed"
        ]
    );
    host.inner.trace.assert_one_terminal();
    assert_eq!(
        report.outcome,
        Err(CallFailure::Interrupted { error: retryable() })
    );
    assert_eq!(harness.attempts.started.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn host_answers_reach_the_attempt_in_order() {
    let harness = Harness::new(vec![Script::ok(&["a", "b", "c"], "done")]);
    let mut stack = harness.local_stack(plan(&[1], 0));
    let mut host = local(RecordingHost::with(|op| Answer::Value(format!("{op}!"))));

    drain(&mut stack, &mut host).await.unwrap();

    assert_eq!(host.inner.trace.route_ops(), ["a", "b", "c"]);
}

#[tokio::test]
async fn host_picking_is_asked_once_per_selection_with_the_remaining_candidates() {
    let harness = Harness::new(vec![
        Script::err(&["send"], reroute()),
        Script::ok(&["send"], "response"),
    ]);
    let mut stack = harness.stack(plan(&[1, 2], 0), Picker::Host);
    let mut host = WithOuter {
        outer: |trace: &mut Trace, op: RoutingOp| {
            record(trace, &op);
            let RoutingOp::Pick { candidates, .. } = op else {
                panic!("a resolved plan is never asked for again")
            };
            Ok(RoutingResult::Picked(candidates.last().map(|c| c.id)))
        },
        inner: RecordingHost::echo(),
    };

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_trace!(
        host.inner.trace,
        [
            "route:pick:0:[1,2]",
            "emit:attempt_started",
            "route:send",
            "emit:attempt_failed",
            "route:pick:0:[1]",
            "emit:attempt_started",
            "route:send",
            "emit:succeeded"
        ]
    );
    assert_eq!(report.outcome, Ok("response".into()));
    assert_eq!(report.selected, Some(DeploymentId(1)));
    let started = harness.attempts.started.lock().unwrap();
    let visited: Vec<_> = started.iter().map(|c| c.deployment).collect();
    assert_eq!(visited, [DeploymentId(2), DeploymentId(1)]);
}

#[tokio::test]
async fn host_resolved_plans_are_asked_for_before_anything_else() {
    let harness = Harness::new(vec![Script::ok(&["send"], "response")]);
    let mut stack = harness.stack(
        PlanSource::Host,
        Picker::local(RoundRobin::default(), NoSignals),
    );
    let mut host = WithOuter {
        outer: |trace: &mut Trace, op: RoutingOp| {
            record(trace, &op);
            Ok(RoutingResult::Plan(plan(&[7], 0)))
        },
        inner: RecordingHost::echo(),
    };

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_trace!(
        host.inner.trace,
        [
            "route:resolve_plan",
            "emit:attempt_started",
            "route:send",
            "emit:succeeded"
        ]
    );
    assert_eq!(report.selected, Some(DeploymentId(7)));
}

#[tokio::test]
async fn picking_nothing_moves_to_the_next_group() {
    let harness = Harness::new(vec![Script::ok(&["send"], "fallback")]);
    let two_groups = RoutePlan {
        groups: vec![
            vec![Deployment::new(DeploymentId(1))],
            vec![Deployment::new(DeploymentId(2))],
        ],
        ..plan(&[], 0)
    };
    let mut stack = harness.stack(two_groups, Picker::Host);
    let mut host = WithOuter {
        outer: |trace: &mut Trace, op: RoutingOp| {
            record(trace, &op);
            Ok(RoutingResult::Picked(match op {
                RoutingOp::Pick { group: 0, .. } => None,
                _ => Some(DeploymentId(2)),
            }))
        },
        inner: RecordingHost::echo(),
    };

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_trace!(
        host.inner.trace,
        [
            "route:pick:0:[1]",
            "route:pick:1:[2]",
            "emit:attempt_started",
            "route:send",
            "emit:succeeded"
        ]
    );
    assert_eq!(report.outcome, Ok("fallback".into()));
    let started = harness.attempts.started.lock().unwrap();
    assert_eq!(
        (started[0].deployment, started[0].group_index),
        (DeploymentId(2), 1)
    );
}

#[tokio::test]
async fn a_pick_outside_the_candidates_or_a_wrong_variant_fails_the_call_as_protocol() {
    let harness = Harness::new(vec![Script::ok(&["never"], "unused")]);
    let mut stack = harness.stack(plan(&[1, 2], 3), Picker::Host);
    let mut host = WithOuter {
        outer: |trace: &mut Trace, op: RoutingOp| {
            record(trace, &op);
            Ok(RoutingResult::Picked(Some(DeploymentId(99))))
        },
        inner: RecordingHost::echo(),
    };
    let report = drain(&mut stack, &mut host).await.unwrap();
    assert_trace!(host.inner.trace, ["route:pick:0:[1,2]", "emit:failed"]);
    assert!(matches!(report.outcome, Err(CallFailure::Protocol(_))));
    assert!(harness.attempts.started.lock().unwrap().is_empty());

    let mut stack = harness.stack(PlanSource::Host, Picker::Host);
    let mut host = WithOuter {
        outer: |trace: &mut Trace, op: RoutingOp| {
            record(trace, &op);
            Ok(RoutingResult::Picked(None))
        },
        inner: RecordingHost::echo(),
    };
    let report = drain(&mut stack, &mut host).await.unwrap();
    assert_trace!(host.inner.trace, ["route:resolve_plan", "emit:failed"]);
    assert!(matches!(report.outcome, Err(CallFailure::Protocol(_))));
}
