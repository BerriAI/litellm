//! Behavior of the assembled stack: Router over per-attempt Cache over the route machine.
//! Each test states the attempts, drives the stack through the scenario host and asserts
//! the exact trace and the report.

mod support;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use litellm_cache::{
    BaseCache, CacheBackend, CacheConnectionResult, CacheConnectionStatus, CacheControls,
    CacheEntry, CacheFuture, CacheKwargs, CacheLayer, Cached,
};
use litellm_callbacks::layer::Stack;
use litellm_callbacks_test::{
    Answer, RecordingHost, Script, Scripted, TestRoute, assert_trace, drain,
};
use litellm_router::{
    CallFailure, CallReport, DeploymentId, FailureClass, Layered, Machine, PerAttempt, Picker,
    RoundRobin, RouterLayer, Routing,
};
use support::{Attempts, Cooldowns, FakeClock, ScenarioHost, TestError, failure, plan};

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
        ids: &[u64],
        retries: u32,
        stream: bool,
    ) -> impl Machine<
        Route = Layered<Routing<Cached<Scripted<TestError>>>, TestRoute<TestError>>,
        Complete = CallReport<String, TestError>,
    > {
        let backend: CacheBackend = self.memory.clone();
        Stack::new(self.attempts.clone())
            .layer(PerAttempt(
                CacheLayer::new(backend, "key".into(), caching()).with_stream(stream),
            ))
            .layer(RouterLayer::new(
                plan(ids, retries),
                Picker::local(RoundRobin::default(), Cooldowns::default()),
                self.clock.clone(),
                1,
            ))
            .build()
    }

    fn host(&self, inner: RecordingHost<TestError>) -> ScenarioHost {
        ScenarioHost {
            inner,
            plan: None,
            pick: support::first,
            groups: Default::default(),
            cooldowns: Cooldowns::default(),
            cool_on: Vec::new(),
            clock: self.clock.clone(),
        }
    }
}

fn retryable() -> TestError {
    failure(FailureClass::InternalServer)
}

#[tokio::test]
async fn success_after_a_retry_yields_every_op_then_one_terminal_and_stores_once() {
    let harness = Harness::new(vec![
        Script::err(&["project", "send"], retryable()),
        Script::ok(&["project", "send"], "response"),
    ]);
    let mut stack = harness.stack(&[1], 3, false);
    let mut host = harness.host(RecordingHost::echo());

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_trace!(
        host.inner.trace,
        [
            "emit:attempt_started",
            "route:project",
            "route:send",
            "failed:0:InternalServer",
            "route:sleep:500",
            "emit:attempt_started",
            "route:project",
            "route:send",
            "emit:succeeded"
        ]
    );
    host.inner.trace.assert_one_terminal();
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
    assert!(
        started
            .iter()
            .all(|context| context.trace_id == report.trace_id)
    );
}

#[tokio::test]
async fn cache_hits_complete_without_route_ops_and_replay_chunks_when_streaming() {
    for (stream, expected) in [
        (false, vec!["emit:attempt_started", "emit:succeeded"]),
        (
            true,
            vec!["emit:attempt_started", "yield:cached", "emit:succeeded"],
        ),
    ] {
        let harness = Harness::new(vec![Script::ok(&["never"], "unused")]);
        harness.memory.entries.lock().unwrap().insert(
            "key".into(),
            CacheEntry {
                timestamp: litellm_callbacks::event::epoch_seconds(),
                response: serde_json::json!("cached"),
            },
        );
        let mut stack = harness.stack(&[1], 3, stream);
        let mut host = harness.host(RecordingHost::echo());

        let report = drain(&mut stack, &mut host).await.unwrap();

        assert_eq!(host.inner.trace.lines(), expected);
        assert_eq!(report.outcome, Ok("cached".into()));
        assert_eq!(
            report.attempts.len(),
            1,
            "a hit is one attempt with no provider op"
        );
        assert!(report.attempts[0].failure.is_none());
    }
}

#[tokio::test]
async fn failures_are_not_cached() {
    let harness = Harness::new(vec![
        Script::err(&["send"], retryable()),
        Script::err(&["send"], retryable()),
    ]);
    let mut stack = harness.stack(&[1], 1, false);
    let mut host = harness.host(RecordingHost::echo());

    let report = drain(&mut stack, &mut host).await.unwrap();

    assert_eq!(host.inner.trace.attempt_failures(), 2);
    assert_eq!(
        report.outcome,
        Err(CallFailure::Exhausted { last: retryable() })
    );
    assert!(harness.memory.entries.lock().unwrap().is_empty());
}

#[tokio::test]
async fn host_failure_mid_attempt_interrupts_the_whole_call_without_a_retry() {
    let harness = Harness::new(vec![Script::ok(&["project", "send"], "unreachable")]);
    let mut stack = harness.stack(&[1, 2], 5, false);
    let mut host = harness.host(RecordingHost::with(|op| {
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
    let mut stack = harness.stack(&[1], 0, false);
    let mut host = harness.host(RecordingHost::with(|op| Answer::Value(format!("{op}!"))));

    drain(&mut stack, &mut host).await.unwrap();

    assert_eq!(host.inner.trace.route_ops(), ["a", "b", "c"]);
}
