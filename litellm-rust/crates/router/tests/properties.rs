//! Properties that hold for any sequence of attempt outcomes and any plan: exactly one
//! terminal event, attempt records that match the events and never exceed the plan's
//! bounds, and an outcome that agrees with what the scripts could produce.

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use litellm_callbacks::layer::Stack;
use litellm_callbacks_test::{RecordingHost, Script, Scripted, Trace, WithOuter, drain};
use litellm_router::{
    AttemptContext, AttemptDisposition, AttemptError, AttemptFactory, CallFailure, Clock,
    Deployment, DeploymentId, NoSignals, Picker, RetryPolicy, RoundRobin, RoutePlan, RouterLayer,
    RoutingOp, RoutingResult,
};
use proptest::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestError(AttemptDisposition);

impl AttemptError for TestError {
    fn disposition(&self) -> AttemptDisposition {
        self.0.clone()
    }
}

#[derive(Clone)]
struct Attempts(Arc<Mutex<Vec<Script<TestError>>>>);

impl AttemptFactory for Attempts {
    type Attempt = Scripted<TestError>;

    fn start(&self, _: &AttemptContext) -> Scripted<TestError> {
        let mut scripts = self.0.lock().unwrap();
        assert!(
            !scripts.is_empty(),
            "the loop ran past the scripted attempts"
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

fn disposition() -> impl Strategy<Value = Option<AttemptDisposition>> {
    prop_oneof![
        3 => Just(None),
        3 => (0u64..50).prop_map(|ms| Some(AttemptDisposition::Retryable {
            retry_after: (ms > 0).then(|| Duration::from_millis(ms)),
        })),
        2 => (0u64..50).prop_map(|ms| Some(AttemptDisposition::Reroute {
            cooldown: (ms > 0).then(|| Duration::from_millis(ms)),
        })),
        1 => Just(Some(AttemptDisposition::Fatal)),
    ]
}

fn scripts() -> impl Strategy<Value = Vec<Script<TestError>>> {
    prop::collection::vec((disposition(), 0usize..3), 40).prop_map(|outcomes| {
        outcomes
            .into_iter()
            .enumerate()
            .map(|(index, (disposition, op_count))| {
                let ops: Vec<String> = (0..op_count).map(|op| format!("op{index}-{op}")).collect();
                let ops: Vec<&str> = ops.iter().map(String::as_str).collect();
                match disposition {
                    None => Script::ok(&ops, &format!("response{index}")),
                    Some(disposition) => Script::err(&ops, TestError(disposition)),
                }
            })
            .collect()
    })
}

fn plans() -> impl Strategy<Value = RoutePlan> {
    (
        prop::collection::vec(prop::collection::vec(1u64..6, 1..4), 1..3),
        0u32..4,
        prop::option::of(1u32..8),
        any::<bool>(),
    )
        .prop_map(|(groups, max_retries, attempt_budget, jitter)| RoutePlan {
            groups: groups
                .into_iter()
                .map(|group| {
                    group
                        .into_iter()
                        .map(|id| Deployment::new(DeploymentId(id)))
                        .collect()
                })
                .collect(),
            retry: RetryPolicy {
                max_retries,
                initial_backoff: Duration::from_millis(5),
                max_backoff: Duration::from_millis(20),
                jitter,
                respect_retry_after: true,
            },
            attempt_budget,
            timeout: None,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn any_plan_and_any_outcomes_end_in_exactly_one_terminal(
        plan in plans(),
        scripts in scripts(),
        seed in any::<u64>(),
        host_picks in any::<bool>(),
        choice in any::<usize>(),
    ) {
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(async move {
            let max_attempts = plan.groups.iter().map(Vec::len).sum::<usize>()
                * (plan.retry.max_retries as usize + 1);
            let budget = plan.attempt_budget.map_or(max_attempts, |b| b as usize);
            let clock = FakeClock::default();
            let picker = if host_picks {
                Picker::Host
            } else {
                Picker::local(RoundRobin::default(), NoSignals)
            };
            let mut stack = Stack::new(Attempts(Arc::new(Mutex::new(scripts.clone()))))
                .layer(RouterLayer::new(plan.clone(), picker, clock.clone(), seed))
                .build();
            let picks = Arc::new(Mutex::new(0usize));
            let counted = Arc::clone(&picks);
            let mut host = WithOuter {
                outer: move |_: &mut Trace, op: RoutingOp| {
                    let RoutingOp::Pick { candidates, .. } = op else {
                        panic!("the plan was given, never resolved")
                    };
                    *counted.lock().unwrap() += 1;
                    Ok(RoutingResult::Picked(Some(candidates[choice % candidates.len()].id)))
                },
                inner: RecordingHost::echo(),
            };

            let report = drain(&mut stack, &mut host).await.unwrap();

            let host = host.inner;
            host.trace.assert_one_terminal();
            let mut expected_picks = 0;
            let mut retries = 0;
            for (index, attempt) in report.attempts.iter().enumerate() {
                let picked = match index.checked_sub(1).map(|i| &report.attempts[i].failed) {
                    None => true,
                    Some(Some(AttemptDisposition::Retryable { .. })) => {
                        retries += 1;
                        retries > plan.retry.max_retries
                    }
                    Some(_) => true,
                };
                if picked {
                    retries = 0;
                    expected_picks += 1;
                } else {
                    prop_assert_eq!(attempt.deployment, report.attempts[index - 1].deployment);
                }
            }
            prop_assert_eq!(*picks.lock().unwrap(), if host_picks { expected_picks } else { 0 });
            prop_assert_eq!(host.trace.emitted("attempt_started"), report.attempts.len());
            prop_assert_eq!(
                host.trace.emitted("attempt_failed"),
                report.attempts.iter().filter(|a| a.failed.is_some()).count()
            );
            prop_assert!(report.attempts.len() <= max_attempts.min(budget));
            prop_assert!(report.attempts.windows(2).all(|pair| pair[0].started <= pair[1].started));
            prop_assert!(report.attempts.iter().all(|a| a.started <= a.ended));

            let ran = &scripts[..report.attempts.len()];
            match &report.outcome {
                Ok(value) => {
                    prop_assert_eq!(ran.last().map(|s| s.outcome.clone()), Some(Ok(value.clone())));
                    prop_assert!(ran[..ran.len() - 1].iter().all(|s| s.outcome.is_err()));
                }
                Err(CallFailure::Fatal { error }) => {
                    prop_assert_eq!(error, &TestError(AttemptDisposition::Fatal));
                    prop_assert_eq!(ran.last().map(|s| s.outcome.clone()), Some(Err(error.clone())));
                }
                Err(CallFailure::Exhausted { last }) | Err(CallFailure::Budget { last: Some(last) }) => {
                    prop_assert!(ran.iter().all(|s| s.outcome.is_err()));
                    prop_assert_eq!(ran.last().map(|s| s.outcome.clone()), Some(Err(last.clone())));
                }
                Err(CallFailure::Budget { last: None }) => prop_assert!(ran.is_empty()),
                Err(CallFailure::Interrupted { .. }) => prop_assert!(false, "no host failure was scripted"),
                Err(CallFailure::Protocol(violation)) => prop_assert!(false, "{violation}"),
            }
            Ok(())
        })?;
    }
}
