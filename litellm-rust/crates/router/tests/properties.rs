//! Properties that hold for any plan and any sequence of attempt outcomes: exactly one
//! terminal event, attempt records that match the events and never exceed the plan's
//! bounds, a skipped deployment never picked again in its group, no attempt after a
//! failure mid-stream, and an outcome that agrees with what the scripts could produce.

mod support;

use std::collections::VecDeque;

use litellm_callbacks::layer::Stack;
use litellm_callbacks_test::{RecordingHost, Script, drain};
use litellm_router::{
    CallFailure, Decision, FailureClass, FallbackChains, Fallbacks, Picker, Retries, RoundRobin,
    RoutePlan, RouterLayer,
};
use proptest::prelude::*;
use support::{
    Attempts, Cooldowns, FakeClock, ScenarioHost, TestError, deployments, python_retries,
};

fn class() -> impl Strategy<Value = FailureClass> {
    prop_oneof![
        Just(FailureClass::RateLimited),
        Just(FailureClass::Timeout),
        Just(FailureClass::InternalServer),
        Just(FailureClass::Connection),
        Just(FailureClass::Authentication),
        Just(FailureClass::NotFound),
        Just(FailureClass::ContextWindow),
        Just(FailureClass::BadRequest),
    ]
}

fn scripts() -> impl Strategy<Value = Vec<Script<TestError>>> {
    prop::collection::vec((prop::option::of(class()), 0usize..3, 0usize..3), 40).prop_map(
        |outcomes| {
            outcomes
                .into_iter()
                .enumerate()
                .map(|(index, (class, op_count, chunk_count))| {
                    let ops: Vec<String> =
                        (0..op_count).map(|op| format!("op{index}-{op}")).collect();
                    let ops: Vec<&str> = ops.iter().map(String::as_str).collect();
                    let chunks: Vec<String> =
                        (0..chunk_count).map(|c| format!("c{index}-{c}")).collect();
                    let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();
                    let response = format!("response{index}");
                    match class {
                        None => Script::stream(&ops, &chunks, Ok(&response)),
                        Some(class) => Script::stream(&ops, &chunks, Err(support::failure(class))),
                    }
                })
                .collect()
        },
    )
}

fn plans() -> impl Strategy<Value = RoutePlan> {
    (
        prop::collection::vec(1u64..6, 1..4),
        prop::collection::vec(prop::collection::vec(1u64..6, 1..4), 0..3),
        0u32..4,
        prop::option::of(1u32..8),
    )
        .prop_map(|(primary, chains, retries, attempt_budget)| RoutePlan {
            primary: deployments(&primary),
            retry: python_retries(Retries::Default(retries)),
            fallbacks: if chains.is_empty() {
                Fallbacks::Disabled
            } else {
                Fallbacks::Chains(FallbackChains::generic(
                    chains.iter().map(|group| deployments(group)).collect(),
                ))
            },
            attempt_budget,
            timeout: None,
        })
}

fn group_count(plan: &RoutePlan) -> usize {
    1 + match &plan.fallbacks {
        Fallbacks::Chains(chains) => chains.generic.len().min(chains.max_depth as usize),
        _ => 0,
    }
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
            let retries = plan.retry.retries;
            let Retries::Default(retries) = retries else { unreachable!() };
            let max_attempts = group_count(&plan) * (retries as usize + 1);
            let budget = plan.attempt_budget.map_or(max_attempts, |b| b as usize);
            let clock = FakeClock::default();
            let picker = if host_picks {
                Picker::Host
            } else {
                Picker::local(RoundRobin::default(), Cooldowns::default())
            };
            let attempts = Attempts::new(scripts.clone());
            let mut stack = Stack::new(attempts.clone())
                .layer(RouterLayer::new(plan.clone(), picker, clock.clone(), seed))
                .build();
            fn pick_first(candidates: &[litellm_router::Candidate]) -> Option<litellm_router::DeploymentId> {
                candidates.first().map(|c| c.id)
            }
            fn pick_last(candidates: &[litellm_router::Candidate]) -> Option<litellm_router::DeploymentId> {
                candidates.last().map(|c| c.id)
            }
            let mut host = ScenarioHost {
                inner: RecordingHost::echo(),
                plan: None,
                pick: if choice % 2 == 0 { pick_first } else { pick_last },
                groups: VecDeque::new(),
                cooldowns: Cooldowns::default(),
                cool_on: Vec::new(),
                clock,
            };

            let report = drain(&mut stack, &mut host).await.unwrap();

            let trace = host.inner.trace;
            trace.assert_one_terminal();
            let started = attempts.started.lock().unwrap().clone();
            prop_assert_eq!(trace.emitted("attempt_started"), report.attempts.len());
            prop_assert_eq!(started.len(), report.attempts.len());
            prop_assert_eq!(
                trace.attempt_failures(),
                report.attempts.iter().filter(|a| a.failure.is_some()).count()
            );
            prop_assert!(report.attempts.len() <= max_attempts.min(budget));
            prop_assert!(report.attempts.windows(2).all(|pair| pair[0].started <= pair[1].started));
            prop_assert!(report.attempts.iter().all(|a| a.started <= a.ended));
            prop_assert!(report.attempts.windows(2).all(|pair| pair[0].group_index <= pair[1].group_index));

            for (index, attempt) in report.attempts.iter().enumerate() {
                if let Some((_, Decision::Retry { skip_failed: true, .. })) = attempt.failure {
                    let repicked = report.attempts[index + 1..]
                        .iter()
                        .take_while(|later| later.group_index == attempt.group_index)
                        .any(|later| later.deployment == attempt.deployment);
                    prop_assert!(!repicked, "a skipped deployment was picked again in its group");
                }
                if let Some((_, Decision::Stop)) = attempt.failure {
                    prop_assert_eq!(index + 1, report.attempts.len(), "an attempt ran after a stop");
                }
            }

            let ran = &scripts[..report.attempts.len()];
            match &report.outcome {
                Ok(value) => {
                    prop_assert_eq!(ran.last().map(|s| s.outcome.clone()), Some(Ok(value.clone())));
                    prop_assert!(ran[..ran.len() - 1].iter().all(|s| s.outcome.is_err()));
                }
                Err(CallFailure::MidStream { error, chunks }) => {
                    let last = ran.last().unwrap();
                    prop_assert_eq!(last.outcome.clone(), Err(error.clone()));
                    prop_assert_eq!(last.chunks.len(), *chunks as usize);
                    prop_assert!(*chunks > 0);
                }
                Err(CallFailure::Exhausted { last }) | Err(CallFailure::Budget { last: Some(last) }) => {
                    prop_assert!(ran.iter().all(|s| s.outcome.is_err()));
                    prop_assert_eq!(ran.last().map(|s| s.outcome.clone()), Some(Err(last.clone())));
                    prop_assert!(ran.iter().all(|s| s.chunks.is_empty() || s.outcome.is_ok()));
                }
                Err(CallFailure::Budget { last: None }) => prop_assert!(ran.is_empty()),
                Err(CallFailure::Interrupted { .. }) => prop_assert!(false, "no host failure was scripted"),
                Err(CallFailure::Protocol(violation)) => prop_assert!(false, "{violation}"),
            }
            Ok(())
        })?;
    }
}
