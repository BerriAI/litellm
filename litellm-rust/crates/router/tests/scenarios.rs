//! The Python router's retry and fallback behavior, one case per rule, each stated as the
//! plan, the attempts, the exact trace the host sees and what the report says. A case that
//! reads `route:sleep:0` or `failed:0:Authentication` is asserting the wall-clock and the
//! callback-visible behavior at once.

mod support;

use litellm_callbacks_test::{Script, assert_trace};
use litellm_router::{
    Deployment, DeploymentId, FailureClass, FallbackChains, Fallbacks, PlanSource, Retries,
    RetryPolicy, RoutePlan,
};
use rstest::rstest;
use support::{
    Outcome, Scenario, chains, deployments, err, ok, outcome, plan, python_retries,
    rate_limited_after, run, scenario, with_fallbacks,
};

use FailureClass::{
    Authentication, BadRequest, Connection, ContextWindow, InternalServer, NotFound, RateLimited,
};

/// What a case asserts beyond the trace.
struct Expect {
    visited: &'static [(u64, u32)],
    decisions: &'static [&'static str],
    outcome: Outcome,
}

fn expect(
    visited: &'static [(u64, u32)],
    decisions: &'static [&'static str],
    outcome: Outcome,
) -> Expect {
    Expect {
        visited,
        decisions,
        outcome,
    }
}

fn class_policy(retries: u32, class_retries: &[(FailureClass, u32)]) -> RetryPolicy {
    RetryPolicy {
        class_retries: class_retries.to_vec(),
        ..python_retries(Retries::Default(retries))
    }
}

fn with_retry(plan: RoutePlan, retry: RetryPolicy) -> RoutePlan {
    RoutePlan { retry, ..plan }
}

fn host_fallbacks(plan: RoutePlan) -> RoutePlan {
    RoutePlan {
        fallbacks: Fallbacks::Host,
        ..plan
    }
}

fn not_one(candidates: &[litellm_router::Candidate]) -> Option<DeploymentId> {
    candidates
        .iter()
        .find(|c| c.id != DeploymentId(1))
        .map(|c| c.id)
}

fn ninety_nine(_: &[litellm_router::Candidate]) -> Option<DeploymentId> {
    Some(DeploymentId(99))
}

#[rstest]
#[case::lone_deployment_rate_limit_retries_with_exponential_backoff(
    scenario(plan(&[1], 2), vec![err(RateLimited), err(RateLimited), ok()]),
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:500",
        "emit:attempt_started", "route:send", "failed:1:RateLimited", "route:sleep:1000",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (1, 0), (1, 0)], &["retry", "retry", "ok"], Outcome::Ok)
)]
#[case::lone_deployment_raises_the_last_error_once_retries_run_out(
    scenario(plan(&[1], 1), vec![err(RateLimited), err(InternalServer)]),
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:500",
        "emit:attempt_started", "route:send", "failed:1:InternalServer", "emit:failed",
    ],
    expect(&[(1, 0), (1, 0)], &["retry", "stop"], Outcome::Exhausted)
)]
#[case::rate_limit_with_an_available_sibling_repicks_without_waiting(
    Scenario { cool_on: vec![RateLimited], ..scenario(plan(&[1, 2], 1), vec![err(RateLimited), ok()]) },
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:0",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (2, 0)], &["retry", "ok"], Outcome::Ok)
)]
#[case::rate_limit_with_every_sibling_cooled_falls_back_when_a_chain_exists(
    Scenario {
        cool_on: vec![RateLimited],
        precooled: vec![2],
        ..scenario(with_fallbacks(plan(&[1, 2], 3), chains(&[&[5]])), vec![err(RateLimited), ok()])
    },
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (5, 1)], &["fallback", "ok"], Outcome::Ok)
)]
#[case::rate_limit_with_every_sibling_cooled_and_no_chain_stops(
    Scenario {
        cool_on: vec![RateLimited],
        precooled: vec![2],
        ..scenario(plan(&[1, 2], 3), vec![err(RateLimited)])
    },
    &["emit:attempt_started", "route:send", "failed:0:RateLimited", "emit:failed"],
    expect(&[(1, 0)], &["stop"], Outcome::Exhausted)
)]
#[case::authentication_error_with_siblings_repicks_and_skips_the_failed_deployment(
    Scenario { host_picks: true, ..scenario(plan(&[1, 2], 1), vec![err(Authentication), ok()]) },
    &[
        "route:pick:0:[1,2]", "emit:attempt_started", "route:send", "failed:0:Authentication",
        "route:sleep:0", "route:pick:0:[2]", "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (2, 0)], &["retry-skip", "ok"], Outcome::Ok)
)]
#[case::authentication_error_on_a_lone_deployment_is_not_retried(
    scenario(plan(&[1], 3), vec![err(Authentication)]),
    &["emit:attempt_started", "route:send", "failed:0:Authentication", "emit:failed"],
    expect(&[(1, 0)], &["stop"], Outcome::Exhausted)
)]
#[case::not_found_is_never_retried_even_with_siblings(
    scenario(plan(&[1, 2], 3), vec![err(NotFound)]),
    &["emit:attempt_started", "route:send", "failed:0:NotFound", "emit:failed"],
    expect(&[(1, 0)], &["stop"], Outcome::Exhausted)
)]
#[case::bad_request_skips_the_retries_and_falls_back(
    scenario(with_fallbacks(plan(&[1, 2], 3), chains(&[&[5]])), vec![err(BadRequest), ok()]),
    &[
        "emit:attempt_started", "route:send", "failed:0:BadRequest",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (5, 1)], &["fallback", "ok"], Outcome::Ok)
)]
#[case::connection_errors_retry_without_skipping_the_failed_deployment(
    Scenario { host_picks: true, ..scenario(plan(&[1, 2], 1), vec![err(Connection), ok()]) },
    &[
        "route:pick:0:[1,2]", "emit:attempt_started", "route:send", "failed:0:Connection",
        "route:sleep:0", "route:pick:0:[1,2]", "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (1, 0)], &["retry", "ok"], Outcome::Ok)
)]
#[case::context_window_error_takes_its_own_chain_without_retrying(
    scenario(
        with_fallbacks(plan(&[1], 3), FallbackChains { context_window: vec![deployments(&[9])], ..chains(&[&[5]]) }),
        vec![err(ContextWindow), ok()],
    ),
    &[
        "emit:attempt_started", "route:send", "failed:0:ContextWindow",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (9, 1)], &["fallback", "ok"], Outcome::Ok)
)]
#[case::context_window_chain_does_not_continue_into_the_generic_chain(
    scenario(
        with_fallbacks(plan(&[1], 0), FallbackChains { context_window: vec![deployments(&[9])], ..chains(&[&[5]]) }),
        vec![err(ContextWindow), err(InternalServer)],
    ),
    &[
        "emit:attempt_started", "route:send", "failed:0:ContextWindow",
        "emit:attempt_started", "route:send", "failed:1:InternalServer", "emit:failed",
    ],
    expect(&[(1, 0), (9, 1)], &["fallback", "fallback"], Outcome::Exhausted)
)]
#[case::context_window_error_without_its_chain_uses_the_generic_chain(
    scenario(with_fallbacks(plan(&[1], 3), chains(&[&[5]])), vec![err(ContextWindow), ok()]),
    &[
        "emit:attempt_started", "route:send", "failed:0:ContextWindow",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (5, 1)], &["fallback", "ok"], Outcome::Ok)
)]
#[case::retry_policy_retries_a_bad_request_and_skips_the_failed_deployment(
    Scenario {
        host_picks: true,
        ..scenario(with_retry(plan(&[1, 2], 0), class_policy(0, &[(BadRequest, 1)])), vec![err(BadRequest), ok()])
    },
    &[
        "route:pick:0:[1,2]", "emit:attempt_started", "route:send", "failed:0:BadRequest",
        "route:sleep:0", "route:pick:0:[2]", "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (2, 0)], &["retry-skip", "ok"], Outcome::Ok)
)]
#[case::retry_policy_treats_a_context_window_error_as_a_bad_request(
    scenario(
        with_fallbacks(
            with_retry(plan(&[1], 0), class_policy(0, &[(BadRequest, 1)])),
            FallbackChains { context_window: vec![deployments(&[9])], ..FallbackChains::default() },
        ),
        vec![err(ContextWindow), ok()],
    ),
    &[
        "emit:attempt_started", "route:send", "failed:0:ContextWindow", "route:sleep:500",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (1, 0)], &["retry", "ok"], Outcome::Ok)
)]
#[case::a_request_that_set_zero_retries_switches_the_retry_policy_off(
    scenario(
        with_retry(plan(&[1], 0), RetryPolicy { class_retries: vec![(RateLimited, 3)], ..python_retries(Retries::Request(0)) }),
        vec![err(RateLimited)],
    ),
    &["emit:attempt_started", "route:send", "failed:0:RateLimited", "emit:failed"],
    expect(&[(1, 0)], &["stop"], Outcome::Exhausted)
)]
#[case::a_deployments_own_retries_beat_the_router_default(
    scenario(
        RoutePlan::single(vec![Deployment::new(DeploymentId(1)).with_retries(1)], python_retries(Retries::Default(0))),
        vec![err(RateLimited), ok()],
    ),
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:500",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (1, 0)], &["retry", "ok"], Outcome::Ok)
)]
#[case::a_deployments_own_retries_yield_to_the_request(
    scenario(
        RoutePlan::single(vec![Deployment::new(DeploymentId(1)).with_retries(1)], python_retries(Retries::Request(0))),
        vec![err(RateLimited)],
    ),
    &["emit:attempt_started", "route:send", "failed:0:RateLimited", "emit:failed"],
    expect(&[(1, 0)], &["stop"], Outcome::Exhausted)
)]
#[case::the_retry_budget_restarts_in_each_fallback_group(
    scenario(
        with_fallbacks(plan(&[1], 1), chains(&[&[9]])),
        vec![err(RateLimited), err(RateLimited), err(RateLimited), ok()],
    ),
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:500",
        "emit:attempt_started", "route:send", "failed:1:RateLimited",
        "emit:attempt_started", "route:send", "failed:2:RateLimited", "route:sleep:500",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (1, 0), (9, 1), (9, 1)], &["retry", "fallback", "retry", "ok"], Outcome::Ok)
)]
#[case::retry_after_is_obeyed_on_a_lone_deployment(
    scenario(plan(&[1], 1), vec![rate_limited_after(3), ok()]),
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:3000",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (1, 0)], &["retry", "ok"], Outcome::Ok)
)]
#[case::retry_after_beyond_a_minute_falls_back_to_exponential_backoff(
    scenario(plan(&[1], 1), vec![rate_limited_after(61), ok()]),
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:500",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (1, 0)], &["retry", "ok"], Outcome::Ok)
)]
#[case::retry_after_is_ignored_when_a_sibling_is_available(
    scenario(plan(&[1, 2], 1), vec![rate_limited_after(3), ok()]),
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:0",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (2, 0)], &["retry", "ok"], Outcome::Ok)
)]
#[case::cooled_down_deployments_are_never_picked(
    Scenario { precooled: vec![1], ..scenario(plan(&[1, 2], 0), vec![ok()]) },
    &["emit:attempt_started", "route:send", "emit:succeeded"],
    expect(&[(2, 0)], &["ok"], Outcome::Ok)
)]
#[case::a_host_resolved_plan_is_asked_for_before_anything_else(
    Scenario { host_plan: Some(plan(&[7], 0)), ..scenario(PlanSource::Host, vec![ok()]) },
    &["route:plan", "emit:attempt_started", "route:send", "emit:succeeded"],
    expect(&[(7, 0)], &["ok"], Outcome::Ok)
)]
#[case::a_host_resolved_group_is_asked_for_with_the_class_depth_and_tried_deployments(
    Scenario { groups: vec![Some(deployments(&[9]))], ..scenario(host_fallbacks(plan(&[1], 0)), vec![err(BadRequest), ok()]) },
    &[
        "emit:attempt_started", "route:send", "failed:0:BadRequest", "route:next_group:BadRequest@0:[1]",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(1, 0), (9, 1)], &["fallback", "ok"], Outcome::Ok)
)]
#[case::a_host_without_a_next_group_ends_the_call_with_the_last_error(
    Scenario { groups: vec![None], ..scenario(host_fallbacks(plan(&[1], 0)), vec![err(BadRequest)]) },
    &["emit:attempt_started", "route:send", "failed:0:BadRequest", "route:next_group:BadRequest@0:[1]", "emit:failed"],
    expect(&[(1, 0)], &["fallback"], Outcome::Exhausted)
)]
#[case::a_host_that_picks_nothing_leaves_the_group_as_a_rate_limit(
    Scenario {
        host_picks: true,
        pick: not_one,
        groups: vec![Some(deployments(&[2]))],
        ..scenario(host_fallbacks(plan(&[1], 0)), vec![ok()])
    },
    &[
        "route:pick:0:[1]", "route:next_group:RateLimited@0:[]", "route:pick:1:[2]",
        "emit:attempt_started", "route:send", "emit:succeeded",
    ],
    expect(&[(2, 1)], &["ok"], Outcome::Ok)
)]
#[case::a_pick_outside_the_candidates_is_a_protocol_failure(
    Scenario { host_picks: true, pick: ninety_nine, ..scenario(plan(&[1, 2], 3), vec![ok()]) },
    &["route:pick:0:[1,2]", "emit:failed"],
    expect(&[], &[], Outcome::Protocol)
)]
#[case::a_mid_stream_failure_is_terminal_even_with_retries_and_siblings(
    scenario(
        with_fallbacks(plan(&[1, 2], 3), chains(&[&[5]])),
        vec![Script::stream(&["send"], &["a", "b"], Err(support::failure(InternalServer)))],
    ),
    &["emit:attempt_started", "route:send", "yield:a", "yield:b", "failed:0:InternalServer", "emit:failed"],
    expect(&[(1, 0)], &["stop"], Outcome::MidStream(2))
)]
#[case::a_streaming_success_yields_every_chunk_then_one_terminal(
    scenario(plan(&[1], 0), vec![Script::stream(&["send"], &["a", "b"], Ok("ab"))]),
    &["emit:attempt_started", "route:send", "yield:a", "yield:b", "emit:succeeded"],
    expect(&[(1, 0)], &["ok"], Outcome::Ok)
)]
#[case::the_attempt_budget_stops_before_the_next_attempt_starts(
    scenario(RoutePlan { attempt_budget: Some(2), ..plan(&[1], 10) }, vec![err(RateLimited), err(RateLimited), ok()]),
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:500",
        "emit:attempt_started", "route:send", "failed:1:RateLimited", "route:sleep:1000", "emit:failed",
    ],
    expect(&[(1, 0), (1, 0)], &["retry", "retry"], Outcome::Budget)
)]
#[case::the_timeout_stops_before_the_next_attempt_starts(
    scenario(
        RoutePlan { timeout: Some(std::time::Duration::from_millis(1200)), ..plan(&[1], 5) },
        vec![err(RateLimited), err(RateLimited), ok()],
    ),
    &[
        "emit:attempt_started", "route:send", "failed:0:RateLimited", "route:sleep:500",
        "emit:attempt_started", "route:send", "failed:1:RateLimited", "route:sleep:1000", "emit:failed",
    ],
    expect(&[(1, 0), (1, 0)], &["retry", "retry"], Outcome::Budget)
)]
#[case::fallback_depth_is_capped(
    scenario(
        with_fallbacks(plan(&[1], 0), FallbackChains { max_depth: 1, ..chains(&[&[5], &[6]]) }),
        vec![err(BadRequest), err(BadRequest)],
    ),
    &[
        "emit:attempt_started", "route:send", "failed:0:BadRequest",
        "emit:attempt_started", "route:send", "failed:1:BadRequest", "emit:failed",
    ],
    expect(&[(1, 0), (5, 1)], &["fallback", "fallback"], Outcome::Exhausted)
)]
#[case::an_empty_primary_group_reports_that_nothing_ran(
    scenario(plan(&[], 3), vec![]),
    &["emit:failed"],
    expect(&[], &[], Outcome::Budget)
)]
#[tokio::test]
async fn python_router_behavior(
    #[case] scenario: Scenario,
    #[case] trace: &[&str],
    #[case] expect: Expect,
) {
    let run = run(scenario).await;
    assert_trace!(run.trace, trace);
    run.trace.assert_one_terminal();
    assert_eq!(run.visited(), expect.visited);
    assert_eq!(run.decisions(), expect.decisions);
    assert_eq!(outcome(&run.report), expect.outcome);
    assert_eq!(run.report.attempts.len(), run.started.len());
    assert_eq!(
        run.report.selected,
        run.started.last().map(|context| context.deployment)
    );
    assert!(
        run.started
            .iter()
            .all(|c| c.trace_id == run.report.trace_id)
    );
}

#[tokio::test]
async fn backoff_is_floored_and_capped_across_a_run() {
    let policy = RetryPolicy {
        min_backoff: std::time::Duration::from_secs(2),
        ..python_retries(Retries::Default(6))
    };
    let attempts = std::iter::repeat_with(|| err(RateLimited))
        .take(6)
        .chain([ok()])
        .collect();
    let run = run(scenario(with_retry(plan(&[1], 6), policy), attempts)).await;
    let sleeps: Vec<&String> = run
        .trace
        .lines()
        .iter()
        .filter(|line| line.starts_with("route:sleep:"))
        .cloned()
        .collect::<Vec<_>>()
        .leak()
        .iter()
        .collect();
    assert_eq!(
        sleeps,
        [
            "route:sleep:2000",
            "route:sleep:2000",
            "route:sleep:2000",
            "route:sleep:4000",
            "route:sleep:8000",
            "route:sleep:8000"
        ]
    );
    assert_eq!(run.elapsed, std::time::Duration::from_secs(26));
    assert_eq!(outcome(&run.report), Outcome::Ok);
}
