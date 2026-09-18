//! Behavior we have not confirmed against Python yet. Every case states the expectation
//! we would write if the answer were what we suspect, and stays ignored until someone
//! reads or runs the Python and either promotes it into `scenarios.rs` or corrects it.
//! CI runs this binary with `--ignored` in a report-only step, so the list is compiled.

mod support;

use litellm_callbacks_test::{Script, assert_trace};
use litellm_router::{FailureClass, FallbackChains, RoutePlan};
use support::{Outcome, chains, err, ok, outcome, plan, run, scenario, with_fallbacks};

#[ignore = "python parity open: run_async_fallback skips a target already attempted for this request; do local chains dedupe by deployment too?"]
#[tokio::test]
async fn a_local_chain_does_not_repeat_a_deployment_already_tried() {
    let run = run(scenario(
        with_fallbacks(plan(&[1], 0), chains(&[&[1], &[5]])),
        vec![err(FailureClass::BadRequest), ok()],
    ))
    .await;
    assert_eq!(run.visited(), [(1, 0), (5, 2)]);
    assert_eq!(outcome(&run.report), Outcome::Ok);
}

#[ignore = "python parity open: when every deployment in a group is skipped after 401s, does the router burn its remaining retries on no-deployment errors before falling back?"]
#[tokio::test]
async fn a_group_whose_deployments_are_all_skipped_falls_back_at_once() {
    let run = run(support::Scenario {
        host_picks: true,
        ..scenario(
            with_fallbacks(plan(&[1, 2], 5), chains(&[&[5]])),
            vec![
                err(FailureClass::Authentication),
                err(FailureClass::Authentication),
                ok(),
            ],
        )
    })
    .await;
    assert_trace!(
        run.trace,
        [
            "route:pick:0:[1,2]",
            "emit:attempt_started",
            "route:send",
            "failed:0:Authentication",
            "route:sleep:0",
            "route:pick:0:[2]",
            "emit:attempt_started",
            "route:send",
            "failed:1:Authentication",
            "route:pick:1:[5]",
            "emit:attempt_started",
            "route:send",
            "emit:succeeded"
        ]
    );
}

#[ignore = "python parity open: _acompletion_streaming_iterator continues on a fallback stream mid-stream and merges usage; the router stops instead"]
#[tokio::test]
async fn a_mid_stream_failure_continues_on_the_next_group() {
    let run = run(scenario(
        with_fallbacks(
            plan(&[1], 0),
            FallbackChains::generic(vec![support::deployments(&[5])]),
        ),
        vec![
            Script::stream(
                &["send"],
                &["a"],
                Err(support::failure(FailureClass::InternalServer)),
            ),
            Script::stream(&["send"], &["b"], Ok("ab")),
        ],
    ))
    .await;
    assert_eq!(run.visited(), [(1, 0), (5, 1)]);
    assert_eq!(outcome(&run.report), Outcome::Ok);
    let _ = RoutePlan::single;
}
