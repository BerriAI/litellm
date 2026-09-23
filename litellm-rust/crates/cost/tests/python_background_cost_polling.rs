// mirrors: test_litellm/interactions/test_background_cost_polling.py::test_poll_intervals_double_up_to_the_cap_and_stay_inside_the_timeout

use litellm_cost::background_cost_polling::{
    is_pollable_background_interaction, missing_usage_is_expected, poll_intervals,
};
use rstest::rstest;

#[rstest]
#[case("queued", "interaction-1", true, true)]
#[case("in_progress", "interaction-1", true, true)]
#[case("queued", "", false, true)]
#[case("completed", "interaction-1", false, false)]
#[case("requires_action", "interaction-1", false, false)]
#[case("failed", "interaction-1", false, true)]
#[case("cancelled", "interaction-1", false, true)]
#[case("incomplete", "interaction-1", false, true)]
#[case("budget_exceeded", "interaction-1", false, true)]
fn background_status_selects_polling_and_missing_usage_expectation(
    #[case] status: &str,
    #[case] interaction_id: &str,
    #[case] pollable: bool,
    #[case] missing_usage_expected: bool,
) {
    assert_eq!(
        is_pollable_background_interaction(status, interaction_id),
        pollable
    );
    assert_eq!(missing_usage_is_expected(status), missing_usage_expected);
}

#[rstest]
#[case(0.0, 0.002)]
#[case(0.001, 0.0)]
#[case(-1.0, 0.002)]
#[case(0.0, 0.0)]
fn polling_stops_with_non_positive_intervals(#[case] initial: f64, #[case] maximum: f64) {
    let intervals = poll_intervals(initial, maximum, 3600.0)
        .take(10)
        .collect::<Vec<_>>();
    assert!(intervals.len() < 10);
    assert!(intervals.iter().all(|interval| *interval > 0.0));
}

#[rstest]
fn polling_doubles_to_cap_without_exceeding_timeout() {
    let intervals = poll_intervals(5.0, 60.0, 3600.0).collect::<Vec<_>>();
    assert_eq!(&intervals[..6], &[5.0, 10.0, 20.0, 40.0, 60.0, 60.0]);
    assert!(intervals.iter().all(|interval| *interval <= 60.0));
    let elapsed = intervals.iter().sum::<f64>();
    assert!(elapsed <= 3600.0);
    assert!(elapsed + 60.0 > 3600.0);
}

#[rstest]
#[case(5.0, 60.0, 4.0, Vec::<f64>::new())]
#[case(5.0, 60.0, 15.0, vec![5.0, 10.0])]
#[case(5.0, 60.0, 14.0, vec![5.0])]
fn polling_stops_before_an_interval_that_crosses_timeout(
    #[case] initial: f64,
    #[case] maximum: f64,
    #[case] timeout: f64,
    #[case] expected: Vec<f64>,
) {
    assert_eq!(
        poll_intervals(initial, maximum, timeout).collect::<Vec<_>>(),
        expected
    );
}
