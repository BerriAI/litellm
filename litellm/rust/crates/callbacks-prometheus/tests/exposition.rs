//! The renderer against `prometheus_client.generate_latest` for the same families and the same
//! operations. `exposition/python.txt` is what Python wrote for `exposition/ops.json` over
//! `families.json`, with `_created` samples turned off.

use litellm_callbacks_prometheus::{
    ManualClock, Op, SeriesLimits,
    exposition::{Scrape, created_series, render},
    families::default_store,
};
use rstest::rstest;
use serde::Deserialize;

#[derive(Deserialize)]
struct Recorded {
    family: String,
    values: Vec<String>,
    op: String,
    amount: f64,
}

fn store() -> litellm_callbacks_prometheus::Store {
    let store = default_store(SeriesLimits::default(), Box::new(ManualClock::default()))
        .unwrap_or_else(|error| panic!("{error}"));
    let ops: Vec<Recorded> = serde_json::from_str(include_str!("exposition/ops.json"))
        .unwrap_or_else(|error| panic!("{error}"));
    for recorded in ops {
        let id = store
            .family(&recorded.family)
            .unwrap_or_else(|| panic!("{} is declared", recorded.family));
        let op = match recorded.op.as_str() {
            "inc" => Op::Inc(recorded.amount),
            "set" => Op::Set(recorded.amount),
            "observe" => Op::Observe(recorded.amount),
            other => panic!("{other} is no op"),
        };
        store
            .record(id, &recorded.values, op)
            .unwrap_or_else(|error| panic!("{error}"));
    }
    store
}

#[rstest]
fn a_scrape_is_written_as_python_writes_it() {
    let text = render(
        &store(),
        &Scrape {
            names: Vec::new(),
            created: false,
        },
    );
    assert_eq!(text, include_str!("exposition/python.txt"));
}

#[rstest]
fn a_restricted_scrape_keeps_only_the_samples_asked_for() {
    let text = render(
        &store(),
        &Scrape {
            names: vec![
                "litellm_total_users".to_owned(),
                "litellm_unknown".to_owned(),
            ],
            created: false,
        },
    );
    assert_eq!(
        text,
        "# HELP litellm_total_users Total number of users in LiteLLM\n# TYPE litellm_total_users gauge\nlitellm_total_users 1.2345678e+07\n"
    );
}

#[rstest]
fn created_samples_follow_a_counter_as_their_own_gauge() {
    let text = render(
        &store(),
        &Scrape {
            names: vec!["litellm_requests_metric_created".to_owned()],
            created: true,
        },
    );
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(
        lines[..3],
        [
            "# HELP litellm_requests_metric_total deprecated - use litellm_proxy_total_requests_metric. Total number of LLM calls to litellm - track total per API Key, team, user",
            "# TYPE litellm_requests_metric_total counter",
            "# HELP litellm_requests_metric_created deprecated - use litellm_proxy_total_requests_metric. Total number of LLM calls to litellm - track total per API Key, team, user",
        ]
    );
    assert_eq!(lines.len(), 5);
}

#[rstest]
#[case::unset(None, true)]
#[case::capital_true(Some("True"), false)]
#[case::lower_true(Some("true"), false)]
#[case::one(Some("1"), false)]
#[case::anything_else(Some("yes"), true)]
fn the_created_switch_reads_as_prometheus_client_reads_it(
    #[case] variable: Option<&str>,
    #[case] created: bool,
) {
    assert_eq!(created_series(variable), created);
}
