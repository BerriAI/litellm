mod attempt;
mod callback;
mod emit;
mod failure;
pub mod labels;
pub mod ops;
pub mod pyfmt;
pub mod store;
mod success;
mod tracker;

use std::cell::RefCell;

use litellm_callbacks::CallbackError;
use serde_json::Value;

pub use crate::{
    attempt::{AttemptFacts, Residual},
    callback::{Attempt, Event, LogSink, Prometheus, Stop, StopKind},
    store::{Clock, FamilySpec, Kind, ManualClock, Op, Store, StoreError, SystemClock, Target},
    success::{Level, LogLine, exception_text},
    tracker::SeriesLimits,
};
use crate::{
    emit::Emitter,
    failure::Failure,
    success::{Success, SuccessStep},
};

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    #[strum(serialize = "success_head")]
    SuccessHead,
    #[strum(serialize = "success_limits")]
    SuccessLimits,
    #[strum(serialize = "success_latency")]
    SuccessLatency,
    #[strum(serialize = "success_tail")]
    SuccessTail,
    #[strum(serialize = "failure")]
    Failure,
}

pub struct Recorded {
    pub logs: Vec<LogLine>,
    pub outcome: Result<(), CallbackError>,
}

pub fn record(
    store: &Store,
    facts: &AttemptFacts,
    step: Step,
    input_sequence_length: Option<String>,
) -> Recorded {
    let logs = RefCell::new(Vec::new());
    let emit = Emitter {
        store,
        settings: facts.settings(),
    };
    let outcome = match step {
        Step::Failure => Failure {
            emit,
            facts,
            logs: &logs,
        }
        .run(),
        Step::SuccessHead => success(emit, facts, None, &logs).run(SuccessStep::Head),
        Step::SuccessLimits => success(emit, facts, None, &logs).run(SuccessStep::Limits),
        Step::SuccessLatency => {
            success(emit, facts, input_sequence_length, &logs).run(SuccessStep::Latency)
        }
        Step::SuccessTail => success(emit, facts, None, &logs).run(SuccessStep::Tail),
    };
    Recorded {
        logs: logs.into_inner(),
        outcome,
    }
}

fn success<'a>(
    emit: Emitter<'a>,
    facts: &'a AttemptFacts,
    input_sequence_length: Option<String>,
    logs: &'a RefCell<Vec<LogLine>>,
) -> Success<'a> {
    Success {
        emit,
        facts,
        input_sequence_length: input_sequence_length.map_or(Value::Null, Value::from),
        logs,
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::json;
    use strum::VariantArray;

    use super::*;

    const COUNTERS: [&str; 5] = [
        "litellm_requests_metric",
        "litellm_spend_metric",
        "litellm_total_tokens_metric",
        "litellm_input_tokens_metric",
        "litellm_output_tokens_metric",
    ];

    #[fixture]
    fn store() -> Store {
        let store = Store::new(SeriesLimits::default(), Box::new(ManualClock::default()));
        for name in COUNTERS {
            store
                .register(FamilySpec {
                    declared: name.to_owned(),
                    documentation: "doc".to_owned(),
                    kind: Kind::Counter,
                    labelnames: vec!["team".to_owned()],
                    excluded: Vec::new(),
                    buckets: Vec::new(),
                    most_recent: false,
                    wrapped: false,
                    shares_overflow: true,
                })
                .unwrap_or_else(|error| panic!("{error}"));
        }
        store
    }

    fn decoded<T: serde::de::DeserializeOwned>(value: Value) -> T {
        serde_json::from_value(value).unwrap_or_else(|error| panic!("{error}"))
    }

    fn facts(completion_tokens: Value) -> AttemptFacts {
        AttemptFacts {
            identity: Some(decoded(json!({
                "metadata_kind": "mapping",
                "user_id": "u",
                "key_hash": "hashed",
                "team_id": "team-1",
            }))),
            usage: decoded(json!({
                "completion_tokens": completion_tokens,
                "total_tokens": 30,
                "prompt_tokens": 20,
            })),
            cost: decoded(json!({"response_cost": 0.5})),
            tags: Some(Vec::new()),
            residual: decoded(json!({
                "settings": {},
                "payload_labels": {
                    "model_group": "group",
                    "model_id": "id",
                    "api_base": "base",
                    "custom_llm_provider": "openai",
                },
            })),
            ..AttemptFacts::default()
        }
    }

    fn totals(store: &Store) -> Vec<(&'static str, Vec<f64>)> {
        COUNTERS
            .into_iter()
            .map(|name| {
                let id = store
                    .family(name)
                    .unwrap_or_else(|| panic!("{name} is registered"));
                let values = store
                    .snapshot(id)
                    .series
                    .into_iter()
                    .map(|(_, series)| series.value)
                    .collect();
                (name, values)
            })
            .collect()
    }

    #[rstest]
    fn steps_parse_from_the_names_python_sends(#[values(0, 1, 2, 3, 4)] index: usize) {
        let step = Step::VARIANTS[index];
        let name: &'static str = step.into();
        assert_eq!(name.parse::<Step>(), Ok(step));
    }

    #[rstest]
    fn the_head_step_stops_where_python_raises(store: Store) {
        let recorded = record(&store, &facts(json!("ten")), Step::SuccessHead, None);
        assert_eq!(
            recorded.outcome.map_err(|error| exception_text(&error)),
            Err("could not convert string to float: 'ten'".to_owned())
        );
        assert_eq!(
            totals(&store),
            vec![
                ("litellm_requests_metric", vec![1.0]),
                ("litellm_spend_metric", vec![0.5]),
                ("litellm_total_tokens_metric", vec![30.0]),
                ("litellm_input_tokens_metric", vec![20.0]),
                ("litellm_output_tokens_metric", vec![]),
            ]
        );
    }

    #[rstest]
    fn only_the_head_step_counts_requests_and_tokens(
        store: Store,
        #[values(Step::SuccessLimits, Step::SuccessLatency, Step::SuccessTail)] step: Step,
    ) {
        let recorded = record(&store, &facts(json!(10)), step, None);
        assert!(recorded.outcome.is_ok());
        assert!(
            totals(&store)
                .into_iter()
                .all(|(_, values)| values.is_empty())
        );
    }

    #[rstest]
    #[case::without_reason(json!({"reason_present": false}), Err("'reason'".to_owned()), vec![])]
    #[case::with_reason(json!({"reason": "failed_call", "reason_present": true}), Ok(()), vec![(vec!["team-1".to_owned(), "failed_call".to_owned()], 1.0)])]
    fn the_failure_step_counts_the_failure_before_the_zero_cost_reason(
        #[case] diagnostic: Value,
        #[case] expected: Result<(), String>,
        #[case] zero_cost: Vec<(Vec<String>, f64)>,
    ) {
        let store = Store::new(SeriesLimits::default(), Box::new(ManualClock::default()));
        for (name, labelnames) in [
            ("litellm_llm_api_failed_requests_metric", vec!["team"]),
            ("litellm_zero_cost_requests_total", vec!["team", "reason"]),
        ] {
            store
                .register(FamilySpec {
                    declared: name.to_owned(),
                    documentation: "doc".to_owned(),
                    kind: Kind::Counter,
                    labelnames: labelnames.into_iter().map(str::to_owned).collect(),
                    excluded: Vec::new(),
                    buckets: Vec::new(),
                    most_recent: false,
                    wrapped: false,
                    shares_overflow: true,
                })
                .unwrap_or_else(|error| panic!("{error}"));
        }
        let mut facts = facts(json!(10));
        facts.cost.zero_cost = Some(decoded(diagnostic));
        let recorded = record(&store, &facts, Step::Failure, None);
        assert_eq!(
            recorded.outcome.map_err(|error| exception_text(&error)),
            expected
        );
        let series = |name: &str| -> Vec<(Vec<String>, f64)> {
            let id = store
                .family(name)
                .unwrap_or_else(|| panic!("{name} is registered"));
            store
                .snapshot(id)
                .series
                .into_iter()
                .map(|(values, series)| (values, series.value))
                .collect()
        };
        assert_eq!(
            series("litellm_llm_api_failed_requests_metric"),
            vec![(vec!["team-1".to_owned()], 1.0)]
        );
        assert_eq!(series("litellm_zero_cost_requests_total"), zero_cost);
    }
}
