use std::sync::Arc;

use litellm_callbacks::{CallbackError, Field, Outcome};
use serde::{Deserialize, Serialize};

use crate::{
    LogLine, Step,
    attempt::AttemptFacts,
    emit::named_error,
    ops::{ErrorKind, OpError},
    record,
    store::Store,
};

pub trait LogSink: Send + Sync {
    fn log(&self, line: &LogLine);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    Success,
    Failure,
}

impl Event {
    /// The event an attempt's outcome logs, `None` for a fallback's, which Python's logger never
    /// hears.
    pub const fn of(outcome: Outcome) -> Option<Self> {
        match outcome {
            Outcome::Success => Some(Self::Success),
            Outcome::Failure => Some(Self::Failure),
            Outcome::FallbackSuccess | Outcome::FallbackFailure => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attempt {
    pub event: Event,
    pub facts: AttemptFacts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopKind {
    Value,
    Key,
    Type,
    Runtime,
    Named,
}

impl From<ErrorKind> for StopKind {
    fn from(kind: ErrorKind) -> Self {
        match kind {
            ErrorKind::Value => Self::Value,
            ErrorKind::Key => Self::Key,
            ErrorKind::Type => Self::Type,
            ErrorKind::Runtime => Self::Runtime,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stop {
    pub step: Step,
    pub kind: StopKind,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
}

impl Stop {
    fn raised(step: Step, error: CallbackError) -> Self {
        if let CallbackError::Failed(message) = &error
            && let Some((name, text)) = named_error(message)
            && !matches!(name, "KeyError" | "TypeError" | "ValueError")
        {
            return Self {
                step,
                kind: StopKind::Named,
                message: text.to_owned(),
                error_type: Some(name.to_owned()),
            };
        }
        let error = OpError::from(error);
        Self {
            step,
            kind: error.kind.into(),
            message: error.message,
            error_type: None,
        }
    }
}

const SUCCESS_STEPS: [Step; 4] = [
    Step::SuccessHead,
    Step::SuccessLimits,
    Step::SuccessLatency,
    Step::SuccessTail,
];
const FAILURE_STEPS: [Step; 1] = [Step::Failure];
/// The contract fields an attempt is read from.
pub const READS: [Field; 15] = [
    Field::Model,
    Field::Outcome,
    Field::LoggedKeyIdentity,
    Field::Deployment,
    Field::Usage,
    Field::Cost,
    Field::RequestTags,
    Field::GuardrailInfo,
    Field::AttemptTiming,
    Field::RateLimitState,
    Field::LoggedRequest,
    Field::McpToolCall,
    Field::LoggedModelGroup,
    Field::MetadataLabels,
    Field::InputSequenceLength,
];

pub struct Prometheus {
    store: Arc<Store>,
    sink: Arc<dyn LogSink>,
    emit_input_sequence_length_label: bool,
}

impl Prometheus {
    pub fn new(
        store: Arc<Store>,
        sink: Arc<dyn LogSink>,
        emit_input_sequence_length_label: bool,
    ) -> Self {
        Self {
            store,
            sink,
            emit_input_sequence_length_label,
        }
    }

    pub fn run(&self, attempt: &Attempt) -> Result<(), Stop> {
        let steps: &[Step] = match attempt.event {
            Event::Success => &SUCCESS_STEPS,
            Event::Failure => &FAILURE_STEPS,
        };
        let facts = &attempt.facts;
        for &step in steps {
            let input_sequence_length =
                if step == Step::SuccessLatency && self.emit_input_sequence_length_label {
                    facts
                        .input_sequence_length()
                        .map_err(|error| Stop::raised(step, error))?
                } else {
                    None
                };
            let recorded = record(&self.store, &attempt.facts, step, input_sequence_length);
            for line in &recorded.logs {
                self.sink.log(line);
            }
            recorded
                .outcome
                .map_err(|error| Stop::raised(step, error))?;
        }
        Ok(())
    }

    /// The store the attempts are recorded into.
    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }
}

/// How a stop is told as a callback error, which the host logs as the logging line does.
impl From<Stop> for CallbackError {
    fn from(stop: Stop) -> Self {
        Self::Failed(serde_json::to_string(&stop).unwrap_or(stop.message))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use litellm_callbacks::{FieldSet, FieldValues, Hook, HookInput};
    use rstest::{fixture, rstest};
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::{FamilySpec, HostFacts, Kind, ManualClock, SeriesLimits};

    const FAMILIES: [(&str, Kind); 7] = [
        ("litellm_requests_metric", Kind::Counter),
        ("litellm_spend_metric", Kind::Counter),
        ("litellm_total_tokens_metric", Kind::Counter),
        ("litellm_input_tokens_metric", Kind::Counter),
        ("litellm_output_tokens_metric", Kind::Counter),
        ("litellm_llm_api_failed_requests_metric", Kind::Counter),
        ("litellm_proxy_total_requests_metric", Kind::Counter),
    ];

    #[derive(Default)]
    struct Lines(Mutex<Vec<LogLine>>);

    impl LogSink for Lines {
        fn log(&self, line: &LogLine) {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(line.clone());
        }
    }

    #[fixture]
    fn prometheus() -> Prometheus {
        built(false)
    }

    fn built(emit_input_sequence_length_label: bool) -> Prometheus {
        let store = Store::new(SeriesLimits::default(), Box::new(ManualClock::default()));
        for (name, kind) in FAMILIES {
            store
                .register(FamilySpec {
                    declared: name.to_owned(),
                    documentation: "doc".to_owned(),
                    kind,
                    labelnames: vec!["team".to_owned()],
                    excluded: Vec::new(),
                    buckets: Vec::new(),
                    most_recent: false,
                    wrapped: false,
                    shares_overflow: true,
                })
                .unwrap_or_else(|error| panic!("{error}"));
        }
        Prometheus::new(
            Arc::new(store),
            Arc::new(Lines::default()),
            emit_input_sequence_length_label,
        )
    }

    fn attempt(event: &str, completion_tokens: Value, failed: bool) -> Vec<(Field, Value)> {
        let mut values = vec![
            (Field::Outcome, json!(event)),
            (
                Field::LoggedKeyIdentity,
                json!({
                    "metadata_kind": "mapping",
                    "user_id": "u",
                    "key_hash": "hashed",
                    "team_id": "team-1",
                }),
            ),
            (
                Field::Usage,
                json!({"logged": {
                    "completion_tokens": completion_tokens,
                    "total_tokens": 30,
                    "prompt_tokens": 20,
                }}),
            ),
            (Field::Cost, json!({"response_cost": 0.5})),
            (Field::RequestTags, json!([])),
        ];
        if failed {
            values.push((
                Field::InputSequenceLength,
                json!({
                    "error_type": "AttributeError",
                    "error_text": "'NoneType' object has no attribute 'get'",
                }),
            ));
        }
        values
    }

    fn host() -> HostFacts {
        let labels = json!({
            "model_group": "group",
            "model_id": "id",
            "api_base": "base",
            "custom_llm_provider": "openai",
        });
        HostFacts {
            deployment_labels: serde_json::from_value::<Map<String, Value>>(labels)
                .unwrap_or_else(|error| panic!("{error}")),
            ..HostFacts::default()
        }
    }

    fn input(hook: Hook, values: Vec<(Field, Value)>) -> HookInput {
        let mut fields = FieldValues::new(FieldSet::of(&READS));
        for (field, value) in values {
            fields.insert(field, value);
        }
        HookInput {
            hook,
            fields,
            call: None,
        }
    }

    /// `None` when the input ends no attempt the logger records.
    fn handled(prometheus: &Prometheus, values: Vec<(Field, Value)>) -> Option<Result<(), Stop>> {
        let (outcome, facts) = AttemptFacts::read(&input(Hook::AttemptEnded, values), host())
            .unwrap_or_else(|error| panic!("{error}"))?;
        let event = Event::of(outcome)?;
        Some(prometheus.run(&Attempt { event, facts }))
    }

    fn total(prometheus: &Prometheus, name: &str) -> Vec<f64> {
        let id = prometheus
            .store
            .family(name)
            .unwrap_or_else(|| panic!("{name} is registered"));
        prometheus
            .store
            .snapshot(id)
            .series
            .into_iter()
            .map(|(_, series)| series.value)
            .collect()
    }

    #[rstest]
    fn a_success_runs_every_step(prometheus: Prometheus) {
        let output = handled(&prometheus, attempt("success", json!(10), false));
        assert_eq!(output, Some(Ok(())));
        assert_eq!(total(&prometheus, "litellm_requests_metric"), vec![1.0]);
        assert_eq!(
            total(&prometheus, "litellm_output_tokens_metric"),
            vec![10.0]
        );
        assert_eq!(
            total(&prometheus, "litellm_proxy_total_requests_metric"),
            vec![1.0]
        );
    }

    #[rstest]
    fn a_failure_runs_only_the_failure_step(prometheus: Prometheus) {
        let output = handled(&prometheus, attempt("failure", json!(10), false));
        assert_eq!(output, Some(Ok(())));
        assert_eq!(
            total(&prometheus, "litellm_llm_api_failed_requests_metric"),
            vec![1.0]
        );
        assert!(total(&prometheus, "litellm_requests_metric").is_empty());
    }

    #[rstest]
    fn a_raise_reports_its_step_and_python_class(prometheus: Prometheus) {
        let output = handled(&prometheus, attempt("success", json!("ten"), false));
        assert_eq!(
            output,
            Some(Err(Stop {
                step: Step::SuccessHead,
                kind: StopKind::Value,
                message: "could not convert string to float: 'ten'".to_owned(),
                error_type: None,
            }))
        );
        assert!(total(&prometheus, "litellm_proxy_total_requests_metric").is_empty());
    }

    #[rstest]
    fn the_input_sequence_error_stops_before_latency_with_its_python_class() {
        let prometheus = built(true);
        let output = handled(&prometheus, attempt("success", json!(10), true));
        assert_eq!(
            output,
            Some(Err(Stop {
                step: Step::SuccessLatency,
                kind: StopKind::Named,
                message: "'NoneType' object has no attribute 'get'".to_owned(),
                error_type: Some("AttributeError".to_owned()),
            }))
        );
        assert_eq!(total(&prometheus, "litellm_requests_metric"), vec![1.0]);
        assert!(total(&prometheus, "litellm_proxy_total_requests_metric").is_empty());
    }

    #[rstest]
    #[case::no_value(Vec::new())]
    #[case::fallback_outcome(attempt("fallback_success", json!(10), false))]
    fn nothing_else_records(prometheus: Prometheus, #[case] values: Vec<(Field, Value)>) {
        assert_eq!(handled(&prometheus, values), None);
        assert!(
            FAMILIES
                .iter()
                .all(|(name, _)| total(&prometheus, name).is_empty())
        );
    }

    #[rstest]
    fn a_malformed_field_is_an_error() {
        let read = AttemptFacts::read(
            &input(
                Hook::AttemptEnded,
                vec![
                    (Field::Outcome, json!("success")),
                    (Field::Cost, json!({"event": "nope"})),
                ],
            ),
            host(),
        );
        assert!(
            matches!(read, Err(CallbackError::Failed(text)) if text.starts_with("prometheus attempt: "))
        );
    }
}
