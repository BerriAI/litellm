use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput, Manifest,
    Outcome, Selection, Subscription,
};
use serde::{Deserialize, Serialize};

use crate::{
    LogLine, Step,
    attempt::AttemptFacts,
    emit::named_error,
    ops::{ErrorKind, OpError},
    record, router_remaining,
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
const READS: [Field; 17] = [
    Field::DeprecatedKwargs,
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
    Field::RouterRef,
];

pub struct Prometheus {
    store: Arc<Store>,
    sink: Arc<dyn LogSink>,
    manifest: Manifest,
}

impl Prometheus {
    pub fn new(store: Arc<Store>, sink: Arc<dyn LogSink>) -> Self {
        Self {
            store,
            sink,
            manifest: Self::declared(),
        }
    }

    pub fn declared() -> Manifest {
        Manifest {
            selection: Selection {
                name: "prometheus".to_owned(),
                aliases: Vec::new(),
                activation: Activation::Required,
                opt_out_by: Vec::new(),
                honors_no_log: false,
            },
            subscriptions: vec![Subscription {
                reads: FieldSet::of(&READS),
                awaited: true,
                ..Subscription::on(Hook::AttemptEnded)
            }],
            has_request_state: false,
        }
    }

    pub fn run(&self, attempt: &Attempt) -> Result<(), Stop> {
        let steps: &[Step] = match attempt.event {
            Event::Success => &SUCCESS_STEPS,
            Event::Failure => &FAILURE_STEPS,
        };
        let facts = &attempt.facts;
        for &step in steps {
            let input_sequence_length = if step == Step::SuccessLatency
                && facts.residual.emit_input_sequence_length_label
            {
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

    async fn attempt_ended(&self, input: &HookInput) -> HookOutput {
        let Some((outcome, facts)) = AttemptFacts::read(input)? else {
            return Ok(None);
        };
        let event = match outcome {
            Outcome::Success => Event::Success,
            Outcome::Failure => Event::Failure,
            Outcome::FallbackSuccess | Outcome::FallbackFailure => return Ok(None),
        };
        let attempt = Attempt { event, facts };
        let ran = self.run(&attempt);
        let tail_ran = ran
            .as_ref()
            .err()
            .is_none_or(|stop| stop.step == Step::SuccessTail);
        if event == Event::Success && tail_ran {
            let router = input.fields.router()?;
            for line in router_remaining(&self.store, &attempt.facts, router).await {
                self.sink.log(&line);
            }
        }
        ran.map_err(|stop| {
            CallbackError::Failed(serde_json::to_string(&stop).unwrap_or(stop.message))
        })?;
        Ok(None)
    }
}

impl Callback for Prometheus {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a Self::State>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match input.hook {
                Hook::AttemptEnded => self.attempt_ended(input).await,
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures_util::FutureExt;
    use litellm_callbacks::{FieldValues, RemainingUsage, RouterRef, RouterUsage, UsageAmount};
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;
    use crate::{FamilySpec, Kind, ManualClock, SeriesLimits};

    const FAMILIES: [(&str, Kind); 9] = [
        ("litellm_requests_metric", Kind::Counter),
        ("litellm_spend_metric", Kind::Counter),
        ("litellm_total_tokens_metric", Kind::Counter),
        ("litellm_input_tokens_metric", Kind::Counter),
        ("litellm_output_tokens_metric", Kind::Counter),
        ("litellm_llm_api_failed_requests_metric", Kind::Counter),
        ("litellm_proxy_total_requests_metric", Kind::Counter),
        ("litellm_remaining_tokens_metric", Kind::Gauge),
        ("litellm_remaining_requests_metric", Kind::Gauge),
    ];

    #[derive(Default)]
    struct Lines(Mutex<Vec<LogLine>>);

    struct Router {
        answer: Result<Vec<(&'static str, UsageAmount)>, String>,
        asked: Mutex<Vec<String>>,
    }

    impl Router {
        fn answering(answer: Result<Vec<(&'static str, UsageAmount)>, String>) -> Arc<Self> {
            Arc::new(Self {
                answer,
                asked: Mutex::default(),
            })
        }

        fn asked(&self) -> Vec<String> {
            self.asked
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl RouterUsage for Router {
        fn remaining(
            &self,
            model_group: &str,
        ) -> BoxFuture<'static, Result<RemainingUsage, String>> {
            self.asked
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(model_group.to_owned());
            let answer = self.answer.clone().map(|headers| RemainingUsage {
                headers: headers
                    .into_iter()
                    .map(|(name, amount)| (name.to_owned(), amount))
                    .collect(),
            });
            Box::pin(async move { answer })
        }
    }

    fn limits() -> Vec<(&'static str, UsageAmount)> {
        vec![
            ("x-ratelimit-remaining-tokens", UsageAmount::Int(75)),
            ("x-ratelimit-limit-tokens", UsageAmount::Int(100)),
            ("x-ratelimit-remaining-requests", UsageAmount::Int(9)),
            ("x-ratelimit-limit-requests", UsageAmount::Int(10)),
        ]
    }

    fn with_provider_headers(headers: Value) -> Vec<(Field, Value)> {
        let mut values = attempt("success", json!(10), false);
        values.push((Field::RateLimitState, headers));
        values
    }

    impl LogSink for Lines {
        fn log(&self, line: &LogLine) {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(line.clone());
        }
    }

    fn logging() -> (Prometheus, Arc<Lines>) {
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
        let lines = Arc::new(Lines::default());
        let sink: Arc<dyn LogSink> = Arc::clone(&lines) as Arc<dyn LogSink>;
        (Prometheus::new(Arc::new(store), sink), lines)
    }

    #[fixture]
    fn prometheus() -> Prometheus {
        logging().0
    }

    impl Lines {
        fn taken(&self) -> Vec<LogLine> {
            std::mem::take(
                &mut *self
                    .0
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
        }
    }

    fn attempt(event: &str, completion_tokens: Value, failed: bool) -> Vec<(Field, Value)> {
        let mut values = vec![
            (Field::Outcome, json!(event)),
            (
                Field::DeprecatedKwargs,
                json!({
                    "settings": {},
                    "emit_input_sequence_length_label": failed,
                    "payload_labels": {
                        "model_group": "group",
                        "model_id": "id",
                        "api_base": "base",
                        "custom_llm_provider": "openai",
                    },
                }),
            ),
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

    fn handled(prometheus: &Prometheus, hook: Hook, values: Vec<(Field, Value)>) -> HookOutput {
        handled_with(prometheus, hook, values, None)
    }

    fn handled_with(
        prometheus: &Prometheus,
        hook: Hook,
        values: Vec<(Field, Value)>,
        router: Option<RouterRef>,
    ) -> HookOutput {
        let mut fields = FieldValues::new(FieldSet::of(&READS));
        for (field, value) in values {
            fields.insert(field, value);
        }
        if let Some(router) = router {
            fields.set_router(router);
        }
        let input = HookInput {
            hook,
            fields,
            call: None,
        };
        prometheus
            .handle(&input, None)
            .now_or_never()
            .unwrap_or_else(|| panic!("the handler completes without waiting"))
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

    fn stop(output: HookOutput) -> Option<Stop> {
        match output {
            Ok(_) => None,
            Err(CallbackError::Failed(text)) => {
                Some(serde_json::from_str(&text).unwrap_or_else(|error| panic!("{error}: {text}")))
            }
            Err(other) => panic!("unexpected error {other}"),
        }
    }

    #[rstest]
    fn the_manifest_is_one_awaited_attempt_ended_read_that_validates() {
        let manifest = Prometheus::declared();
        assert_eq!(manifest.validate(), Ok(()));
        assert_eq!(manifest.selection.activation, Activation::Required);
        assert_eq!(
            manifest
                .subscriptions
                .iter()
                .map(|subscription| (
                    subscription.hook,
                    subscription.awaited,
                    subscription.reads.contains(Field::DeprecatedKwargs)
                ))
                .collect::<Vec<_>>(),
            vec![(Hook::AttemptEnded, true, true)]
        );
    }

    #[rstest]
    fn a_success_runs_every_step(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            attempt("success", json!(10), false),
        );
        assert!(matches!(output, Ok(None)));
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
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            attempt("failure", json!(10), false),
        );
        assert!(matches!(output, Ok(None)));
        assert_eq!(
            total(&prometheus, "litellm_llm_api_failed_requests_metric"),
            vec![1.0]
        );
        assert!(total(&prometheus, "litellm_requests_metric").is_empty());
    }

    #[rstest]
    fn a_raise_reports_its_step_and_python_class(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            attempt("success", json!("ten"), false),
        );
        assert_eq!(
            stop(output),
            Some(Stop {
                step: Step::SuccessHead,
                kind: StopKind::Value,
                message: "could not convert string to float: 'ten'".to_owned(),
                error_type: None,
            })
        );
        assert!(total(&prometheus, "litellm_proxy_total_requests_metric").is_empty());
    }

    #[rstest]
    fn the_input_sequence_error_stops_before_latency_with_its_python_class(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            attempt("success", json!(10), true),
        );
        assert_eq!(
            stop(output),
            Some(Stop {
                step: Step::SuccessLatency,
                kind: StopKind::Named,
                message: "'NoneType' object has no attribute 'get'".to_owned(),
                error_type: Some("AttributeError".to_owned()),
            })
        );
        assert_eq!(total(&prometheus, "litellm_requests_metric"), vec![1.0]);
        assert!(total(&prometheus, "litellm_proxy_total_requests_metric").is_empty());
    }

    #[rstest]
    #[case::other_hook(Hook::RouterEvent, attempt("success", json!(10), false))]
    #[case::no_value(Hook::AttemptEnded, Vec::new())]
    #[case::fallback_outcome(Hook::AttemptEnded, attempt("fallback_success", json!(10), false))]
    fn nothing_else_records(
        prometheus: Prometheus,
        #[case] hook: Hook,
        #[case] values: Vec<(Field, Value)>,
    ) {
        assert!(matches!(handled(&prometheus, hook, values), Ok(None)));
        assert!(
            FAMILIES
                .iter()
                .all(|(name, _)| total(&prometheus, name).is_empty())
        );
    }

    #[rstest]
    fn a_malformed_attempt_is_an_error(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            vec![
                (Field::Outcome, json!("success")),
                (Field::DeprecatedKwargs, json!({"event": "nope"})),
            ],
        );
        assert!(
            matches!(output, Err(CallbackError::Failed(text)) if text.starts_with("prometheus attempt: "))
        );
    }

    const NO_GAUGES: [Vec<f64>; 2] = [Vec::new(), Vec::new()];

    fn gauges(prometheus: &Prometheus) -> [Vec<f64>; 2] {
        [
            total(prometheus, "litellm_remaining_tokens_metric"),
            total(prometheus, "litellm_remaining_requests_metric"),
        ]
    }

    #[rstest]
    fn a_success_sets_both_router_gauges_from_the_remaining_usage(prometheus: Prometheus) {
        let router = Router::answering(Ok(limits()));
        let output = handled_with(
            &prometheus,
            Hook::AttemptEnded,
            attempt("success", json!(10), false),
            Some(RouterRef::new(router.clone())),
        );
        assert!(matches!(output, Ok(None)));
        assert_eq!(router.asked(), vec!["group".to_owned()]);
        assert_eq!(gauges(&prometheus), [vec![75.0], vec![9.0]]);
    }

    #[rstest]
    #[case::tokens_only(json!({"x-ratelimit-remaining-tokens": 12345}), [vec![12345.0], vec![]])]
    #[case::float(json!({"x-ratelimit-remaining-requests": 2.5}), [vec![], vec![2.5]])]
    #[case::empty(json!({}), [vec![], vec![]])]
    fn only_the_gauges_the_router_reports_are_set(
        prometheus: Prometheus,
        #[case] answer: Value,
        #[case] expected: [Vec<f64>; 2],
    ) {
        let headers = answer
            .as_object()
            .map(|map| {
                map.iter()
                    .map(|(name, value)| {
                        let name: &'static str = if name.ends_with("tokens") {
                            "x-ratelimit-remaining-tokens"
                        } else {
                            "x-ratelimit-remaining-requests"
                        };
                        let amount = value.as_i64().map_or_else(
                            || UsageAmount::Float(value.as_f64().unwrap_or_default()),
                            |int| UsageAmount::Int(i128::from(int)),
                        );
                        (name, amount)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let router = Router::answering(Ok(headers));
        let output = handled_with(
            &prometheus,
            Hook::AttemptEnded,
            attempt("success", json!(10), false),
            Some(RouterRef::new(router)),
        );
        assert!(matches!(output, Ok(None)));
        assert_eq!(gauges(&prometheus), expected);
    }

    #[rstest]
    fn both_provider_headers_skip_the_router_read(prometheus: Prometheus) {
        let router = Router::answering(Ok(limits()));
        let output = handled_with(
            &prometheus,
            Hook::AttemptEnded,
            with_provider_headers(json!({
                "provider_remaining_tokens": 999,
                "provider_remaining_requests": 99,
            })),
            Some(RouterRef::new(router.clone())),
        );
        assert!(matches!(output, Ok(None)));
        assert!(router.asked().is_empty());
        assert_eq!(gauges(&prometheus), NO_GAUGES);
    }

    #[rstest]
    #[case::requests_header(json!({"provider_remaining_requests": 7}), [vec![75.0], vec![]])]
    #[case::tokens_header(json!({"provider_remaining_tokens": 7}), [vec![], vec![9.0]])]
    #[case::unreadable_tokens_header(json!({"rejected": ["provider_remaining_tokens"]}), [vec![], vec![9.0]])]
    fn one_provider_header_leaves_only_the_missing_gauge_to_the_router(
        prometheus: Prometheus,
        #[case] state: Value,
        #[case] expected: [Vec<f64>; 2],
    ) {
        let router = Router::answering(Ok(limits()));
        let output = handled_with(
            &prometheus,
            Hook::AttemptEnded,
            with_provider_headers(state),
            Some(RouterRef::new(router.clone())),
        );
        assert!(matches!(output, Ok(None)));
        assert_eq!(router.asked(), vec!["group".to_owned()]);
        assert_eq!(gauges(&prometheus), expected);
    }

    #[rstest]
    fn no_router_sets_no_router_gauge(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            attempt("success", json!(10), false),
        );
        assert!(matches!(output, Ok(None)));
        assert_eq!(gauges(&prometheus), NO_GAUGES);
    }

    #[rstest]
    #[case::none(Value::Null)]
    #[case::empty(json!(""))]
    fn a_request_without_a_model_group_is_not_read(
        prometheus: Prometheus,
        #[case] model_group: Value,
    ) {
        let router = Router::answering(Ok(limits()));
        let mut values = attempt("success", json!(10), false);
        for (field, value) in &mut values {
            if *field == Field::DeprecatedKwargs {
                value["payload_labels"]["model_group"] = model_group.clone();
            }
        }
        let output = handled_with(
            &prometheus,
            Hook::AttemptEnded,
            values,
            Some(RouterRef::new(router.clone())),
        );
        assert!(matches!(output, Ok(None)));
        assert!(router.asked().is_empty());
        assert_eq!(gauges(&prometheus), NO_GAUGES);
    }

    #[rstest]
    fn a_failed_read_is_logged_and_is_not_a_callback_failure() {
        let (prometheus, lines) = logging();
        let router = Router::answering(Err("router boom".to_owned()));
        let output = handled_with(
            &prometheus,
            Hook::AttemptEnded,
            attempt("success", json!(10), false),
            Some(RouterRef::new(router)),
        );
        assert!(matches!(output, Ok(None)));
        assert_eq!(gauges(&prometheus), NO_GAUGES);
        assert_eq!(
            lines.taken().pop(),
            Some(LogLine::new(
                crate::Level::Error,
                "Prometheus: get_remaining_model_group_usage failed for model_group=%s: %s",
                vec!["group".to_owned(), "router boom".to_owned()],
            ))
        );
    }

    #[rstest]
    fn a_failure_never_reads_the_router(prometheus: Prometheus) {
        let router = Router::answering(Ok(limits()));
        let output = handled_with(
            &prometheus,
            Hook::AttemptEnded,
            attempt("failure", json!(10), false),
            Some(RouterRef::new(router.clone())),
        );
        assert!(matches!(output, Ok(None)));
        assert!(router.asked().is_empty());
    }

    #[rstest]
    fn a_stop_before_the_tail_skips_the_router_read(prometheus: Prometheus) {
        let router = Router::answering(Ok(limits()));
        let output = handled_with(
            &prometheus,
            Hook::AttemptEnded,
            attempt("success", json!("ten"), false),
            Some(RouterRef::new(router.clone())),
        );
        assert_eq!(stop(output).map(|stop| stop.step), Some(Step::SuccessHead));
        assert!(router.asked().is_empty());
        assert_eq!(gauges(&prometheus), NO_GAUGES);
    }

    #[rstest]
    fn a_stop_in_the_tail_still_reads_the_router_before_it_is_reported(prometheus: Prometheus) {
        let router = Router::answering(Ok(limits()));
        let mut values = attempt("success", json!(10), false);
        for (field, value) in &mut values {
            if *field == Field::Cost {
                *value = json!({"response_cost": "0.5"});
            }
        }
        values.push((
            Field::McpToolCall,
            json!({"name": "tool", "server_name": "server"}),
        ));
        let output = handled_with(
            &prometheus,
            Hook::AttemptEnded,
            values,
            Some(RouterRef::new(router.clone())),
        );
        assert_eq!(stop(output).map(|stop| stop.step), Some(Step::SuccessTail));
        assert_eq!(router.asked(), vec!["group".to_owned()]);
        assert_eq!(gauges(&prometheus), [vec![75.0], vec![9.0]]);
    }
}
