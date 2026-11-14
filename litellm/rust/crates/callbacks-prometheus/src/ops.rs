use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    attempt::Moment,
    emit::{Emitter, Labels, py_float},
    labels::{LabelValues, Settings, is_py_dict, label_strings, sanitize_label_value},
    pyfmt::py_str,
    store::{Op, Store, StoreError},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Value,
    Key,
    Type,
    Runtime,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpError {
    pub kind: ErrorKind,
    pub message: String,
}

impl OpError {
    fn value(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Value,
            message: message.into(),
        }
    }
}

impl From<StoreError> for OpError {
    fn from(error: StoreError) -> Self {
        let kind = match error {
            StoreError::MostRecent(_) => ErrorKind::Runtime,
            StoreError::Missing => ErrorKind::Key,
            _ => ErrorKind::Value,
        };
        Self {
            kind,
            message: error.to_string(),
        }
    }
}

impl From<litellm_callbacks::CallbackError> for OpError {
    fn from(error: litellm_callbacks::CallbackError) -> Self {
        let message = match error {
            litellm_callbacks::CallbackError::Failed(message) => message,
            other => other.to_string(),
        };
        if let Some(key) = message.strip_prefix("KeyError: ") {
            return Self {
                kind: ErrorKind::Key,
                message: key.trim_matches('\'').to_owned(),
            };
        }
        if let Some(rest) = message.strip_prefix("TypeError: ") {
            return Self {
                kind: ErrorKind::Type,
                message: rest.to_owned(),
            };
        }
        if let Some(rest) = message.strip_prefix("ValueError: ") {
            return Self::value(rest);
        }
        if message.contains("must not be used with the mostrecent mode") {
            return Self {
                kind: ErrorKind::Runtime,
                message,
            };
        }
        Self::value(message)
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
pub enum OpName {
    #[strum(serialize = "deployment_state")]
    DeploymentState,
    #[strum(serialize = "deployment_failure")]
    DeploymentFailure,
    #[strum(serialize = "cooled_down")]
    CooledDown,
    #[strum(serialize = "guardrail")]
    Guardrail,
    #[strum(serialize = "managed_batch_created")]
    ManagedBatchCreated,
    #[strum(serialize = "managed_file_created")]
    ManagedFileCreated,
    #[strum(serialize = "managed_file_size")]
    ManagedFileSize,
    #[strum(serialize = "managed_batch_duration")]
    ManagedBatchDuration,
    #[strum(serialize = "managed_file_deleted")]
    ManagedFileDeleted,
    #[strum(serialize = "check_batch_cost_run")]
    CheckBatchCostRun,
    #[strum(serialize = "check_batch_cost_error")]
    CheckBatchCostError,
    #[strum(serialize = "callback_failure")]
    CallbackFailure,
    #[strum(serialize = "provider_budget")]
    ProviderBudget,
    #[strum(serialize = "spend_capture_rate")]
    SpendCaptureRate,
    #[strum(serialize = "dynamic_limiter_share")]
    DynamicLimiterShare,
    #[strum(serialize = "dynamic_limiter_throttled")]
    DynamicLimiterThrottled,
    #[strum(serialize = "fallback")]
    Fallback,
    #[strum(serialize = "proxy_failure")]
    ProxyFailure,
    #[strum(serialize = "budget")]
    Budget,
    #[strum(serialize = "team_members")]
    TeamMembers,
    #[strum(serialize = "counts")]
    Counts,
}

type OpResult = Result<(), OpError>;

#[derive(Deserialize)]
struct Facts {
    #[serde(default)]
    settings: Settings,
    #[serde(flatten)]
    rest: Map<String, Value>,
}

fn field<'a>(facts: &'a Map<String, Value>, key: &str) -> &'a Value {
    facts.get(key).unwrap_or(&Value::Null)
}

fn values_of(facts: &Map<String, Value>) -> LabelValues {
    is_py_dict(field(facts, "values")).map_or_else(LabelValues::default, LabelValues::from_enum)
}

fn text(value: &Value) -> String {
    py_str(value)
}

fn or_empty(value: &Value) -> Value {
    if crate::pyfmt::py_truthy(value) {
        value.clone()
    } else {
        Value::from("")
    }
}

fn hours_until(reset_at: Moment, now: Moment) -> Result<f64, OpError> {
    if reset_at.aware != now.aware {
        return Err(OpError {
            kind: ErrorKind::Type,
            message: "can't subtract offset-naive and offset-aware datetimes".into(),
        });
    }
    #[allow(clippy::cast_precision_loss)]
    let seconds = (reset_at.us - now.us) as f64 / 1_000_000.0;
    Ok(seconds / 3600.0)
}

fn remaining_budget(max_budget: &Value, spend: &Value) -> Result<f64, OpError> {
    if max_budget.is_null() {
        return Ok(f64::INFINITY);
    }
    let max = py_float(max_budget)?;
    if spend.is_null() {
        return Ok(max);
    }
    Ok(max - py_float(spend)?)
}

pub struct Ops<'a> {
    store: &'a Store,
    settings: &'a Settings,
    facts: &'a Map<String, Value>,
}

impl Ops<'_> {
    const fn emit(&self) -> Emitter<'_> {
        Emitter {
            store: self.store,
            settings: self.settings,
        }
    }

    fn named(&self, metric: &str, pairs: &[(&str, Value)], op: Op) -> OpResult {
        self.named_with(metric, pairs, || Ok(op))
    }

    fn named_with(
        &self,
        metric: &str,
        pairs: &[(&str, Value)],
        op: impl FnOnce() -> Result<Op, OpError>,
    ) -> OpResult {
        let labels = Labels {
            names: pairs.iter().map(|(name, _)| (*name).to_owned()).collect(),
            values: pairs.iter().map(|(_, value)| Some(text(value))).collect(),
        };
        self.emit().record_with(metric, &labels, op, false)
    }

    fn positional(&self, metric: &str, values: &[String], op: Op) -> OpResult {
        self.positional_with(metric, values, || Ok(op))
    }

    fn positional_with(
        &self,
        metric: &str,
        values: &[String],
        op: impl FnOnce() -> Result<Op, OpError>,
    ) -> OpResult {
        let Some(id) = self.store.family(metric) else {
            return Ok(());
        };
        let target = self.store.labels(id, values)?;
        self.store.apply(id, &target, op()?)?;
        Ok(())
    }

    fn factory(&self, metric: &str, values: &LabelValues, op: Op, track: bool) -> OpResult {
        self.factory_with(metric, values, || Ok(op), track)
    }

    fn factory_with(
        &self,
        metric: &str,
        values: &LabelValues,
        op: impl FnOnce() -> Result<Op, OpError>,
        track: bool,
    ) -> OpResult {
        let context = values.context(self.settings);
        let labels = self.emit().labels(metric, &context);
        self.emit().record_with(metric, &labels, op, track)
    }

    fn number(&self, key: &str) -> Result<f64, OpError> {
        Ok(py_float(field(self.facts, key))?)
    }

    fn present(&self, metric: &str) -> bool {
        self.store.family(metric).is_some()
    }

    fn observed(&self, key: &str) -> Result<f64, OpError> {
        let value = field(self.facts, key);
        match value {
            Value::Bool(_) => self.number(key),
            _ if crate::pyfmt::is_py_number(value) => self.number(key),
            other => Err(OpError {
                kind: ErrorKind::Type,
                message: format!(
                    "unsupported operand type(s) for +=: 'float' and '{}'",
                    crate::pyfmt::py_type_name(other)
                ),
            }),
        }
    }

    fn deployment_state(&self) -> OpResult {
        let values = LabelValues::default()
            .with(
                "litellm_model_name",
                field(self.facts, "litellm_model_name").clone(),
            )
            .with("model_id", field(self.facts, "model_id").clone())
            .with("api_base", field(self.facts, "api_base").clone())
            .with("api_provider", field(self.facts, "api_provider").clone());
        self.factory_with(
            "litellm_deployment_state",
            &values,
            || Ok(Op::Set(self.number("state")?)),
            false,
        )
    }

    fn deployment_failure(&self) -> OpResult {
        let values = values_of(self.facts);
        if field(self.facts, "has_exception") == &Value::Bool(true) {
            self.factory(
                "litellm_deployment_failure_responses",
                &values,
                Op::Inc(1.0),
                true,
            )?;
        }
        self.factory(
            "litellm_deployment_total_requests",
            &values,
            Op::Inc(1.0),
            true,
        )
    }

    fn cooled_down(&self) -> OpResult {
        let values: Vec<Option<String>> = match field(self.facts, "values") {
            Value::Array(items) => items.iter().map(sanitize_label_value).collect(),
            _ => Vec::new(),
        };
        self.positional(
            "litellm_deployment_cooled_down",
            &label_strings(&values),
            Op::Inc(1.0),
        )
    }

    fn guardrail(&self) -> OpResult {
        let name = field(self.facts, "guardrail_name").clone();
        let status = field(self.facts, "status").clone();
        let error_type = field(self.facts, "error_type").clone();
        let hook_type = field(self.facts, "hook_type").clone();
        if self.present("litellm_guardrail_latency_seconds") {
            self.named_with(
                "litellm_guardrail_latency_seconds",
                &[
                    ("guardrail_name", name.clone()),
                    ("status", status.clone()),
                    ("error_type", or_none(&error_type)),
                    ("hook_type", hook_type.clone()),
                ],
                || Ok(Op::Observe(self.number("latency_seconds")?)),
            )?;
        }
        self.named(
            "litellm_guardrail_requests_total",
            &[
                ("guardrail_name", name.clone()),
                ("status", status.clone()),
                ("hook_type", hook_type.clone()),
            ],
            Op::Inc(1.0),
        )?;
        if status.as_str() == Some("error") && crate::pyfmt::py_truthy(&error_type) {
            self.named(
                "litellm_guardrail_errors_total",
                &[
                    ("guardrail_name", name),
                    ("error_type", error_type),
                    ("hook_type", hook_type),
                ],
                Op::Inc(1.0),
            )?;
        }
        Ok(())
    }

    fn identity(&self, metric: &str) -> OpResult {
        self.named(
            metric,
            &[
                ("model", field(self.facts, "model").clone()),
                ("api_provider", field(self.facts, "api_provider").clone()),
                ("user", field(self.facts, "user").clone()),
                ("user_email", field(self.facts, "user_email").clone()),
                ("api_key_alias", field(self.facts, "api_key_alias").clone()),
            ],
            Op::Inc(1.0),
        )
    }

    fn file_size(&self) -> OpResult {
        if !self.present("litellm_managed_file_size_bytes") {
            return Ok(());
        }
        self.named_with(
            "litellm_managed_file_size_bytes",
            &[
                ("purpose", field(self.facts, "purpose").clone()),
                ("file_type", field(self.facts, "file_type").clone()),
                ("model", or_empty(field(self.facts, "model"))),
                ("api_provider", or_empty(field(self.facts, "api_provider"))),
                ("user", or_empty(field(self.facts, "user"))),
            ],
            || Ok(Op::Set(self.number("size_bytes")?)),
        )
    }

    fn batch_duration(&self) -> OpResult {
        if !self.present("litellm_managed_batch_duration_seconds") {
            return Ok(());
        }
        self.named_with(
            "litellm_managed_batch_duration_seconds",
            &[
                ("model", or_empty(field(self.facts, "model"))),
                ("api_provider", or_empty(field(self.facts, "api_provider"))),
            ],
            || Ok(Op::Observe(self.observed("duration_seconds")?)),
        )
    }

    fn file_deleted(&self) -> OpResult {
        self.named(
            "litellm_managed_file_deleted_total",
            &[("result", field(self.facts, "result").clone())],
            Op::Inc(1.0),
        )
    }

    fn check_batch_cost_run(&self) -> OpResult {
        if self.present("litellm_check_batch_cost_last_run_timestamp") {
            self.positional(
                "litellm_check_batch_cost_last_run_timestamp",
                &[],
                Op::Set(self.number("now")?),
            )?;
        }
        if self.present("litellm_check_batch_cost_jobs_polled") {
            self.positional(
                "litellm_check_batch_cost_jobs_polled",
                &[],
                Op::Set(self.number("jobs_polled")?),
            )?;
        }
        if let Value::Array(models) = field(self.facts, "processed_models") {
            for pair in models {
                let (model, provider) = match pair {
                    Value::Array(items) if items.len() == 2 => (&items[0], &items[1]),
                    _ => return Err(OpError::value("not enough values to unpack")),
                };
                self.named(
                    "litellm_check_batch_cost_jobs_processed_total",
                    &[
                        ("model", or_empty(model)),
                        ("api_provider", or_empty(provider)),
                    ],
                    Op::Inc(1.0),
                )?;
            }
        }
        Ok(())
    }

    fn check_batch_cost_error(&self) -> OpResult {
        self.named(
            "litellm_check_batch_cost_errors_total",
            &[("error_type", field(self.facts, "error_type").clone())],
            Op::Inc(1.0),
        )
    }

    fn callback_failure(&self) -> OpResult {
        self.named(
            "litellm_callback_logging_failures_metric",
            &[("callback_name", field(self.facts, "callback_name").clone())],
            Op::Inc(1.0),
        )
    }

    fn provider_budget(&self) -> OpResult {
        self.positional_with(
            "litellm_provider_remaining_budget_metric",
            &[text(field(self.facts, "provider"))],
            || {
                Ok(Op::Set(remaining_budget(
                    field(self.facts, "budget_limit"),
                    field(self.facts, "spend"),
                )?))
            },
        )
    }

    fn spend_capture_rate(&self) -> OpResult {
        let metric = "litellm_spend_capture_rate";
        let values =
            LabelValues::default().with("api_provider", field(self.facts, "api_provider").clone());
        let rate = field(self.facts, "capture_rate");
        let value = || -> Result<Op, OpError> {
            Ok(Op::Set(if rate.is_null() {
                f64::NAN
            } else {
                py_float(rate)?
            }))
        };
        let labels = self.emit().labels(metric, &values.context(self.settings));
        if labels.names.is_empty() {
            return self.positional_with(metric, &[], value);
        }
        self.emit().record_with(metric, &labels, value, false)
    }

    fn limiter_values(&self) -> LabelValues {
        LabelValues::default()
            .with("model_group", field(self.facts, "model_group").clone())
            .with("team", field(self.facts, "team").clone())
            .with("team_alias", field(self.facts, "team_alias").clone())
            .with("priority", field(self.facts, "priority").clone())
    }

    fn dynamic_limiter_share(&self) -> OpResult {
        let values = self.limiter_values();
        for (metric, key) in [
            ("litellm_dynamic_limiter_share_tokens", "share_tokens"),
            ("litellm_dynamic_limiter_borrowed_tokens", "borrowed_tokens"),
        ] {
            if self.store.family(metric).is_none() {
                continue;
            }
            self.factory_with(metric, &values, || Ok(Op::Set(self.number(key)?)), false)?;
        }
        Ok(())
    }

    fn dynamic_limiter_throttled(&self) -> OpResult {
        let values = self
            .limiter_values()
            .with(
                "rate_limit_type",
                field(self.facts, "rate_limit_type").clone(),
            )
            .with("reason", field(self.facts, "reason").clone());
        self.factory(
            "litellm_dynamic_limiter_throttled_total",
            &values,
            Op::Inc(1.0),
            true,
        )
    }

    fn fallback(&self) -> OpResult {
        let metric = if field(self.facts, "success") == &Value::Bool(true) {
            "litellm_deployment_successful_fallbacks"
        } else {
            "litellm_deployment_failed_fallbacks"
        };
        self.factory(metric, &values_of(self.facts), Op::Inc(1.0), true)
    }

    fn proxy_failure(&self) -> OpResult {
        let values = values_of(self.facts);
        self.factory(
            "litellm_proxy_failed_requests_metric",
            &values,
            Op::Inc(1.0),
            true,
        )?;
        self.factory(
            "litellm_proxy_total_requests_metric",
            &values,
            Op::Inc(1.0),
            true,
        )
    }

    fn moment(&self, key: &str) -> Result<Option<Moment>, OpError> {
        let raw = field(self.facts, key);
        if raw.is_null() {
            return Ok(None);
        }
        serde_json::from_value(raw.clone())
            .map(Some)
            .map_err(|error| OpError::value(error.to_string()))
    }

    fn budget(&self) -> OpResult {
        let entity = text(field(self.facts, "entity"));
        let (remaining, max, hours) = match entity.as_str() {
            "team" => (
                "litellm_remaining_team_budget_metric",
                "litellm_team_max_budget_metric",
                "litellm_team_budget_remaining_hours_metric",
            ),
            "org" => (
                "litellm_remaining_org_budget_metric",
                "litellm_org_max_budget_metric",
                "litellm_org_budget_remaining_hours_metric",
            ),
            "api_key" => (
                "litellm_remaining_api_key_budget_metric",
                "litellm_api_key_max_budget_metric",
                "litellm_api_key_budget_remaining_hours_metric",
            ),
            "user" => (
                "litellm_remaining_user_budget_metric",
                "litellm_user_max_budget_metric",
                "litellm_user_budget_remaining_hours_metric",
            ),
            "customer" => (
                "litellm_remaining_customer_budget_metric",
                "litellm_customer_max_budget_metric",
                "litellm_customer_budget_remaining_hours_metric",
            ),
            other => return Err(OpError::value(format!("unknown budget entity {other}"))),
        };
        let context = values_of(self.facts).context(self.settings);
        let customer = entity == "customer";
        let mut labels = self.emit().labels(remaining, &context);
        if customer {
            let end_user = labels.names.iter().position(|name| name == "end_user");
            if end_user
                .and_then(|position| labels.values[position].as_ref())
                .is_none()
            {
                return Ok(());
            }
        }
        let max_budget = field(self.facts, "max_budget");
        let spend = field(self.facts, "spend");
        self.emit().record_with(
            remaining,
            &labels,
            || Ok::<_, OpError>(Op::Set(remaining_budget(max_budget, spend)?)),
            customer,
        )?;
        if !max_budget.is_null() {
            if !customer {
                labels = self.emit().labels(max, &context);
            }
            if self.present(max) {
                self.emit().record_with(
                    max,
                    &labels,
                    || Ok::<_, OpError>(Op::Set(py_float(max_budget)?)),
                    customer,
                )?;
            }
        }
        if let Some(reset_at) = self.moment("budget_reset_at")? {
            let now = self
                .moment("now")?
                .ok_or_else(|| OpError::value("now is required"))?;
            if !customer && entity != "api_key" {
                labels = self.emit().labels(hours, &context);
            }
            self.emit().record_with(
                hours,
                &labels,
                || Ok::<_, OpError>(Op::Set(hours_until(reset_at, now)?)),
                customer,
            )?;
        }
        Ok(())
    }

    fn team_members(&self) -> OpResult {
        self.factory(
            "litellm_team_members_metric",
            &values_of(self.facts),
            Op::Set(self.number("count")?),
            false,
        )
    }

    fn counts(&self) -> OpResult {
        for (metric, key) in [
            ("litellm_total_users", "total_users"),
            ("litellm_active_users", "active_users"),
            ("litellm_teams_count", "teams"),
        ] {
            if self.facts.contains_key(key) {
                self.positional(metric, &[], Op::Set(self.number(key)?))?;
            }
        }
        Ok(())
    }

    fn run(&self, op: OpName) -> OpResult {
        match op {
            OpName::DeploymentState => self.deployment_state(),
            OpName::DeploymentFailure => self.deployment_failure(),
            OpName::CooledDown => self.cooled_down(),
            OpName::Guardrail => self.guardrail(),
            OpName::ManagedBatchCreated => self.identity("litellm_managed_batch_created_total"),
            OpName::ManagedFileCreated => self.identity("litellm_managed_file_created_total"),
            OpName::ManagedFileSize => self.file_size(),
            OpName::ManagedBatchDuration => self.batch_duration(),
            OpName::ManagedFileDeleted => self.file_deleted(),
            OpName::CheckBatchCostRun => self.check_batch_cost_run(),
            OpName::CheckBatchCostError => self.check_batch_cost_error(),
            OpName::CallbackFailure => self.callback_failure(),
            OpName::ProviderBudget => self.provider_budget(),
            OpName::SpendCaptureRate => self.spend_capture_rate(),
            OpName::DynamicLimiterShare => self.dynamic_limiter_share(),
            OpName::DynamicLimiterThrottled => self.dynamic_limiter_throttled(),
            OpName::Fallback => self.fallback(),
            OpName::ProxyFailure => self.proxy_failure(),
            OpName::Budget => self.budget(),
            OpName::TeamMembers => self.team_members(),
            OpName::Counts => self.counts(),
        }
    }
}

fn or_none(value: &Value) -> Value {
    if crate::pyfmt::py_truthy(value) {
        value.clone()
    } else {
        Value::from("none")
    }
}

pub fn apply(store: &Store, op: &str, facts: &str) -> OpResult {
    let op: OpName = op
        .parse()
        .map_err(|_| OpError::value(format!("unknown prometheus op {op}")))?;
    let facts: Facts = serde_json::from_str(facts)
        .map_err(|error| OpError::value(format!("prometheus op facts: {error}")))?;
    Ops {
        store,
        settings: &facts.settings,
        facts: &facts.rest,
    }
    .run(op)
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;
    use crate::{
        store::{FamilySpec, Kind, ManualClock},
        tracker::SeriesLimits,
    };

    #[fixture]
    fn store() -> Store {
        let store = Store::new(SeriesLimits::default(), Box::new(ManualClock::default()));
        for name in [
            "litellm_deployment_failure_responses",
            "litellm_deployment_total_requests",
        ] {
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

    fn counted(store: &Store, name: &str) -> Vec<(Vec<String>, f64)> {
        let id = store
            .family(name)
            .unwrap_or_else(|| panic!("{name} is registered"));
        store
            .snapshot(id)
            .series
            .into_iter()
            .map(|(values, series)| (values, series.value))
            .collect()
    }

    #[rstest]
    fn an_unknown_op_is_a_value_error(store: Store) {
        assert_eq!(
            apply(&store, "nope", "{}"),
            Err(OpError::value("unknown prometheus op nope"))
        );
    }

    #[rstest]
    #[case::with_exception(true, vec![(vec!["team-1".to_owned()], 1.0)])]
    #[case::without_exception(false, vec![])]
    fn a_deployment_failure_counts_the_request_and_only_a_raised_failure(
        store: Store,
        #[case] has_exception: bool,
        #[case] failures: Vec<(Vec<String>, f64)>,
    ) {
        let facts = json!({"values": {"team": "team-1"}, "has_exception": has_exception});
        assert_eq!(
            apply(&store, "deployment_failure", &facts.to_string()),
            Ok(())
        );
        assert_eq!(
            counted(&store, "litellm_deployment_total_requests"),
            vec![(vec!["team-1".to_owned()], 1.0)]
        );
        assert_eq!(
            counted(&store, "litellm_deployment_failure_responses"),
            failures
        );
    }
}
