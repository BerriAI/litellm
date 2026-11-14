use std::cell::RefCell;

use litellm_callbacks::{
    CallbackError, GuardrailEntry, LoggedKeyIdentity, MetadataKind, Scalar, UsageRecord,
};
use serde_json::{Map, Value};

use crate::{
    attempt::{AttemptFacts, Moment, moment, py, rejected},
    emit::{
        Emitter, failed, not_comparable, not_subtractable, py_float, py_int, py_int_or_bool,
        py_positive_number,
    },
    labels::{LabelValues, Settings, tag_match_error},
    pyfmt::{
        ascii_decimal_digits, is_py_number, py_float_int, py_number, py_str, py_truthy,
        py_type_name,
    },
    store::Op,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuccessStep {
    Head,
    Limits,
    Latency,
    Tail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum Level {
    #[strum(serialize = "debug")]
    Debug,
    #[strum(serialize = "error")]
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogLine {
    pub level: Level,
    pub format: &'static str,
    pub args: Vec<String>,
}

impl LogLine {
    pub const fn new(level: Level, format: &'static str, args: Vec<String>) -> Self {
        Self {
            level,
            format,
            args,
        }
    }
}

pub fn exception_text(error: &CallbackError) -> String {
    let message = match error {
        CallbackError::Failed(message) => message.clone(),
        other => other.to_string(),
    };
    for prefix in ["KeyError: ", "TypeError: ", "ValueError: "] {
        if let Some(rest) = message.strip_prefix(prefix) {
            return rest.to_owned();
        }
    }
    message
}

pub fn duration(start: Option<Moment>, end: Option<Moment>) -> Result<Option<f64>, CallbackError> {
    match (start, end) {
        (Some(start), Some(end)) if start.aware != end.aware => Err(failed(
            "TypeError: can't subtract offset-naive and offset-aware datetimes",
        )),
        #[allow(clippy::cast_precision_loss)]
        (Some(start), Some(end)) => Ok(Some((end.us - start.us) as f64 / 1_000_000.0)),
        _ => Ok(None),
    }
}

pub fn required<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a Value, CallbackError> {
    map.get(key)
        .ok_or_else(|| failed(format!("KeyError: '{key}'")))
}

pub fn get<'a>(map: &'a Map<String, Value>, key: &str) -> &'a Value {
    map.get(key).unwrap_or(&Value::Null)
}

pub fn metadata(facts: &AttemptFacts) -> Result<&LoggedKeyIdentity, CallbackError> {
    let identity = facts
        .identity
        .as_ref()
        .ok_or_else(|| failed("standard_logging_object is not a mapping"))?;
    match identity.metadata_kind {
        Some(MetadataKind::Mapping) => Ok(identity),
        Some(MetadataKind::Missing) => Err(failed("KeyError: 'metadata'")),
        _ => Err(failed("metadata is not subscriptable")),
    }
}

pub fn stamped(
    identity: &LoggedKeyIdentity,
    key: &str,
    value: Option<&Scalar>,
) -> Result<Value, CallbackError> {
    if identity
        .missing_keys
        .as_ref()
        .is_some_and(|missing| missing.iter().any(|name| name == key))
    {
        return Err(failed(format!("KeyError: '{key}'")));
    }
    Ok(py(value))
}

pub struct Success<'a> {
    pub emit: Emitter<'a>,
    pub facts: &'a AttemptFacts,
    pub input_sequence_length: Value,
    pub logs: &'a RefCell<Vec<LogLine>>,
}

impl Success<'_> {
    fn settings(&self) -> &Settings {
        self.facts.settings()
    }

    pub fn run(&self, step: SuccessStep) -> Result<(), CallbackError> {
        let facts = self.facts;
        let identity = metadata(facts)?;
        let labels = &facts.residual.payload_labels;
        let user_id = stamped(identity, "user_api_key_user_id", identity.user_id.as_ref())?;
        let hashed_api_key = stamped(identity, "user_api_key_hash", identity.key_hash.as_ref())?;
        let api_key_alias = stamped(identity, "user_api_key_alias", identity.key_alias.as_ref())?;
        let team = stamped(identity, "user_api_key_team_id", identity.team_id.as_ref())?;
        let team_alias = stamped(
            identity,
            "user_api_key_team_alias",
            identity.team_alias.as_ref(),
        )?;
        let response_cost = py(facts.cost.response_cost.as_ref());
        let tags = facts.tag_values()?;
        let model_group = required(labels, "model_group")?;
        let user_email = stamped(
            identity,
            "user_api_key_user_email",
            identity.user_email.as_ref(),
        )?;
        let model_id = required(labels, "model_id")?;
        let api_base = required(labels, "api_base")?;
        let api_provider = required(labels, "custom_llm_provider")?;
        let model = &facts.model;
        let stream = if self.settings().emit_stream_label {
            Value::from(py_str(&py(facts.request.stream.as_ref())))
        } else {
            Value::Null
        };
        let values = LabelValues {
            tags,
            custom_metadata_labels: facts.custom_metadata_labels(),
            ..LabelValues::default()
        }
        .with("end_user", facts.end_user_id())
        .with("hashed_api_key", hashed_api_key)
        .with("api_key_alias", api_key_alias)
        .with("requested_model", model_group.clone())
        .with("model_group", model_group.clone())
        .with("team", team)
        .with("team_alias", team_alias)
        .with("project_id", py(identity.project_id.as_ref()))
        .with("project_alias", py(identity.project_alias.as_ref()))
        .with("org_id", py(identity.org_id.as_ref()))
        .with("org_alias", py(identity.org_alias.as_ref()))
        .with("user", user_id)
        .with("user_email", user_email)
        .with("status_code", Value::from("200"))
        .with("model", model.clone())
        .with("litellm_model_name", model.clone())
        .with("model_id", model_id.clone())
        .with("api_base", api_base.clone())
        .with("api_provider", api_provider.clone())
        .with("route", py(identity.request_route.as_ref()))
        .with("client_ip", py(facts.request.client_ip.as_ref()))
        .with("user_agent", py(facts.request.user_agent.as_ref()))
        .with("stream", stream)
        .with(
            "service_tier",
            facts
                .request
                .service_tier
                .clone()
                .map_or(Value::Null, Value::from),
        );
        match step {
            SuccessStep::Head => {
                let context = values.checked_context(self.settings()).map_err(failed)?;
                self.emit.inc("litellm_requests_metric", &context, 1.0)?;
                self.emit
                    .inc("litellm_spend_metric", &context, py_float(&response_cost)?)?;
                self.zero_cost(&values)?;
                self.tokens(&values)
            }
            SuccessStep::Limits => {
                self.virtual_key(&values)?;
                self.rate_limits(&values)
            }
            SuccessStep::Latency => {
                self.latency(&values)?;
                if let Err(error) = self.deployment(&values) {
                    self.log(LogLine::new(
                        Level::Error,
                        "Prometheus Error: set_llm_deployment_success_metrics. Exception occured - %s",
                        vec![exception_text(&error)],
                    ));
                }
                Ok(())
            }
            SuccessStep::Tail => {
                self.cache(&values)?;
                self.media(&values)?;
                self.mcp(&values, &response_cost)?;
                self.proxy_total(model_group, &values)
            }
        }
    }

    pub fn zero_cost(&self, values: &LabelValues) -> Result<(), CallbackError> {
        let cost = &self.facts.cost;
        let diagnostic = cost.zero_cost.as_ref();
        if diagnostic.is_none() && !rejected(cost.rejected.as_ref(), "zero_cost") {
            return Ok(());
        }
        let name = "litellm_zero_cost_requests_total";
        let mut labels = self.emit.labels(name, &values.context(self.settings()));
        if labels.names.iter().any(|label| label == "reason") {
            let reason = diagnostic
                .filter(|diagnostic| diagnostic.reason_present == Some(true))
                .ok_or_else(|| failed("KeyError: 'reason'"))?;
            labels.set("reason", Some(py_str(&py(reason.reason.as_ref()))));
        }
        self.emit.record(name, &labels, Op::Inc(1.0), false)
    }

    fn tokens(&self, values: &LabelValues) -> Result<(), CallbackError> {
        let usage = &self.facts.usage;
        let context = values.context(self.settings());
        for (name, value) in [
            (
                "litellm_total_tokens_metric",
                py(usage.total_tokens.as_ref()),
            ),
            (
                "litellm_input_tokens_metric",
                py(usage.prompt_tokens.as_ref()),
            ),
            (
                "litellm_output_tokens_metric",
                self.facts.completion_tokens(),
            ),
        ] {
            self.emit.inc(name, &context, py_float(&value)?)?;
        }
        let cache_write = cache_write_tokens(usage).map_or(Value::Null, Value::from);
        let sparse = [
            (
                "litellm_input_cached_tokens_metric",
                py(usage.cached_tokens.as_ref()),
            ),
            ("litellm_input_cache_creation_tokens_metric", cache_write),
            (
                "litellm_input_audio_tokens_metric",
                py(usage.prompt_audio_tokens.as_ref()),
            ),
            (
                "litellm_output_reasoning_tokens_metric",
                py(usage.reasoning_tokens.as_ref()),
            ),
            (
                "litellm_output_audio_tokens_metric",
                py(usage.completion_audio_tokens.as_ref()),
            ),
        ];
        self.sparse(&sparse, &context)
    }

    fn sparse(
        &self,
        counters: &[(&str, Value)],
        context: &crate::labels::LabelContext,
    ) -> Result<(), CallbackError> {
        for (name, value) in counters {
            if let Some(amount) = py_positive_number(value) {
                self.emit.inc(name, context, amount)?;
            }
        }
        Ok(())
    }

    fn rate_limits(&self, values: &LabelValues) -> Result<(), CallbackError> {
        const DESCRIPTORS: [(&str, &str, &str, &str, &str); 8] = [
            (
                "api_key",
                "requests",
                "requests",
                "litellm_api_key_rate_limit_allowed_metric",
                "litellm_api_key_rate_limit_used_metric",
            ),
            (
                "api_key",
                "tokens",
                "tokens",
                "litellm_api_key_rate_limit_allowed_metric",
                "litellm_api_key_rate_limit_used_metric",
            ),
            (
                "team",
                "requests",
                "requests",
                "litellm_team_rate_limit_allowed_metric",
                "litellm_team_rate_limit_used_metric",
            ),
            (
                "team",
                "tokens",
                "tokens",
                "litellm_team_rate_limit_allowed_metric",
                "litellm_team_rate_limit_used_metric",
            ),
            (
                "model_per_project",
                "requests",
                "requests",
                "litellm_project_model_rate_limit_allowed_metric",
                "litellm_project_model_rate_limit_used_metric",
            ),
            (
                "model_per_project",
                "tokens",
                "tokens",
                "litellm_project_model_rate_limit_allowed_metric",
                "litellm_project_model_rate_limit_used_metric",
            ),
            (
                "model_per_project_itpm",
                "tokens",
                "input_tokens",
                "litellm_project_model_rate_limit_allowed_metric",
                "litellm_project_model_rate_limit_used_metric",
            ),
            (
                "model_per_project_otpm",
                "tokens",
                "output_tokens",
                "litellm_project_model_rate_limit_allowed_metric",
                "litellm_project_model_rate_limit_used_metric",
            ),
        ];
        let header = |name: String| -> Option<i128> {
            self.facts
                .limiter_header(&name)
                .and_then(|value| py_int(&py(Some(value))))
        };
        for (descriptor, header_type, rate_limit_type, allowed, used) in DESCRIPTORS {
            let limit = header(format!("x-ratelimit-{descriptor}-limit-{header_type}"));
            let remaining = header(format!("x-ratelimit-{descriptor}-remaining-{header_type}"));
            let context = values
                .clone()
                .with("rate_limit_type", Value::from(rate_limit_type))
                .checked_context(self.settings())
                .map_err(failed)?;
            let labels = self.emit.labels(allowed, &context);
            let (Some(limit), Some(remaining)) = (limit, remaining) else {
                self.emit.remove(allowed, &labels);
                self.emit.remove(used, &labels);
                continue;
            };
            #[allow(clippy::cast_precision_loss)]
            let (limit_value, used_value) = (limit as f64, (limit - remaining) as f64);
            self.emit
                .record(allowed, &labels, Op::Set(limit_value), false)?;
            self.emit
                .record(used, &labels, Op::Set(used_value), false)?;
        }
        Ok(())
    }

    fn tagged(&self, values: &LabelValues) -> Result<(), CallbackError> {
        tag_match_error(&values.tags, &self.settings().custom_tags)
            .map_or(Ok(()), |error| Err(failed(error)))
    }

    pub fn log(&self, line: LogLine) {
        self.logs.borrow_mut().push(line);
    }

    fn times(&self) -> [Option<Moment>; 4] {
        let timing = &self.facts.timing;
        [
            moment(timing.start.as_ref()),
            moment(timing.end.as_ref()),
            moment(timing.api_call_start.as_ref()),
            moment(timing.completion_start.as_ref()),
        ]
    }

    fn latency(&self, values: &LabelValues) -> Result<(), CallbackError> {
        let [_, _, api_call_start_time, completion_start_time] = self.times();
        let (start_time, end_time) = self.facts.latency_window();
        let sequenced = if self.input_sequence_length.is_null() {
            values.clone()
        } else {
            values
                .clone()
                .with("input_sequence_length", self.input_sequence_length.clone())
        };
        let context = sequenced.context(self.settings());
        let plain = values.context(self.settings());
        let ttft = duration(api_call_start_time, completion_start_time)?;
        if let Some(ttft) = ttft
            && self.facts.stream
        {
            self.tagged(values)?;
            self.emit
                .observe("litellm_llm_api_time_to_first_token_metric", &context, ttft)?;
        }
        if let Some(total) = duration(api_call_start_time, end_time)? {
            self.tagged(values)?;
            self.emit
                .observe("litellm_llm_api_latency_metric", &context, total)?;
        }
        let queue = py(self.facts.timing.queue_seconds.as_ref());
        let queue_seconds = if queue.is_null() {
            None
        } else {
            Some(py_number(&queue).ok_or_else(|| not_comparable(">=", &queue))?)
        }
        .filter(|seconds| *seconds >= 0.0);
        if let Some(total) = duration(start_time, end_time)? {
            let observed = queue_seconds.map_or(total, |queue| total + queue);
            self.tagged(values)?;
            self.emit
                .observe("litellm_request_total_latency_metric", &context, observed)?;
        }
        if let Some(queue) = queue_seconds {
            self.tagged(values)?;
            self.emit
                .observe("litellm_request_queue_time_seconds", &plain, queue)?;
        }
        Ok(())
    }

    fn cache(&self, values: &LabelValues) -> Result<(), CallbackError> {
        let context = values.context(self.settings());
        let usage = &self.facts.usage;
        match self.facts.cache_hit {
            None => {}
            Some(true) => {
                self.tagged(values)?;
                self.emit.inc("litellm_cache_hits_metric", &context, 1.0)?;
                let total = py(usage.total_tokens.as_ref());
                let total = py_number(&total)
                    .or_else(|| total.as_bool().map(|flag| if flag { 1.0 } else { 0.0 }))
                    .ok_or_else(|| not_comparable(">", &total))?;
                if total > 0.0 {
                    self.emit
                        .inc("litellm_cached_tokens_metric", &context, total)?;
                }
            }
            Some(false) => {
                self.tagged(values)?;
                self.emit
                    .inc("litellm_cache_misses_metric", &context, 1.0)?;
            }
        }
        let (read, creation) = provider_cache_tokens(usage);
        if read > 0.0 {
            self.tagged(values)?;
            self.emit.inc(
                "litellm_provider_cache_read_input_tokens_metric",
                &context,
                read,
            )?;
        }
        if creation > 0.0 {
            self.tagged(values)?;
            self.emit.inc(
                "litellm_provider_cache_creation_input_tokens_metric",
                &context,
                creation,
            )?;
        }
        Ok(())
    }

    fn media(&self, values: &LabelValues) -> Result<(), CallbackError> {
        let usage = &self.facts.usage;
        let context = values.context(self.settings());
        let counters = [
            (
                "litellm_video_duration_seconds_metric",
                py(usage.duration_seconds.as_ref()),
            ),
            (
                "litellm_images_generated_metric",
                py(usage.output_image_count.as_ref()),
            ),
        ];
        for (name, value) in &counters {
            if let Some(amount) = py_positive_number(value) {
                self.tagged(values)?;
                self.emit.inc(name, &context, amount)?;
            }
        }
        Ok(())
    }

    fn mcp(&self, values: &LabelValues, response_cost: &Value) -> Result<(), CallbackError> {
        let Some(tool) = &self.facts.mcp else {
            return Ok(());
        };
        let mcp = LabelValues::default()
            .with("mcp_tool_name", py(tool.name.as_ref()))
            .with("mcp_server_name", py(tool.server_name.as_ref()))
            .with("hashed_api_key", values.get("hashed_api_key").clone())
            .with("api_key_alias", values.get("api_key_alias").clone())
            .with("team", values.get("team").clone())
            .with("team_alias", values.get("team_alias").clone())
            .with("user", values.get("user").clone())
            .with("end_user", values.get("end_user").clone());
        let context = mcp.context(self.settings());
        self.emit
            .inc("litellm_mcp_tool_calls_total", &context, 1.0)?;
        let cost = py_number(response_cost)
            .or_else(|| response_cost.as_bool().map(f64::from))
            .ok_or_else(|| not_comparable(">", response_cost))?;
        if cost > 0.0 {
            self.emit
                .inc("litellm_mcp_tool_call_spend_metric", &context, cost)?;
        }
        Ok(())
    }

    fn virtual_key(&self, values: &LabelValues) -> Result<(), CallbackError> {
        let Some(key) = self.facts.key()? else {
            return Ok(());
        };
        let remaining = |given: &Value, kind: &str| -> Value {
            if !given.is_null() {
                return given.clone();
            }
            self.v3_header(&format!("x-ratelimit-model_per_key-remaining-{kind}"))
                .map_or_else(|| Value::from(i64::MAX), Value::from)
        };
        let requests = remaining(&key.remaining_requests, "requests");
        let tokens = remaining(&key.remaining_tokens, "tokens");
        let key_values = LabelValues {
            custom_metadata_labels: self.facts.custom_metadata_labels(),
            ..LabelValues::default()
        }
        .with("hashed_api_key", key.hashed)
        .with("api_key_alias", values.get("api_key_alias").clone())
        .with("model", key.model_group)
        .with("model_id", values.get("model_id").clone());
        let context = key_values.context(self.settings());
        for (name, value) in [
            ("litellm_remaining_api_key_requests_for_model", requests),
            ("litellm_remaining_api_key_tokens_for_model", tokens),
        ] {
            let labels = self.emit.labels(name, &context);
            self.emit
                .record_with(name, &labels, || py_float(&value).map(Op::Set), false)?;
        }
        Ok(())
    }

    fn v3_header(&self, name: &str) -> Option<i64> {
        match self.facts.limiter_header(name) {
            Some(Scalar::Int(number)) => i64::try_from(*number).ok(),
            _ => None,
        }
    }

    fn model_info(&self, key: &str) -> Result<Value, CallbackError> {
        let deployment = &self.facts.deployment;
        match &deployment.model_info {
            Some(info) => Ok(py(match key {
                "id" => info.id.as_ref(),
                "tpm" => info.tpm.as_ref(),
                _ => info.rpm.as_ref(),
            })),
            None if rejected(deployment.rejected.as_ref(), "model_info") => {
                Err(failed("'model_info' has no attribute 'get'"))
            }
            None => Ok(Value::Null),
        }
    }

    fn deployment(&self, values: &LabelValues) -> Result<(), CallbackError> {
        let facts = self.facts;
        let deployment = &facts.deployment;
        let api_base = required(&facts.residual.payload_labels, "api_base")?;
        let model = &facts.model;
        let provider = py(deployment.provider.as_ref());
        let model_id = self.model_info("id")?;
        let info_truthy =
            deployment.model_info.is_some() || rejected(deployment.rejected.as_ref(), "model_info");
        if info_truthy || deployment.params_present == Some(true) {
            let deployment_values = LabelValues::default()
                .with("litellm_model_name", model.clone())
                .with("model_id", model_id.clone())
                .with("api_base", api_base.clone())
                .with("api_provider", provider.clone())
                .with("model_group", values.get("model_group").clone());
            let context = deployment_values.context(self.settings());
            for (name, key, fallback) in [
                (
                    "litellm_deployment_tpm_limit",
                    "tpm",
                    py(deployment.tpm.as_ref()),
                ),
                (
                    "litellm_deployment_rpm_limit",
                    "rpm",
                    py(deployment.rpm.as_ref()),
                ),
            ] {
                let own = self.model_info(key)?;
                let limit = if py_truthy(&own) { own } else { fallback };
                if limit.is_null() {
                    continue;
                }
                let labels = self.emit.labels(name, &context);
                self.emit
                    .record_with(name, &labels, || py_float(&limit).map(Op::Set), false)?;
            }
        }
        let state = &facts.rate_limits;
        if state.hidden_params_present != Some(true) {
            return Err(failed("hidden_params is not subscriptable"));
        }
        let remaining_requests = gauge_reading(&py(state.provider_remaining_requests.as_ref()));
        let remaining_tokens = gauge_reading(&py(state.provider_remaining_tokens.as_ref()));
        let context = values.context(self.settings());
        let overhead = py(facts.timing.overhead_ms.as_ref());
        if py_truthy(&overhead) {
            let name = "litellm_overhead_latency_metric";
            let labels = self.emit.labels(name, &context);
            self.emit.record_with(
                name,
                &labels,
                || py_divide(&overhead, 1000.0).map(Op::Observe),
                false,
            )?;
        }
        let guardrail = guardrail_overhead_seconds(facts.guardrails.as_deref())?;
        if !overhead.is_null() || guardrail > 0.0 {
            let base = if py_truthy(&overhead) {
                py_divide(&overhead, 1000.0)?
            } else {
                0.0
            };
            let name = "litellm_overhead_with_guardrails_latency_metric";
            let labels = self.emit.labels(name, &context);
            self.emit
                .record(name, &labels, Op::Observe(base + guardrail), false)?;
        }
        for (name, reading) in [
            ("litellm_remaining_requests_metric", remaining_requests),
            ("litellm_remaining_tokens_metric", remaining_tokens),
        ] {
            if let Some(reading) = reading {
                let labels = self.emit.labels(name, &context);
                self.emit.record(name, &labels, Op::Set(reading), false)?;
            }
        }
        set_deployment_state(
            &self.emit,
            0.0,
            &or_text(model),
            &or_text(&model_id),
            &or_text(api_base),
            &or_text(&provider),
        )?;
        self.emit
            .inc("litellm_deployment_success_responses", &context, 1.0)?;
        self.emit
            .inc("litellm_deployment_total_requests", &context, 1.0)?;
        let [start_time, end_time, _, completion_start_time] = self.times();
        let response = duration(start_time, end_time)?
            .ok_or_else(|| not_subtractable(end_time.is_some(), start_time.is_some()))?;
        let mut latency = response;
        if facts.stream {
            let first = if facts.timing.completion_start_present == Some(true) {
                completion_start_time.ok_or_else(|| not_subtractable(false, true))?
            } else {
                end_time.ok_or_else(|| not_subtractable(false, true))?
            };
            let ttft = duration(start_time, Some(first))?.unwrap_or(0.0);
            if ttft != 0.0 {
                latency = ttft;
            }
        }
        let output_tokens = facts.completion_tokens();
        if output_tokens.is_null() {
            return Ok(());
        }
        let tokens = py_number(&output_tokens)
            .or_else(|| {
                output_tokens
                    .as_bool()
                    .map(|flag| if flag { 1.0 } else { 0.0 })
            })
            .ok_or_else(|| not_comparable(">", &output_tokens))?;
        if tokens > 0.0 {
            let name = "litellm_deployment_latency_per_output_token";
            let labels = self.emit.labels(name, &context);
            self.emit
                .record(name, &labels, Op::Observe(latency / tokens), false)?;
        }
        Ok(())
    }

    fn proxy_total(&self, model_group: &Value, values: &LabelValues) -> Result<(), CallbackError> {
        let requested = match &self.facts.deployment.original_model_group {
            Some(text) if !text.is_empty() => Value::from(text.clone()),
            _ => model_group.clone(),
        };
        let context = values
            .clone()
            .with("requested_model", requested)
            .checked_context(self.settings())
            .map_err(failed)?;
        self.emit
            .inc("litellm_proxy_total_requests_metric", &context, 1.0)
    }
}

fn or_text(value: &Value) -> Value {
    if py_truthy(value) {
        value.clone()
    } else {
        Value::from("")
    }
}

pub fn set_deployment_state(
    emit: &Emitter<'_>,
    state: f64,
    litellm_model_name: &Value,
    model_id: &Value,
    api_base: &Value,
    api_provider: &Value,
) -> Result<(), CallbackError> {
    let values = LabelValues::default()
        .with("litellm_model_name", litellm_model_name.clone())
        .with("model_id", model_id.clone())
        .with("api_base", api_base.clone())
        .with("api_provider", api_provider.clone());
    let name = "litellm_deployment_state";
    let labels = emit.labels(name, &values.context(emit.settings));
    emit.record(name, &labels, Op::Set(state), false)
}

fn py_float_text(text: &str) -> Option<f64> {
    let digits = ascii_decimal_digits(text);
    let trimmed = digits.trim();
    let lowered = trimmed.to_ascii_lowercase();
    let unsigned = lowered.trim_start_matches(['+', '-']);
    if matches!(unsigned, "inf" | "infinity" | "nan") {
        return lowered.parse::<f64>().ok();
    }
    if trimmed.contains('_') {
        let bytes = trimmed.as_bytes();
        let valid = bytes.iter().enumerate().all(|(index, byte)| {
            *byte != b'_'
                || (index > 0
                    && index + 1 < bytes.len()
                    && bytes[index - 1].is_ascii_digit()
                    && bytes[index + 1].is_ascii_digit())
        });
        if !valid {
            return None;
        }
        return trimmed.replace('_', "").parse::<f64>().ok();
    }
    trimmed.parse::<f64>().ok()
}

fn gauge_reading(value: &Value) -> Option<f64> {
    match value {
        _ if is_py_number(value) => py_float_int(value),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        Value::String(text) => py_float_text(text),
        _ => None,
    }
}

fn py_divide(value: &Value, by: f64) -> Result<f64, CallbackError> {
    match value {
        _ if is_py_number(value) => py_number(value).map(|number| number / by),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 } / by),
        _ => None,
    }
    .ok_or_else(|| {
        failed(format!(
            "unsupported operand type(s) for /: '{}' and 'int'",
            py_type_name(value)
        ))
    })
}

fn guardrail_overhead_seconds(entries: Option<&[GuardrailEntry]>) -> Result<f64, CallbackError> {
    let mut total = 0.0;
    for entry in entries.unwrap_or_default() {
        if entry.mapping != Some(true) {
            return Err(failed("guardrail info has no attribute 'get'"));
        }
        let names: Vec<&str> = entry
            .modes
            .iter()
            .flatten()
            .filter_map(|mode| match mode {
                Some(Scalar::Text(name)) => Some(name.as_str()),
                _ => None,
            })
            .collect();
        let additive = !names.is_empty()
            && names
                .iter()
                .all(|name| matches!(*name, "pre_call" | "post_call"));
        if !additive {
            continue;
        }
        let duration = py(entry.duration.as_ref());
        if py_truthy(&duration) {
            total += py_float(&duration)?;
        }
    }
    Ok(total)
}

fn cache_write_tokens(usage: &UsageRecord) -> Option<i64> {
    [&usage.cache_write_tokens, &usage.cache_creation_tokens]
        .into_iter()
        .find_map(|value| match value {
            Some(Scalar::Int(number)) => i64::try_from(*number).ok(),
            _ => None,
        })
}

fn provider_cache_tokens(usage: &UsageRecord) -> (f64, f64) {
    let read_raw = py(usage.cache_read_input_tokens.as_ref());
    let write_raw = py(usage.cache_creation_input_tokens.as_ref());
    let mut read = py_int_or_bool(&read_raw).unwrap_or(0.0);
    let mut creation = py_int_or_bool(&write_raw).unwrap_or(0.0);
    if read_raw.is_null()
        && let Some(cached) = py_int_or_bool(&py(usage.cached_tokens.as_ref()))
    {
        read = cached;
    }
    if write_raw.is_null()
        && let Some(write) = cache_write_tokens(usage)
    {
        #[allow(clippy::cast_precision_loss)]
        {
            creation = write as f64;
        }
    }
    (read, creation)
}
