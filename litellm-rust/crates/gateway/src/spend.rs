//! The spend writer: an observer that records every request in the proxy's Postgres tables, as
//! the proxy's own spend writer does, so the proxy's dashboards, spend reports and budget checks
//! see the gateway's traffic.
//!
//! Requests are added up in memory and written in batches: spend logs every couple of seconds,
//! and the key, user, team, team membership, end user and daily activity rows every
//! `proxy_batch_write_at` seconds. Every write adds to what is stored, so replicas never
//! overwrite each other.

use std::{
    collections::HashMap,
    mem,
    sync::{Arc, Mutex, Weak},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures_util::future::BoxFuture;
use litellm_config::Config;
use litellm_db::{
    DailySpend, DailyTable, EntitySpend, PgPool, SpendLog, TeamMemberSpend, add_end_user_spend,
    add_key_spend, add_team_member_spend, add_team_spend, add_user_spend, insert_spend_logs,
    upsert_daily_spend,
};
use litellm_gateway_hooks::{Facts, Observer, Record, Route, Usage};
use serde_json::{Map, Value, json};

/// How often spend logs are written, as the proxy's spend-log queue monitor polls.
const LOG_INTERVAL: Duration = Duration::from_secs(2);
/// Rows per spend-log statement, the proxy's `SPEND_LOG_WRITE_BATCH_MAX_ROWS`.
const LOG_BATCH: usize = 100;
/// Rows per daily-table statement.
const DAILY_BATCH: usize = 100;
/// Spend logs held while the database is unreachable; older ones are dropped past this.
const MAX_PENDING_LOGS: usize = 10_000;
/// Retries of a write that failed on a deadlock, a lock timeout or a lost connection.
const RETRIES: u32 = 3;

/// The keys of the proxy's spend-log `metadata`, every one present and null when unset.
const METADATA_KEYS: [&str; 49] = [
    "actor_agent_id",
    "additional_usage_values",
    "agent_execution_mode",
    "applied_guardrails",
    "attempted_fallbacks",
    "attempted_retries",
    "autorouter_baseline_observation",
    "autorouter_savings",
    "autorouter_savings_estimate",
    "azure_spillover",
    "batch_failed_requests",
    "batch_models",
    "batch_successful_requests",
    "billing_agent_id",
    "cold_storage_object_key",
    "compression_savings",
    "cost_breakdown",
    "error_information",
    "eval_information",
    "guardrail_information",
    "internal_call_origin",
    "litellm_call_id",
    "litellm_gateway_injected_cache",
    "litellm_overhead_time_ms",
    "litellm_roi_estimator",
    "max_retries",
    "mcp_tool_call_metadata",
    "model_map_information",
    "original_model_group",
    "proxy_server_request",
    "requester_ip_address",
    "router_metadata",
    "routing_decision",
    "spend_logs_metadata",
    "status",
    "target_agent_id",
    "usage_object",
    "used_client_oauth_token",
    "user_agent",
    "user_api_key",
    "user_api_key_alias",
    "user_api_key_org_id",
    "user_api_key_project_alias",
    "user_api_key_project_id",
    "user_api_key_team_alias",
    "user_api_key_team_id",
    "user_api_key_user_id",
    "vector_store_request_metadata",
    "verified_human_user_id",
];

/// The annotation hooks set to add to a request's `spend_logs_metadata`.
pub const SPEND_LOGS_METADATA: &str = "spend_logs_metadata";

#[derive(Clone, Debug, PartialEq)]
pub struct SpendSettings {
    /// `general_settings.proxy_batch_write_at`, plus the proxy's 0-5 s of start-up jitter.
    pub interval: Duration,
    /// `general_settings.disable_spend_logs`: everything but the spend-log rows is written.
    pub disable_spend_logs: bool,
    /// `litellm_settings.disable_end_user_cost_tracking`.
    pub disable_end_user_tracking: bool,
}

impl SpendSettings {
    /// The writer's settings, or `None` when `general_settings.disable_spend_updates` turns
    /// spend tracking off.
    pub fn from_config(config: &Config) -> Option<Self> {
        let general = &config.general_settings.additional_fields;
        if flag(general.get("disable_spend_updates")) {
            return None;
        }
        let seconds = match general.get("proxy_batch_write_at") {
            Some(litellm_config::Value::Number(seconds)) => seconds.as_f64(),
            Some(litellm_config::Value::String(seconds)) => seconds.trim().parse().ok(),
            _ => std::env::var("PROXY_BATCH_WRITE_AT")
                .ok()
                .and_then(|seconds| seconds.trim().parse().ok()),
        }
        .filter(|seconds: &f64| seconds.is_finite() && *seconds > 0.0)
        .unwrap_or(10.0);
        let jitter = f64::from(rand::random::<u8>() % 6);
        Some(Self {
            interval: Duration::from_secs_f64(seconds + jitter),
            disable_spend_logs: flag(general.get("disable_spend_logs")),
            disable_end_user_tracking: flag(
                config
                    .litellm_settings
                    .additional_fields
                    .get("disable_end_user_cost_tracking"),
            ),
        })
    }
}

fn flag(value: Option<&litellm_config::Value>) -> bool {
    match value {
        Some(litellm_config::Value::Bool(flag)) => *flag,
        Some(litellm_config::Value::String(text)) => text.eq_ignore_ascii_case("true"),
        _ => false,
    }
}

type DailyKey = (DailyTable, String, String, String, String, String, String);

/// Spend added up since the last write.
#[derive(Default)]
struct Pending {
    keys: HashMap<String, f64>,
    users: HashMap<String, f64>,
    teams: HashMap<String, f64>,
    end_users: HashMap<String, f64>,
    members: HashMap<(String, String), f64>,
    daily: HashMap<DailyKey, DailySpend>,
}

impl Pending {
    fn add_daily(&mut self, table: DailyTable, row: DailySpend) {
        let key = (
            table,
            row.entity_id.clone(),
            row.date.clone(),
            row.api_key.clone(),
            row.model.clone(),
            row.custom_llm_provider.clone(),
            row.endpoint.clone(),
        );
        match self.daily.get_mut(&key) {
            Some(total) => {
                total.prompt_tokens += row.prompt_tokens;
                total.completion_tokens += row.completion_tokens;
                total.api_requests += row.api_requests;
                total.successful_requests += row.successful_requests;
                total.failed_requests += row.failed_requests;
                total.cache_read_input_tokens += row.cache_read_input_tokens;
                total.cache_creation_input_tokens += row.cache_creation_input_tokens;
                total.total_response_time_ms += row.total_response_time_ms;
                total.timed_requests += row.timed_requests;
                total.spend += row.spend;
            }
            None => {
                self.daily.insert(key, row);
            }
        }
    }
}

pub struct SpendWriter {
    pool: PgPool,
    settings: SpendSettings,
    pending: Mutex<Pending>,
    logs: Mutex<Vec<SpendLog>>,
    /// Serializes flushes, so the periodic one and the one at shutdown never interleave.
    flushing: tokio::sync::Mutex<()>,
}

impl SpendWriter {
    /// Starts the writer and its timers. Must be called inside a tokio runtime.
    pub fn start(pool: PgPool, settings: SpendSettings) -> Arc<Self> {
        let writer = Arc::new(Self {
            pool,
            settings,
            pending: Mutex::default(),
            logs: Mutex::default(),
            flushing: tokio::sync::Mutex::new(()),
        });
        tokio::spawn(run(Arc::downgrade(&writer)));
        writer
    }

    fn add(&self, record: &Record) {
        let request = &record.request;
        let key = record.key();
        let cost = record.outcome.cost.unwrap_or(0.0);
        let end_user = request
            .end_user
            .as_deref()
            .filter(|_| !self.settings.disable_end_user_tracking);
        let hash = key.map(|key| key.hash.as_str()).unwrap_or_default();
        let user_id = key.and_then(|key| key.user_id.as_deref());
        let team_id = key.and_then(|key| key.team_id.as_deref());
        if hash.is_empty() && user_id.is_none() && team_id.is_none() && end_user.is_none() {
            return;
        }
        let log = spend_log(record, cost, end_user);
        {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            if !hash.is_empty() {
                *pending.keys.entry(hash.to_owned()).or_default() += cost;
            }
            if let Some(user_id) = user_id {
                *pending.users.entry(user_id.to_owned()).or_default() += cost;
            }
            if let Some(team_id) = team_id {
                *pending.teams.entry(team_id.to_owned()).or_default() += cost;
                if let Some(user_id) = user_id {
                    *pending
                        .members
                        .entry((team_id.to_owned(), user_id.to_owned()))
                        .or_default() += cost;
                }
            }
            if let Some(end_user) = end_user {
                *pending.end_users.entry(end_user.to_owned()).or_default() += cost;
            }
            let row = daily_row(record, &log);
            pending.add_daily(
                DailyTable::User,
                DailySpend {
                    entity_id: log.user.clone(),
                    ..row.clone()
                },
            );
            pending.add_daily(
                DailyTable::Team,
                DailySpend {
                    entity_id: log.team_id.clone(),
                    ..row.clone()
                },
            );
            if let Some(end_user) = end_user {
                pending.add_daily(
                    DailyTable::EndUser,
                    DailySpend {
                        entity_id: end_user.to_owned(),
                        ..row
                    },
                );
            }
        }
        if !self.settings.disable_spend_logs {
            let mut logs = self.logs.lock().unwrap_or_else(|e| e.into_inner());
            logs.push(log);
            if logs.len() > MAX_PENDING_LOGS {
                let excess = logs.len() - MAX_PENDING_LOGS;
                logs.drain(..excess);
                tracing::warn!(dropped = excess, "spend log queue full; dropped the oldest");
            }
        }
    }

    /// Writes the spend logs held so far.
    pub async fn flush_logs(&self) {
        let _flushing = self.flushing.lock().await;
        self.write_logs().await;
    }

    /// Writes everything held so far.
    pub async fn flush_all(&self) {
        let _flushing = self.flushing.lock().await;
        self.write_logs().await;
        self.write_spend().await;
    }

    async fn write_logs(&self) {
        let logs = mem::take(&mut *self.logs.lock().unwrap_or_else(|e| e.into_inner()));
        let mut failed = Vec::new();
        for batch in logs.chunks(LOG_BATCH) {
            if let Err(error) = retry(|| insert_spend_logs(&self.pool, batch)).await {
                tracing::error!(%error, rows = batch.len(), "writing spend logs failed");
                if error.is_retryable() {
                    failed.extend_from_slice(batch);
                }
            }
        }
        if !failed.is_empty() {
            let mut logs = self.logs.lock().unwrap_or_else(|e| e.into_inner());
            failed.append(&mut logs);
            *logs = failed;
        }
    }

    async fn write_spend(&self) {
        let pending = mem::take(&mut *self.pending.lock().unwrap_or_else(|e| e.into_inner()));
        let entity = |spend: HashMap<String, f64>| spend.into_iter().collect::<EntitySpend>();
        // The proxy's order: user, end user, key, team, team membership, then the daily tables.
        // Entity spend that cannot be written is dropped, as the proxy drops it.
        let users = entity(pending.users);
        log_failure("user", retry(|| add_user_spend(&self.pool, &users)).await);
        let end_users = entity(pending.end_users);
        log_failure(
            "end user",
            retry(|| add_end_user_spend(&self.pool, &end_users)).await,
        );
        let keys = entity(pending.keys);
        log_failure("key", retry(|| add_key_spend(&self.pool, &keys)).await);
        let teams = entity(pending.teams);
        log_failure("team", retry(|| add_team_spend(&self.pool, &teams)).await);
        let members: Vec<_> = pending
            .members
            .into_iter()
            .map(|((team_id, user_id), cost)| TeamMemberSpend {
                team_id,
                user_id,
                cost,
            })
            .collect();
        log_failure(
            "team member",
            retry(|| add_team_member_spend(&self.pool, &members)).await,
        );

        let mut tables: HashMap<DailyTable, Vec<DailySpend>> = HashMap::new();
        for ((table, ..), row) in pending.daily {
            tables.entry(table).or_default().push(row);
        }
        let mut requeue = Vec::new();
        for table in [DailyTable::User, DailyTable::Team, DailyTable::EndUser] {
            let rows = tables.remove(&table).unwrap_or_default();
            for batch in rows.chunks(DAILY_BATCH) {
                if let Err(error) = retry(|| upsert_daily_spend(&self.pool, table, batch)).await {
                    tracing::error!(%error, ?table, rows = batch.len(), "writing daily spend failed");
                    if error.is_retryable() {
                        requeue.extend(batch.iter().cloned().map(|row| (table, row)));
                    }
                }
            }
        }
        if !requeue.is_empty() {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            for (table, row) in requeue {
                pending.add_daily(table, row);
            }
        }
    }
}

impl Observer for SpendWriter {
    fn name(&self) -> &'static str {
        "spend_writer"
    }

    fn reads(&self) -> Facts {
        Facts::KEY
            | Facts::END_USER
            | Facts::CLIENT_HEADERS
            | Facts::DEPLOYMENT
            | Facts::USAGE
            | Facts::COST
            | Facts::TIMINGS
            | Facts::ANNOTATIONS
    }

    fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()> {
        self.add(record);
        Box::pin(async {})
    }

    fn flush(&self) -> BoxFuture<'_, ()> {
        Box::pin(self.flush_all())
    }
}

/// Writes spend logs every [`LOG_INTERVAL`] and everything else every batch interval, until the
/// writer is dropped.
async fn run(writer: Weak<SpendWriter>) {
    let interval = match writer.upgrade() {
        Some(writer) => writer.settings.interval,
        None => return,
    };
    let mut since_spend = Duration::ZERO;
    loop {
        tokio::time::sleep(LOG_INTERVAL).await;
        let Some(writer) = writer.upgrade() else {
            return;
        };
        since_spend += LOG_INTERVAL;
        if since_spend >= interval {
            since_spend = Duration::ZERO;
            writer.flush_all().await;
        } else {
            writer.flush_logs().await;
        }
    }
}

async fn retry<F, Fut>(mut write: F) -> Result<(), litellm_db::Error>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<(), litellm_db::Error>>,
{
    let mut attempt = 0;
    loop {
        match write().await {
            Err(error) if error.is_retryable() && attempt < RETRIES => {
                let base = 1u64 << (attempt + 1);
                let jitter = u64::from(rand::random::<u16>()) * base / u64::from(u16::MAX);
                tokio::time::sleep(Duration::from_millis((base * 1000 + jitter * 1000) / 2)).await;
                attempt += 1;
            }
            result => return result,
        }
    }
}

fn log_failure(table: &str, result: Result<(), litellm_db::Error>) {
    if let Err(error) = result {
        tracing::error!(%error, table, "writing spend failed; the spend is dropped");
    }
}

fn status(record: &Record) -> &'static str {
    match (200..300).contains(&record.outcome.status) {
        true => "success",
        false => "failure",
    }
}

/// The model as the proxy logs it: the deployment's model without its provider route.
pub(crate) fn logged_model(record: &Record) -> String {
    match &record.outcome.deployment {
        Some(deployment) => match (deployment.provider(), deployment.model.split_once('/')) {
            (Some(provider), Some((prefix, rest))) if prefix == provider => rest.to_owned(),
            _ => deployment.model.clone(),
        },
        None => record.request.requested_model.to_string(),
    }
}

fn spend_log(record: &Record, cost: f64, end_user: Option<&str>) -> SpendLog {
    let request = &record.request;
    let outcome = &record.outcome;
    let key = record.key();
    let deployment = outcome.deployment.as_ref();
    let usage = outcome.usage.unwrap_or_default();
    let duration = record.duration();
    let completion_start = match (request.stream, outcome.time_to_first_byte) {
        (true, Some(first_byte)) => request.started_at + first_byte,
        _ => outcome.ended_at,
    };
    let tokens = |count: u64| i32::try_from(count).unwrap_or(i32::MAX);
    SpendLog {
        request_id: request.request_id.to_string(),
        call_type: request.route.call_type().into(),
        api_key: key.map(|key| key.hash.clone()).unwrap_or_default(),
        spend: cost,
        total_tokens: tokens(usage.total_tokens()),
        prompt_tokens: tokens(usage.prompt_tokens()),
        completion_tokens: tokens(usage.output_tokens),
        start_time: timestamp(request.started_at),
        end_time: timestamp(outcome.ended_at),
        completion_start_time: Some(timestamp(completion_start)),
        request_duration_ms: Some(i32::try_from(duration.as_millis()).unwrap_or(i32::MAX)),
        model: logged_model(record),
        model_id: deployment.map(|d| d.id.to_string()).unwrap_or_default(),
        model_group: deployment
            .map(|d| d.model_group.to_string())
            .unwrap_or_else(|| request.requested_model.to_string()),
        custom_llm_provider: deployment
            .and_then(|d| d.provider())
            .unwrap_or_default()
            .to_owned(),
        api_base: deployment
            .and_then(|d| d.api_base.clone())
            .unwrap_or_default(),
        user: key.and_then(|key| key.user_id.clone()).unwrap_or_default(),
        metadata: metadata(record),
        cache_hit: if request.stream { "False" } else { "None" }.into(),
        cache_key: "Cache OFF".into(),
        team_id: key.and_then(|key| key.team_id.clone()).unwrap_or_default(),
        organization_id: key.and_then(|key| key.org_id.clone()).unwrap_or_default(),
        end_user: end_user.unwrap_or_default().to_owned(),
        requester_ip_address: None,
        session_id: Some(request.request_id.to_string()),
        status: status(record).into(),
        litellm_call_id: Some(request.request_id.to_string()),
    }
}

fn daily_row(record: &Record, log: &SpendLog) -> DailySpend {
    let usage = record.outcome.usage.unwrap_or_default();
    let success = status(record) == "success";
    let count = |count: u64| i64::try_from(count).unwrap_or(i64::MAX);
    DailySpend {
        entity_id: String::new(),
        date: date(record.request.started_at),
        api_key: log.api_key.clone(),
        model: log.model.clone(),
        custom_llm_provider: log.custom_llm_provider.clone(),
        // The proxy maps only chat completions to an endpoint here.
        endpoint: match record.request.route {
            Route::ChatCompletions => "/chat/completions".into(),
            Route::Messages => String::new(),
        },
        model_group: Some(log.model_group.clone()),
        prompt_tokens: count(usage.prompt_tokens()),
        completion_tokens: count(usage.output_tokens),
        api_requests: 1,
        successful_requests: i64::from(success),
        failed_requests: i64::from(!success),
        cache_read_input_tokens: count(usage.cache_read_tokens),
        cache_creation_input_tokens: count(usage.cache_write_tokens),
        total_response_time_ms: match success {
            true => i64::from(log.request_duration_ms.unwrap_or(0)),
            false => 0,
        },
        timed_requests: i64::from(success),
        spend: log.spend,
    }
}

/// The spend log's `metadata`, with every key the proxy writes.
fn metadata(record: &Record) -> Value {
    let mut metadata: Map<String, Value> = METADATA_KEYS
        .iter()
        .map(|key| ((*key).to_owned(), Value::Null))
        .collect();
    let mut set = |key: &str, value: Value| {
        metadata.insert(key.to_owned(), value);
    };
    let text =
        |value: Option<&String>| value.map_or(Value::Null, |value| Value::from(value.as_str()));
    let request = &record.request;
    if let Some(key) = record.key() {
        set("user_api_key", Value::from(key.hash.as_str()));
        set("user_api_key_alias", text(key.alias.as_ref()));
        set("user_api_key_team_id", text(key.team_id.as_ref()));
        set("user_api_key_user_id", text(key.user_id.as_ref()));
        set("user_api_key_team_alias", text(key.team_alias.as_ref()));
        set("user_api_key_org_id", text(key.org_id.as_ref()));
    }
    set("spend_logs_metadata", spend_logs_metadata(record));
    set(
        "user_agent",
        request
            .client_headers
            .get(http::header::USER_AGENT)
            .and_then(|value| value.to_str().ok())
            .map_or(Value::Null, Value::from),
    );
    set("litellm_call_id", Value::from(&*request.request_id));
    set("litellm_roi_estimator", Value::Bool(false));
    if let Some(usage) = record.outcome.usage {
        let (usage_object, additional) = usage_values(&usage);
        set("usage_object", usage_object);
        set("additional_usage_values", additional);
    }
    if status(record) == "failure" {
        set("status", Value::from("failure"));
        set(
            "error_information",
            json!({
                "error_code": record.outcome.status.to_string(),
                "error_class": "",
                "llm_provider": record
                    .outcome
                    .deployment
                    .as_ref()
                    .and_then(|deployment| deployment.provider())
                    .unwrap_or_default(),
                "traceback": "",
                "error_message": "",
            }),
        );
    }
    Value::Object(metadata)
}

/// `spend_logs_metadata` as the proxy merges it: the team's, then the key's, then what hooks
/// attached for the request, later keys winning.
pub(crate) fn spend_logs_metadata(record: &Record) -> Value {
    let mut merged = Map::new();
    if let Some(key) = record.key() {
        for metadata in [&key.team_metadata, &key.metadata] {
            if let Some(Value::Object(fields)) = metadata.get("spend_logs_metadata") {
                merged.extend(fields.clone());
            }
        }
    }
    if let Some(Value::Object(fields)) = record.annotations.get(SPEND_LOGS_METADATA) {
        merged.extend(fields.clone());
    }
    match merged.is_empty() {
        true => Value::Null,
        false => Value::Object(merged),
    }
}

/// The proxy's usage object for a response, and its `additional_usage_values`: everything but
/// the three totals.
pub(crate) fn usage_values(usage: &Usage) -> (Value, Value) {
    let mut additional = Map::new();
    additional.insert(
        "prompt_tokens_details".into(),
        json!({
            "cached_tokens": usage.cache_read_tokens,
            "cache_creation_tokens": usage.cache_write_tokens,
        }),
    );
    additional.insert(
        "cache_read_input_tokens".into(),
        Value::from(usage.cache_read_tokens),
    );
    additional.insert(
        "cache_creation_input_tokens".into(),
        Value::from(usage.cache_write_tokens),
    );
    if usage.web_search_requests > 0 {
        additional.insert(
            "server_tool_use".into(),
            json!({"web_search_requests": usage.web_search_requests}),
        );
    }
    let mut object = additional.clone();
    object.insert("prompt_tokens".into(), Value::from(usage.prompt_tokens()));
    object.insert("completion_tokens".into(), Value::from(usage.output_tokens));
    object.insert("total_tokens".into(), Value::from(usage.total_tokens()));
    (Value::Object(object), Value::Object(additional))
}

/// Days since 1970-01-01 as a UTC calendar date.
pub(crate) fn civil(days: i64) -> (i64, u32, u32) {
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let day_of_era = days.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_index + 2) / 5 + 1) as u32;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn since_epoch(time: SystemTime) -> Duration {
    time.duration_since(UNIX_EPOCH).unwrap_or_default()
}

/// `YYYY-MM-DD`, UTC.
fn date(time: SystemTime) -> String {
    let (year, month, day) = civil((since_epoch(time).as_secs() / 86_400) as i64);
    format!("{year:04}-{month:02}-{day:02}")
}

/// `YYYY-MM-DD HH:MM:SS.fff`, UTC, as the proxy stores `timestamp(3)` columns.
fn timestamp(time: SystemTime) -> String {
    let elapsed = since_epoch(time);
    let seconds = elapsed.as_secs() % 86_400;
    format!(
        "{} {:02}:{:02}:{:02}.{:03}",
        date(time),
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60,
        elapsed.subsec_millis()
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::epoch(0, "1970-01-01 00:00:00.000")]
    #[case::leap_day(1_709_208_000_123, "2024-02-29 12:00:00.123")]
    #[case::year_end(1_798_761_599_999, "2026-12-31 23:59:59.999")]
    fn timestamps_are_utc(#[case] millis: u64, #[case] expected: &str) {
        let time = UNIX_EPOCH + Duration::from_millis(millis);
        assert_eq!(timestamp(time), expected);
        assert_eq!(date(time), expected[..10]);
    }
}
