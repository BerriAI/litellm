use std::{borrow::Cow, sync::Mutex};

use litellm_callbacks_spend_log::SpendLogRow;
use litellm_db::Error;
use litellm_repositories::SpendLogWrite;
use serde_json::{Map, Value, json};

use crate::{LedgerStore, QueuedSpendLog};

#[derive(Debug)]
pub(crate) struct Sqlstate(&'static str);

impl std::fmt::Display for Sqlstate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ERROR: sqlstate {}", self.0)
    }
}

impl std::error::Error for Sqlstate {}

impl sqlx::error::DatabaseError for Sqlstate {
    fn message(&self) -> &str {
        "sqlstate"
    }

    fn code(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(self.0))
    }

    fn as_error(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
        self
    }

    fn as_error_mut(&mut self) -> &mut (dyn std::error::Error + Send + Sync + 'static) {
        self
    }

    fn into_error(self: Box<Self>) -> Box<dyn std::error::Error + Send + Sync + 'static> {
        self
    }

    fn kind(&self) -> sqlx::error::ErrorKind {
        sqlx::error::ErrorKind::Other
    }
}

pub(crate) fn sqlstate(code: &'static str) -> Error {
    Error::Query(sqlx::Error::Database(Box::new(Sqlstate(code))))
}

pub(crate) fn connect_error() -> Error {
    Error::Query(sqlx::Error::Io(
        std::io::ErrorKind::ConnectionRefused.into(),
    ))
}

pub(crate) fn full_pool() -> Error {
    Error::Query(sqlx::Error::PoolTimedOut)
}

/// A whole row as Python's queue holds it, with `fields` (a `, "key": value` list) replacing or
/// adding columns.
pub(crate) fn python_row(request_id: &str, fields: &str) -> String {
    let mut row = match json!({
        "request_id": request_id,
        "call_type": "acompletion",
        "api_key": "hashed",
        "spend": 0.25,
        "total_tokens": 3,
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "startTime": "2026-11-02T12:00:00+00:00",
        "endTime": "2026-11-02T12:00:01+00:00",
        "completionStartTime": null,
        "model": "gpt",
        "model_id": "deployment",
        "model_group": "gpt",
        "mcp_namespaced_tool_name": null,
        "agent_id": null,
        "billing_agent_id": null,
        "api_base": "https://example.test",
        "user": "user",
        "metadata": "{\"status\": \"success\"}",
        "cache_hit": "False",
        "cache_key": "Cache OFF",
        "request_tags": "[]",
        "team_id": "team",
        "organization_id": "",
        "end_user": "",
        "requester_ip_address": null,
        "custom_llm_provider": "openai",
        "messages": "[]",
        "response": "{}",
        "proxy_server_request": "{}",
        "session_id": null,
        "request_duration_ms": 1000,
        "status": "success",
        "litellm_call_id": null,
    }) {
        Value::Object(row) => row,
        _ => Map::new(),
    };
    if let Some(fields) = fields.strip_prefix(',') {
        let Ok(Value::Object(overrides)) =
            litellm_python_compat::json::loads(&format!("{{{fields}}}"))
                .and_then(|value| litellm_python_compat::json::to_json(&value))
        else {
            panic!("fields are a JSON object body: {fields}");
        };
        row.extend(overrides);
    }
    litellm_python_compat::json::dumps(&litellm_python_compat::json::from_json(Value::Object(row)))
        .expect("dumps")
}

pub(crate) fn full_row(request_id: &str) -> SpendLogRow {
    serde_json::from_str(&python_row(request_id, "")).expect("row parses")
}

pub(crate) fn rows(count: usize) -> Vec<QueuedSpendLog> {
    (0..count)
        .map(|index| {
            QueuedSpendLog::from_python(&python_row(&index.to_string(), "")).expect("row is queued")
        })
        .collect()
}

pub(crate) fn charged(request_id: &str) -> QueuedSpendLog {
    QueuedSpendLog::from_python(&python_row(
        request_id,
        &format!(r#", "accounting_delivery_id": "{request_id}-d", "accounting_payload": {{"spend": 1.0}}"#),
    ))
    .expect("row is queued")
}

pub(crate) type Fail = Box<dyn Fn(&[SpendLogWrite], usize) -> Option<Error> + Send + Sync>;

#[derive(Default)]
pub(crate) struct Calls {
    pub(crate) statements: Vec<Vec<String>>,
    pub(crate) written: Vec<String>,
    pub(crate) charges_kept: Vec<String>,
}

pub(crate) struct FakeStore {
    calls: Mutex<Calls>,
    fail: Fail,
}

pub(crate) fn request_ids(writes: &[SpendLogWrite]) -> Vec<String> {
    writes
        .iter()
        .map(|write| match (&write.log, &write.delivery) {
            (Some(log), _) => log.request_id.clone(),
            (None, Some(delivery)) => delivery.request_id.clone(),
            (None, None) => String::new(),
        })
        .collect()
}

pub(crate) fn charges_only(writes: &[SpendLogWrite]) -> bool {
    writes.iter().all(|write| write.log.is_none())
}

impl FakeStore {
    pub(crate) fn new(fail: Fail) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            fail,
        }
    }

    pub(crate) fn writing() -> Self {
        Self::new(Box::new(|_, _| None))
    }

    pub(crate) fn calls(&self) -> std::sync::MutexGuard<'_, Calls> {
        self.calls.lock().expect("calls lock")
    }
}

impl LedgerStore for FakeStore {
    async fn insert(&self, writes: &[SpendLogWrite]) -> Result<usize, Error> {
        let call = {
            let mut calls = self.calls();
            calls.statements.push(request_ids(writes));
            calls.statements.len()
        };
        if let Some(error) = (self.fail)(writes, call) {
            return Err(error);
        }
        let mut calls = self.calls();
        if charges_only(writes) {
            calls.charges_kept.extend(request_ids(writes));
        } else {
            calls.written.extend(request_ids(writes));
        }
        Ok(writes.len())
    }
}
