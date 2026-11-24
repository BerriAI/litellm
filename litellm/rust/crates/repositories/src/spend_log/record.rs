//! The rows the spend log repository writes: a `LiteLLM_SpendLogs` request log, and the
//! `LiteLLM_SpendLogDelivery` row that carries its accounting work to the ledger worker.

use chrono::NaiveDateTime;
use serde::{Serialize, Serializer};
use serde_json::{Map, Value, value::RawValue};

fn timestamp_text<S: Serializer>(value: &NaiveDateTime, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&value.format("%Y-%m-%dT%H:%M:%S%.f"))
}

fn optional_timestamp_text<S: Serializer>(
    value: &Option<NaiveDateTime>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => timestamp_text(value, serializer),
        None => serializer.serialize_none(),
    }
}

/// A `jsonb` column's JSON text, sent to Postgres as written, so an integer past 64 bits keeps
/// every digit as it does in the Python insert's `safe_dumps` text.
#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
pub struct JsonText(Box<RawValue>);

impl JsonText {
    /// `text` as a column value, or the error when it is not one JSON value.
    pub fn new(text: String) -> Result<Self, serde_json::Error> {
        RawValue::from_string(text).map(Self)
    }

    /// The column's JSON text.
    pub fn get(&self) -> &str {
        self.0.get()
    }
}

impl PartialEq for JsonText {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

/// One `LiteLLM_SpendLogs` row with every column set, serialized under the column names
/// `jsonb_populate_recordset` reads. Timestamps are UTC; a `None` is stored as `NULL`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpendLogRecord {
    pub request_id: String,
    pub call_type: String,
    pub api_key: String,
    pub spend: f64,
    pub total_tokens: i32,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    #[serde(rename = "startTime", serialize_with = "timestamp_text")]
    pub start_time: NaiveDateTime,
    #[serde(rename = "endTime", serialize_with = "timestamp_text")]
    pub end_time: NaiveDateTime,
    pub request_duration_ms: Option<i32>,
    #[serde(
        rename = "completionStartTime",
        serialize_with = "optional_timestamp_text"
    )]
    pub completion_start_time: Option<NaiveDateTime>,
    pub model: String,
    pub model_id: Option<String>,
    pub model_group: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub api_base: Option<String>,
    pub user: Option<String>,
    pub metadata: Option<JsonText>,
    pub cache_hit: Option<String>,
    pub cache_key: Option<String>,
    pub request_tags: Option<JsonText>,
    pub team_id: Option<String>,
    pub organization_id: Option<String>,
    pub end_user: Option<String>,
    pub requester_ip_address: Option<String>,
    pub messages: Option<JsonText>,
    pub response: Option<JsonText>,
    pub session_id: Option<String>,
    pub status: Option<String>,
    pub mcp_namespaced_tool_name: Option<String>,
    pub billing_agent_id: Option<String>,
    pub agent_id: Option<String>,
    pub proxy_server_request: Option<JsonText>,
    pub litellm_call_id: Option<String>,
    #[serde(serialize_with = "timestamp_text")]
    pub created_at: NaiveDateTime,
    #[serde(serialize_with = "timestamp_text")]
    pub updated_at: NaiveDateTime,
}

/// The accounting work one request leaves for the ledger worker. `accounting_payload` is the
/// worker's own JSON contract and is stored as given; `is_batch` marks a batch or video job that
/// settles once under its request id.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpendLogDelivery {
    pub delivery_id: String,
    pub request_id: String,
    pub accounting_payload: Map<String, Value>,
    pub is_batch: bool,
}

/// What one request writes: its request log, its ledger delivery, or both. A log with no
/// delivery carries no charge and goes in on its own; a delivery with no log charges a request
/// whose log is not kept.
#[derive(Clone, Debug, PartialEq)]
pub struct SpendLogWrite {
    pub log: Option<SpendLogRecord>,
    pub delivery: Option<SpendLogDelivery>,
}
