//! Typed payloads for the LiteLLM `/v1/callbacks/logs` realtime-logging contract.
//!
//! Field names below are the EXACT JSON keys the Python replay path + spend-logs
//! builder read. Note the deliberate mix:
//!   - `startTime` / `endTime` are camelCase (epoch f64 seconds)
//!   - `response_cost` / `prompt_tokens` / etc. are snake_case
//!
//! Mirrors Python `litellm/integrations/` + the proxy `CallbackLogsRequest`
//! contract 1:1.

use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

/// Cumulative token usage for a realtime session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Cost-attribution metadata threaded from the authenticated request.
#[derive(Clone, Debug, Default)]
pub struct RequestMetadata {
    pub user_api_key_hash: Option<String>,
    pub user_api_key_user_id: Option<String>,
    pub user_api_key_team_id: Option<String>,
}

/// The self-describing payload. Field names are the EXACT JSON keys the Python
/// replay path + spend-logs builder read.
#[derive(Clone, Debug, Serialize)]
pub struct StandardLoggingPayload {
    pub id: String,
    pub litellm_call_id: String,

    /// e.g. "realtime", "acompletion". Falls back to "acompletion" if absent.
    pub call_type: String,

    pub model: String,
    pub custom_llm_provider: String,

    /// Spend ($) written to LiteLLM_SpendLogs.spend.
    pub response_cost: f64,

    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,

    /// EPOCH SECONDS as float — camelCase keys, NOT snake_case.
    #[serde(rename = "startTime")]
    pub start_time: f64,
    #[serde(rename = "endTime")]
    pub end_time: f64,

    pub stream: bool,

    pub metadata: StandardLoggingMetadata,

    /// Optional; stored as request input on the spend log row.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Value>,
}

/// Cost-attribution keys. The replayer maps these into litellm_params.metadata,
/// which the spend-logs builder reads to set user / team_id / organization_id.
#[derive(Clone, Debug, Serialize, Default)]
pub struct StandardLoggingMetadata {
    pub user_api_key_hash: Option<String>, // -> SpendLogs.api_key
    pub user_api_key_user_id: Option<String>, // -> SpendLogs.user
    pub user_api_key_team_id: Option<String>, // -> SpendLogs.team_id

    // Optional but read by the builder; include when known:
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_api_key_alias: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_api_key_org_id: Option<String>, // -> SpendLogs.organization_id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_api_key_end_user_id: Option<String>, // -> SpendLogs.end_user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spend_logs_metadata: Option<HashMap<String, Value>>,
}
