use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize, Serialize)]
pub struct ReadAccessParams {
    pub all_teams: u8,
    pub user_id: String,
    pub team_ids: Vec<String>,
    pub api_key_hash: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ListTracesParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub start_ms: i64,
    pub end_ms: i64,
    pub cursor_ms: i64,
    pub cursor_trace_id: String,
    pub limit: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ListTracesRow {
    pub trace_id: String,
    pub trace_ref: String,
    pub team_id: String,
    pub api_key_hash: String,
    pub user_id: String,
    pub name: String,
    pub service: String,
    pub input_preview: String,
    pub status: String,
    pub start_ms: i64,
    pub duration_ms: i64,
    pub span_count: u64,
    pub agent_count: u64,
    pub agent_invocations: u64,
    #[serde(default)]
    pub agent_names: Vec<String>,
    #[serde(default)]
    pub frameworks: Vec<String>,
    pub llm_calls: u64,
    pub tool_calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub models: Vec<String>,
    pub error_count: u64,
    pub request_ids: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TraceSpansParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub trace_id: String,
    pub trace_ref: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TraceSpansRow {
    pub span_id: String,
    pub parent_span_id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub agent: String,
    #[serde(default)]
    pub framework: String,
    pub status: String,
    pub status_message: String,
    pub error_truncated: u8,
    pub start_ns: i64,
    pub duration_ns: u64,
    pub service: String,
    pub input_preview: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub litellm_request_id: String,
    pub team_id: String,
    pub api_key_hash: String,
    pub user_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpanDetailParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub trace_id: String,
    pub trace_ref: String,
    pub span_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpanDetailRow {
    pub span_id: String,
    pub input: String,
    pub output: String,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpanErrorParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub trace_id: String,
    pub trace_ref: String,
    pub span_id: String,
    pub error_offset: u64,
    pub error_version: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpanErrorRow {
    pub span_id: String,
    pub message: String,
    pub total_chars: u64,
    pub version: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpendByResponseIdsParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub response_ids: Vec<String>,
    pub start_ms: i64,
    pub end_ms: i64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpendByResponseIdsRow {
    pub request_id: String,
    pub response_id: String,
    pub team_id: String,
    pub api_key: String,
    pub user: String,
    pub spend: f64,
    pub start_ms: i64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TraceIdentityParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub trace_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TraceIdentityRow {
    pub trace_ref: String,
}
