use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "schema", schemars(rename = "TraceScope"))]
pub struct ReadAccessParams {
    #[serde(
        deserialize_with = "crate::wire::flag",
        serialize_with = "crate::wire::serialize_flag"
    )]
    #[cfg_attr(feature = "schema", schemars(schema_with = "crate::schema::flag"))]
    pub all_teams: bool,
    pub user_id: String,
    pub team_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ListTracesParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub start_ms: i64,
    pub end_ms: i64,
    pub cursor_ms: i64,
    pub cursor_trace_id: String,
    pub limit: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ListTracesRow {
    pub trace_id: String,
    pub trace_ref: String,
    pub team_id: String,
    pub api_key_hash: String,
    pub user_id: String,
    pub name: String,
    pub service: String,
    pub input_preview: String,
    #[serde(serialize_with = "crate::wire::serialize_status")]
    pub status: crate::SpanStatus,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TraceSpansParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub trace_id: String,
    pub trace_ref: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TraceSpansRow {
    #[serde(default)]
    pub trace_id: String,
    #[serde(default)]
    pub original_trace_id: String,
    pub span_id: String,
    pub parent_span_id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: crate::ObservationType,
    #[serde(
        default,
        deserialize_with = "crate::wire::flag",
        serialize_with = "crate::wire::serialize_flag"
    )]
    pub wrapper_candidate: bool,
    pub agent: String,
    #[serde(default)]
    pub framework: String,
    #[serde(serialize_with = "crate::wire::serialize_status")]
    pub status: crate::SpanStatus,
    pub status_message: String,
    #[serde(
        deserialize_with = "crate::wire::flag",
        serialize_with = "crate::wire::serialize_flag"
    )]
    pub error_truncated: bool,
    pub start_ns: i64,
    pub duration_ns: u64,
    pub service: String,
    pub input_preview: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub litellm_request_id: String,
    #[serde(default)]
    pub call_keys: Vec<crate::CallKey>,
    #[serde(
        default,
        deserialize_with = "crate::wire::evidence",
        serialize_with = "crate::wire::serialize_evidence"
    )]
    pub call_evidence: Option<crate::CallEvidenceKind>,
    #[serde(default)]
    pub tool_call_id: String,
    #[serde(default)]
    pub source_type: String,
    #[serde(default)]
    pub source_url: String,
    #[serde(default)]
    pub source_title: String,
    #[serde(default)]
    pub source_user: String,
    pub team_id: String,
    pub api_key_hash: String,
    pub user_id: String,
}

impl TraceSpansRow {
    pub(crate) fn transport_trace_id(&self) -> &str {
        if self.original_trace_id.is_empty() {
            &self.trace_id
        } else {
            &self.original_trace_id
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TracePageSpansParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub trace_refs: Vec<String>,
    pub start_ms: i64,
    pub end_ms: i64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpanDetailParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub trace_id: String,
    pub trace_ref: String,
    pub span_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpanDetailRow {
    pub span_id: String,
    pub input: String,
    pub output: String,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpanErrorParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub trace_id: String,
    pub trace_ref: String,
    pub span_id: String,
    pub error_offset: u64,
    pub error_version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpanErrorRow {
    pub span_id: String,
    pub message: String,
    pub total_chars: u64,
    pub version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpendByResponseIdsParams {
    #[serde(flatten)]
    pub access: ReadAccessParams,
    pub response_ids: Vec<String>,
    pub provider_request_ids: Vec<String>,
    pub request_ids: Vec<String>,
    pub trace_ids: Vec<String>,
    pub start_ms: i64,
    pub end_ms: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpendByResponseIdsRow {
    pub request_id: String,
    pub litellm_call_id: String,
    pub response_id: String,
    pub upstream_response_id: String,
    #[serde(default)]
    pub provider_request_id: String,
    pub trace_id: String,
    pub span_id: String,
    pub team_id: String,
    pub api_key: String,
    pub user: String,
    pub spend: Option<f64>,
    pub start_ms: i64,
}

impl SpendByResponseIdsRow {
    pub(crate) fn identity(&self) -> (&str, i64, &str) {
        (&self.team_id, self.start_ms, &self.request_id)
    }
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
