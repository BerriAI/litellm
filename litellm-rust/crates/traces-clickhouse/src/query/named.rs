use litellm_storage_clickhouse::Query;
use litellm_traces::query::named as contracts;
use serde::{Deserialize, Serialize};

pub use contracts::ReadAccessParams;

#[derive(Deserialize, Serialize)]
#[serde(remote = "contracts::ListTracesParams")]
struct ListTracesParamsEncoding {
    #[serde(flatten)]
    pub access: contracts::ReadAccessParams,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub start_ms: i64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub end_ms: i64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub cursor_ms: i64,
    pub cursor_trace_id: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub limit: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ListTracesParams(
    #[serde(with = "ListTracesParamsEncoding")] pub contracts::ListTracesParams,
);

impl From<contracts::ListTracesParams> for ListTracesParams {
    fn from(value: contracts::ListTracesParams) -> Self {
        Self(value)
    }
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "contracts::ListTracesRow")]
struct ListTracesRowEncoding {
    pub trace_id: String,
    pub trace_ref: String,
    pub team_id: String,
    pub api_key_hash: String,
    pub user_id: String,
    pub name: String,
    pub service: String,
    pub input_preview: String,
    pub status: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub start_ms: i64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub duration_ms: i64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub span_count: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub agent_count: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub agent_invocations: u64,
    #[serde(default)]
    pub agent_names: Vec<String>,
    #[serde(default)]
    pub frameworks: Vec<String>,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub llm_calls: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub tool_calls: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub input_tokens: u64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub output_tokens: u64,
    pub models: Vec<String>,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub error_count: u64,
    pub request_ids: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ListTracesRow(#[serde(with = "ListTracesRowEncoding")] pub contracts::ListTracesRow);

pub use contracts::TraceSpansParams;

#[derive(Deserialize, Serialize)]
#[serde(remote = "contracts::TraceSpansRow")]
struct TraceSpansRowEncoding {
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
    #[serde(deserialize_with = "super::number::deserialize")]
    pub error_truncated: u8,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub start_ns: i64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub duration_ns: u64,
    pub service: String,
    pub input_preview: String,
    pub model: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub input_tokens: u32,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub output_tokens: u32,
    pub litellm_request_id: String,
    pub team_id: String,
    pub api_key_hash: String,
    pub user_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TraceSpansRow(#[serde(with = "TraceSpansRowEncoding")] pub contracts::TraceSpansRow);

pub use contracts::SpanDetailParams;

pub use contracts::SpanDetailRow;

#[derive(Deserialize, Serialize)]
#[serde(remote = "contracts::SpanErrorParams")]
struct SpanErrorParamsEncoding {
    #[serde(flatten)]
    pub access: contracts::ReadAccessParams,
    pub trace_id: String,
    pub trace_ref: String,
    pub span_id: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub error_offset: u64,
    pub error_version: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpanErrorParams(
    #[serde(with = "SpanErrorParamsEncoding")] pub contracts::SpanErrorParams,
);

impl From<contracts::SpanErrorParams> for SpanErrorParams {
    fn from(value: contracts::SpanErrorParams) -> Self {
        Self(value)
    }
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "contracts::SpanErrorRow")]
struct SpanErrorRowEncoding {
    pub span_id: String,
    pub message: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub total_chars: u64,
    pub version: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpanErrorRow(#[serde(with = "SpanErrorRowEncoding")] pub contracts::SpanErrorRow);

#[derive(Deserialize, Serialize)]
#[serde(remote = "contracts::SpendByResponseIdsParams")]
struct SpendByResponseIdsParamsEncoding {
    #[serde(flatten)]
    pub access: contracts::ReadAccessParams,
    pub response_ids: Vec<String>,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub start_ms: i64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub end_ms: i64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpendByResponseIdsParams(
    #[serde(with = "SpendByResponseIdsParamsEncoding")] pub contracts::SpendByResponseIdsParams,
);

impl From<contracts::SpendByResponseIdsParams> for SpendByResponseIdsParams {
    fn from(value: contracts::SpendByResponseIdsParams) -> Self {
        Self(value)
    }
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "contracts::SpendByResponseIdsRow")]
struct SpendByResponseIdsRowEncoding {
    pub request_id: String,
    pub response_id: String,
    pub team_id: String,
    pub api_key: String,
    pub user: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub spend: f64,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub start_ms: i64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SpendByResponseIdsRow(
    #[serde(with = "SpendByResponseIdsRowEncoding")] pub contracts::SpendByResponseIdsRow,
);

pub struct ListTraces;

impl Query for ListTraces {
    type Params = ListTracesParams;
    type Row = ListTracesRow;

    const SQL: &'static str = include_str!("../../query/list_traces.sql");
}

pub struct TraceSpans;

impl Query for TraceSpans {
    type Params = TraceSpansParams;
    type Row = TraceSpansRow;

    const SQL: &'static str = include_str!("../../query/trace_spans.sql");
}

pub struct SpanDetail;

impl Query for SpanDetail {
    type Params = SpanDetailParams;
    type Row = SpanDetailRow;

    const SQL: &'static str = include_str!("../../query/span_detail.sql");
}

pub struct SpanError;

impl Query for SpanError {
    type Params = SpanErrorParams;
    type Row = SpanErrorRow;

    const SQL: &'static str = include_str!("../../query/span_error.sql");
}

pub struct SpendByResponseIds;

impl Query for SpendByResponseIds {
    type Params = SpendByResponseIdsParams;
    type Row = SpendByResponseIdsRow;

    const SQL: &'static str = include_str!("../../query/spend_by_response_ids.sql");
}

pub use contracts::{TraceIdentityParams, TraceIdentityRow};

pub struct TraceIdentity;

impl Query for TraceIdentity {
    type Params = TraceIdentityParams;
    type Row = TraceIdentityRow;
    const SQL: &'static str = include_str!("../../query/trace_identity.sql");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::{Value, json};

    fn round_trip<T: serde::de::DeserializeOwned + Serialize>(wire: Value, quoted: bool) {
        let encoded = Value::Object(
            wire.as_object()
                .unwrap()
                .iter()
                .map(|(name, value)| {
                    let encoded = if quoted && value.is_number() && name != "all_teams" {
                        json!(value.to_string())
                    } else {
                        value.clone()
                    };
                    (name.clone(), encoded)
                })
                .collect(),
        );
        let decoded: T = serde_json::from_value(encoded).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), wire);
    }

    #[rstest]
    #[case::unquoted(false)]
    #[case::quoted(true)]
    fn rows_decode_into_neutral_contracts(#[case] quoted: bool) {
        round_trip::<ListTracesRow>(
            json!({"trace_id": "trace", "trace_ref": "ref", "team_id": "team", "api_key_hash": "key", "user_id": "user", "name": "agent", "service": "service", "input_preview": "input", "status": "ok", "start_ms": -1, "duration_ms": 20, "span_count": u64::MAX, "agent_count": 1, "agent_invocations": 2, "agent_names": ["agent"], "frameworks": ["claude-agent-sdk"], "llm_calls": 3, "tool_calls": 4, "input_tokens": 5, "output_tokens": 6, "models": ["model"], "error_count": 0, "request_ids": ["request"]}),
            quoted,
        );
        round_trip::<TraceSpansRow>(
            json!({"span_id": "span", "parent_span_id": "parent", "name": "agent", "type": "agent", "agent": "agent", "framework": "claude-agent-sdk", "status": "error", "status_message": "error", "error_truncated": 1, "start_ns": -1, "duration_ns": u64::MAX, "service": "service", "input_preview": "input", "model": "model", "input_tokens": u32::MAX, "output_tokens": 6, "litellm_request_id": "request", "team_id": "team", "api_key_hash": "key", "user_id": "user"}),
            quoted,
        );
        round_trip::<SpanDetailRow>(
            json!({"span_id": "span", "input": "input", "output": "output", "attributes": {"count": "42"}}),
            quoted,
        );
        round_trip::<SpanErrorRow>(
            json!({"span_id": "span", "message": "error", "total_chars": u64::MAX, "version": "version"}),
            quoted,
        );
        round_trip::<SpendByResponseIdsRow>(
            json!({"request_id": "request", "response_id": "response", "team_id": "team", "api_key": "key", "user": "user", "spend": 0.125, "start_ms": -1}),
            quoted,
        );
    }

    #[rstest]
    #[case::unquoted(false)]
    #[case::quoted(true)]
    fn parameters_preserve_flattened_multi_team_access(#[case] quoted: bool) {
        round_trip::<ListTracesParams>(
            json!({"all_teams": 0, "user_id": "user", "team_ids": ["team-a", "team-b"], "start_ms": -1, "end_ms": 10, "cursor_ms": 0, "cursor_trace_id": "", "limit": u32::MAX}),
            quoted,
        );
        round_trip::<SpanErrorParams>(
            json!({"all_teams": 0, "user_id": "", "team_ids": [], "trace_id": "trace", "trace_ref": "ref", "span_id": "span", "error_offset": u64::MAX, "error_version": "version"}),
            quoted,
        );
        round_trip::<SpendByResponseIdsParams>(
            json!({"all_teams": 0, "user_id": "user", "team_ids": ["team-a", "team-b"], "response_ids": ["response"], "start_ms": -1, "end_ms": 10}),
            quoted,
        );
    }
}
