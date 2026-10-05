use litellm_storage_clickhouse::Query;
use litellm_traces::{
    QueryScope,
    search::{FieldFilter, RunFilter, RunSearch, SearchKey},
    store::{
        CallQuery, CallRow, CountValue, RunCount, RunCountQuery, RunQuery, RunRow, RunSelection,
        RunSortKey, SpanQuery, SpanRow, SpanSelection, SpanText, SpanTextQuery, TextRange,
    },
};
use serde::{Deserialize, Serialize};

use crate::access::{AccessParams, owned};

/// `LIKE`'s own metacharacters match literally.
fn like_literal(value: &str) -> String {
    value
        .chars()
        .flat_map(|char| match char {
            '\\' | '%' | '_' => vec!['\\', char],
            _ => vec![char],
        })
        .collect()
}

fn contains(value: &str) -> String {
    format!("%{}%", like_literal(value))
}

fn pattern(filter: &FieldFilter) -> String {
    like_literal(&filter.pattern).replace('*', "%")
}

fn mode(filter: &FieldFilter) -> &'static str {
    if filter.exclude { "exclude" } else { "include" }
}

/// Query parameters only carry string arrays, so filters travel as parallel columns.
#[derive(Debug, Default, Serialize)]
struct SearchColumns {
    text: Vec<String>,
    filter_fields: Vec<&'static str>,
    filter_patterns: Vec<String>,
    filter_modes: Vec<&'static str>,
    attribute_keys: Vec<String>,
    attribute_patterns: Vec<String>,
    attribute_modes: Vec<&'static str>,
}

impl From<&RunSearch> for SearchColumns {
    fn from(search: &RunSearch) -> Self {
        let fields: Vec<_> = search
            .filters
            .iter()
            .filter_map(|filter| match &filter.key {
                SearchKey::Field(field) => Some(((*field).into(), filter)),
                SearchKey::Attribute(_) => None,
            })
            .collect();
        let attributes: Vec<_> = search
            .filters
            .iter()
            .filter_map(|filter| match &filter.key {
                SearchKey::Attribute(key) => Some((key.clone(), filter)),
                SearchKey::Field(_) => None,
            })
            .collect();
        Self {
            text: search.text.iter().map(|term| contains(term)).collect(),
            filter_fields: fields.iter().map(|(field, _)| *field).collect(),
            filter_patterns: fields.iter().map(|(_, filter)| pattern(filter)).collect(),
            filter_modes: fields.iter().map(|(_, filter)| mode(filter)).collect(),
            attribute_patterns: attributes
                .iter()
                .map(|(_, filter)| pattern(filter))
                .collect(),
            attribute_modes: attributes.iter().map(|(_, filter)| mode(filter)).collect(),
            attribute_keys: attributes.into_iter().map(|(key, _)| key).collect(),
        }
    }
}

#[derive(Debug, Default, Serialize)]
struct RunsFilter {
    trace_id: String,
    start_ms: i64,
    end_ms: i64,
    #[serde(flatten)]
    search: SearchColumns,
    trace_refs: Vec<String>,
}

impl From<&RunFilter> for RunsFilter {
    fn from(filter: &RunFilter) -> Self {
        Self {
            trace_id: String::new(),
            start_ms: filter.start_ms,
            end_ms: filter.end_ms,
            search: (&filter.search).into(),
            trace_refs: filter.trace_refs.clone(),
        }
    }
}

impl From<&RunSelection> for RunsFilter {
    fn from(selection: &RunSelection) -> Self {
        match selection {
            RunSelection::Matching(filter) => filter.into(),
            RunSelection::TraceId(trace_id) => Self {
                trace_id: trace_id.clone(),
                ..Self::default()
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct RunsParams {
    #[serde(flatten)]
    access: AccessParams,
    #[serde(flatten)]
    filter: RunsFilter,
    sort_key: RunSortKey,
    descending: u8,
    has_cursor: u8,
    cursor_value: i64,
    cursor_ref: String,
    limit: u32,
}

impl RunsParams {
    pub(crate) fn new(access: &QueryScope, query: &RunQuery) -> Self {
        let after = query.after.clone().unwrap_or_default();
        Self {
            access: access.into(),
            filter: (&query.selection).into(),
            sort_key: query.order.key,
            descending: query.order.descending.into(),
            has_cursor: query.after.is_some().into(),
            cursor_value: after.value,
            cursor_ref: after.trace_ref,
            limit: query.limit,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "RunRow")]
struct RunRowEncoding {
    pub trace_id: String,
    pub trace_ref: String,
    pub team_id: String,
    pub api_key_hash: String,
    pub user_id: String,
    pub name: String,
    pub service: String,
    pub input_preview: String,
    #[serde(serialize_with = "litellm_traces::wire::serialize_status")]
    pub status: litellm_traces::SpanStatus,
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
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct RunRowWire(#[serde(with = "RunRowEncoding")] pub RunRow);

macro_rules! over_matching_runs {
    ($($tail:expr),+ $(,)?) => {
        owned!(
            ",\nruns AS (\n",
            include_str!("../../query/matching_runs.sql"),
            ")",
            $($tail),+
        )
    };
}

pub(crate) struct RunsPage;

impl Query for RunsPage {
    type Params = RunsParams;
    type Row = RunRowWire;

    const SQL: &'static str = over_matching_runs!(
        ",\n",
        include_str!("../../query/runs_page.sql"),
        include_str!("../../query/runs.sql")
    );
}

#[derive(Debug, Serialize)]
pub(crate) struct RunCountsParams {
    #[serde(flatten)]
    access: AccessParams,
    #[serde(flatten)]
    filter: RunsFilter,
    buckets: u32,
    by_failed: u8,
    value: &'static str,
    attribute_key: String,
    contains: String,
    limit: u64,
}

impl RunCountsParams {
    pub(crate) fn new(access: &QueryScope, query: &RunCountQuery) -> Self {
        Self {
            access: access.into(),
            filter: (&query.filter).into(),
            buckets: query.by.buckets.unwrap_or(0),
            by_failed: query.by.failed.into(),
            value: match &query.by.value {
                None => "",
                Some(CountValue::PrimaryAgent) => "primary_agent",
                Some(CountValue::Field(field)) => (*field).into(),
                Some(CountValue::AttributeKey | CountValue::Attribute(_)) => "attribute",
            },
            attribute_key: match &query.by.value {
                Some(CountValue::Attribute(key)) => key.clone(),
                _ => String::new(),
            },
            contains: contains(&query.contains),
            limit: query.limit.map_or(u64::MAX, u64::from),
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "RunCount")]
struct RunCountEncoding {
    #[serde(deserialize_with = "super::number::deserialize")]
    pub bucket: u32,
    #[serde(
        deserialize_with = "super::number::boolean",
        serialize_with = "litellm_traces::wire::serialize_flag"
    )]
    pub failed: bool,
    pub value: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub runs: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct RunCountRow(#[serde(with = "RunCountEncoding")] pub RunCount);

impl RunCountsParams {
    pub(crate) fn counts_attributes(&self) -> bool {
        self.value == "attribute"
    }
}

pub(crate) struct RunCounts;

impl Query for RunCounts {
    type Params = RunCountsParams;
    type Row = RunCountRow;

    const SQL: &'static str = over_matching_runs!("\n", include_str!("../../query/run_counts.sql"));
}

pub(crate) struct RunAttributeCounts;

impl Query for RunAttributeCounts {
    type Params = RunCountsParams;
    type Row = RunCountRow;

    const SQL: &'static str =
        over_matching_runs!("\n", include_str!("../../query/run_attribute_counts.sql"));
}

#[derive(Debug, Default, Serialize)]
struct SpanKeyset {
    as_of_ms: u64,
    after_team: String,
    after_key: String,
    after_trace: String,
    after_span: String,
    limit: u32,
}

impl From<&SpanQuery> for SpanKeyset {
    fn from(query: &SpanQuery) -> Self {
        let after = query.after.clone().unwrap_or_default();
        Self {
            as_of_ms: query.as_of_ms,
            after_team: after.team_id,
            after_key: after.api_key_hash,
            after_trace: after.trace_id,
            after_span: after.span_id,
            limit: query.limit,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct TraceSpansParams {
    #[serde(flatten)]
    access: AccessParams,
    trace_id: String,
    trace_ref: String,
    #[serde(flatten)]
    keyset: SpanKeyset,
}

#[derive(Debug, Serialize)]
pub(crate) struct RunSpansParams {
    #[serde(flatten)]
    access: AccessParams,
    trace_ids: Vec<String>,
    trace_refs: Vec<String>,
    start_ms: i64,
    end_ms: i64,
    #[serde(flatten)]
    keyset: SpanKeyset,
}

pub(crate) enum SpansParams {
    Trace(TraceSpansParams),
    Runs(RunSpansParams),
}

impl SpansParams {
    pub(crate) fn new(access: &QueryScope, query: &SpanQuery) -> Self {
        let keyset = query.into();
        match &query.selection {
            SpanSelection::Trace {
                trace_id,
                trace_ref,
            } => Self::Trace(TraceSpansParams {
                access: access.into(),
                trace_id: trace_id.clone(),
                trace_ref: trace_ref.clone(),
                keyset,
            }),
            SpanSelection::Runs {
                trace_ids,
                trace_refs,
                window,
            } => Self::Runs(RunSpansParams {
                access: access.into(),
                trace_ids: trace_ids.clone(),
                trace_refs: trace_refs.clone(),
                start_ms: window.start,
                end_ms: window.end,
                keyset,
            }),
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "SpanRow")]
struct SpanRowEncoding {
    #[serde(default)]
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: litellm_traces::ObservationType,
    #[serde(
        default,
        deserialize_with = "super::number::boolean",
        serialize_with = "litellm_traces::wire::serialize_flag"
    )]
    pub wrapper_candidate: bool,
    pub agent: String,
    #[serde(default)]
    pub framework: String,
    #[serde(serialize_with = "litellm_traces::wire::serialize_status")]
    pub status: litellm_traces::SpanStatus,
    pub status_message: String,
    #[serde(
        deserialize_with = "super::number::boolean",
        serialize_with = "litellm_traces::wire::serialize_flag"
    )]
    pub error_truncated: bool,
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
    #[serde(default)]
    pub call_keys: Vec<litellm_traces::CallKey>,
    #[serde(
        default,
        deserialize_with = "litellm_traces::wire::evidence",
        serialize_with = "litellm_traces::wire::serialize_evidence"
    )]
    pub call_evidence: Option<litellm_traces::CallEvidenceKind>,
    #[serde(default)]
    pub tool_call_id: String,
    pub team_id: String,
    pub api_key_hash: String,
    pub user_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct SpanRowWire(#[serde(with = "SpanRowEncoding")] pub SpanRow);

macro_rules! span_page {
    ($selection:literal) => {
        owned!(
            "\nSELECT * FROM (\n",
            include_str!("../../query/span_columns.sql"),
            include_str!(concat!("../../query/", $selection)),
            include_str!("../../query/span_page.sql"),
        )
    };
}

pub(crate) struct TraceSpans;

impl Query for TraceSpans {
    type Params = TraceSpansParams;
    type Row = SpanRowWire;

    const SQL: &'static str = span_page!("trace_spans.sql");
}

pub(crate) struct RunSpans;

impl Query for RunSpans {
    type Params = RunSpansParams;
    type Row = SpanRowWire;

    const SQL: &'static str = span_page!("run_spans.sql");
}

#[derive(Debug, Serialize)]
pub(crate) struct SpanTextParams {
    #[serde(flatten)]
    access: AccessParams,
    trace_id: String,
    trace_ref: String,
    span_ids: Vec<String>,
    part: &'static str,
    range: &'static str,
    offset: u64,
    bounded: u8,
    chars: u64,
    needle: String,
}

impl SpanTextParams {
    pub(crate) fn new(access: &QueryScope, query: &SpanTextQuery) -> Self {
        let (range, offset, chars) = match query.range {
            TextRange::From { offset, max_chars } => ("from", offset, max_chars),
            TextRange::Last { chars } => ("last", 0, Some(chars)),
        };
        Self {
            access: access.into(),
            trace_id: query.trace_id.clone(),
            trace_ref: query.trace_ref.clone(),
            span_ids: query.span_ids.clone(),
            part: query.part.into(),
            range,
            offset,
            bounded: chars.is_some().into(),
            chars: chars.unwrap_or(0),
            needle: query.contains.clone().unwrap_or_default(),
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "SpanText")]
struct SpanTextEncoding {
    pub span_id: String,
    pub text: String,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub total_chars: u64,
    pub version: String,
    #[serde(
        deserialize_with = "super::number::boolean",
        serialize_with = "litellm_traces::wire::serialize_flag"
    )]
    pub contains: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct SpanTextRow(#[serde(with = "SpanTextEncoding")] pub SpanText);

pub(crate) struct SpanTexts;

impl Query for SpanTexts {
    type Params = SpanTextParams;
    type Row = SpanTextRow;

    const SQL: &'static str = owned!("\n", include_str!("../../query/span_text.sql"));
}

#[derive(Debug, Serialize)]
pub(crate) struct CallsParams {
    #[serde(flatten)]
    access: AccessParams,
    start_ms: i64,
    end_ms: i64,
    response_ids: Vec<String>,
    request_ids: Vec<String>,
    trace_ids: Vec<String>,
    has_cursor: u8,
    after_team: String,
    after_ms: i64,
    after_id: String,
    limit: u32,
}

impl CallsParams {
    pub(crate) fn new(access: &QueryScope, query: &CallQuery) -> Self {
        let after = query.after.clone().unwrap_or_default();
        Self {
            access: access.into(),
            start_ms: query.window.start,
            end_ms: query.window.end,
            response_ids: query.response_ids.clone(),
            request_ids: query.request_ids.clone(),
            trace_ids: query.trace_ids.clone(),
            has_cursor: query.after.is_some().into(),
            after_team: after.team_id,
            after_ms: after.start_ms,
            after_id: after.request_id,
            limit: query.limit,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(remote = "CallRow")]
struct CallRowEncoding {
    pub request_id: String,
    pub litellm_call_id: String,
    pub response_id: String,
    pub upstream_response_id: String,
    pub trace_id: String,
    pub span_id: String,
    pub team_id: String,
    pub api_key: String,
    pub user: String,
    #[serde(deserialize_with = "super::number::optional_finite")]
    pub spend: Option<f64>,
    #[serde(deserialize_with = "super::number::deserialize")]
    pub start_ms: i64,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct CallRowWire(#[serde(with = "CallRowEncoding")] pub CallRow);

pub(crate) struct Calls;

impl Query for Calls {
    type Params = CallsParams;
    type Row = CallRowWire;

    const SQL: &'static str = owned!("\n", include_str!("../../query/calls.sql"));
}

#[cfg(test)]
mod tests {
    use litellm_traces::search::RunField;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest]
    #[case::plain("plan", "%plan%")]
    #[case::like_metacharacters("50%_off\\", "%50\\%\\_off\\\\%")]
    fn text_terms_match_literally_anywhere(#[case] term: &str, #[case] pattern: &str) {
        let columns = SearchColumns::from(&RunSearch {
            text: vec![term.into()],
            filters: Vec::new(),
        });
        assert_eq!(columns.text, [pattern]);
    }

    #[rstest]
    #[case::glob("res*er", "res%er")]
    #[case::glob_escapes_the_rest("gpt_4*", "gpt\\_4%")]
    #[case::exact("plan trip", "plan trip")]
    fn field_patterns_turn_globs_into_like(#[case] pattern: &str, #[case] like: &str) {
        let columns = SearchColumns::from(&RunSearch {
            text: Vec::new(),
            filters: vec![
                FieldFilter {
                    key: SearchKey::Field(RunField::Model),
                    pattern: pattern.into(),
                    exclude: true,
                },
                FieldFilter {
                    key: SearchKey::Attribute("tenant.tier".into()),
                    pattern: "x".into(),
                    exclude: false,
                },
            ],
        });
        assert_eq!(
            (
                columns.filter_fields,
                columns.filter_patterns,
                columns.filter_modes
            ),
            (vec!["model"], vec![like.to_owned()], vec!["exclude"])
        );
        assert_eq!(
            (
                columns.attribute_keys,
                columns.attribute_patterns,
                columns.attribute_modes
            ),
            (
                vec!["tenant.tier".to_owned()],
                vec!["x".to_owned()],
                vec!["include"]
            )
        );
    }

    fn decoded<T: serde::de::DeserializeOwned + Serialize>(wire: Value, quoted: bool) -> Value {
        let encoded = Value::Object(
            wire.as_object()
                .unwrap()
                .iter()
                .map(|(name, value)| {
                    let encoded = if quoted && value.is_number() {
                        json!(value.to_string())
                    } else {
                        value.clone()
                    };
                    (name.clone(), encoded)
                })
                .collect(),
        );
        serde_json::to_value(serde_json::from_value::<T>(encoded).unwrap()).unwrap()
    }

    #[rstest]
    #[case::unquoted(false)]
    #[case::quoted(true)]
    fn rows_decode_clickhouse_numbers(#[case] quoted: bool) {
        let run = json!({"trace_id": "trace", "trace_ref": "ref", "team_id": "team", "api_key_hash": "key", "user_id": "user", "name": "agent", "service": "service", "input_preview": "input", "status": "STATUS_CODE_OK", "start_ms": -1, "duration_ms": 20, "span_count": u64::MAX, "agent_count": 1, "agent_invocations": 2, "agent_names": ["agent"], "frameworks": ["claude-agent-sdk"], "llm_calls": 3, "tool_calls": 4, "input_tokens": 5, "output_tokens": 6, "models": ["model"], "error_count": 0});
        assert_eq!(decoded::<RunRowWire>(run.clone(), quoted), run);
        let span = json!({"trace_id": "trace", "span_id": "span", "parent_span_id": "parent", "name": "agent", "type": "agent", "wrapper_candidate": 1, "agent": "agent", "framework": "claude-agent-sdk", "status": "STATUS_CODE_ERROR", "status_message": "error", "error_truncated": 1, "start_ns": -1, "duration_ns": u64::MAX, "service": "service", "input_preview": "input", "model": "model", "input_tokens": u32::MAX, "output_tokens": 6, "litellm_request_id": "request", "call_keys": ["provider_response:request"], "call_evidence": "complete", "tool_call_id": "call", "team_id": "team", "api_key_hash": "key", "user_id": "user"});
        assert_eq!(decoded::<SpanRowWire>(span.clone(), quoted), span);
        let count = json!({"bucket": 2, "failed": 1, "value": "v", "runs": u64::MAX});
        assert_eq!(decoded::<RunCountRow>(count.clone(), quoted), count);
        let text = json!({"span_id": "span", "text": "error", "total_chars": u64::MAX, "version": "version", "contains": 1});
        assert_eq!(decoded::<SpanTextRow>(text.clone(), quoted), text);
        let call = json!({"request_id": "request", "litellm_call_id": "gateway", "response_id": "response", "upstream_response_id": "upstream", "trace_id": "trace", "span_id": "span", "team_id": "team", "api_key": "key", "user": "user", "spend": 0.125, "start_ms": -1});
        assert_eq!(decoded::<CallRowWire>(call.clone(), quoted), call);
    }

    #[rstest]
    #[case::unknown(json!(null), None)]
    #[case::free(json!(0), Some(0.0))]
    #[case::paid(json!("0.125"), Some(0.125))]
    fn spend_rows_preserve_unknown_and_known_cost(
        #[case] cost: Value,
        #[case] expected: Option<f64>,
    ) {
        let row: CallRowWire = serde_json::from_value(json!({
            "request_id": "request", "litellm_call_id": "gateway", "response_id": "response", "upstream_response_id": "",
            "trace_id": "trace", "span_id": "span", "team_id": "team", "api_key": "key",
            "user": "user", "spend": cost, "start_ms": 0
        }))
        .unwrap();
        assert_eq!(row.0.spend, expected);
    }

    #[rstest]
    #[case::nan(json!("NaN"))]
    #[case::infinity(json!("1e999"))]
    #[case::boolean(json!(true))]
    fn spend_rows_reject_invalid_cost(#[case] cost: Value) {
        let row = serde_json::from_value::<CallRowWire>(json!({
            "request_id": "request", "litellm_call_id": "gateway", "response_id": "response", "upstream_response_id": "",
            "trace_id": "trace", "span_id": "span", "team_id": "team", "api_key": "key",
            "user": "user", "spend": cost, "start_ms": 0
        }));
        assert!(row.is_err());
    }
}
