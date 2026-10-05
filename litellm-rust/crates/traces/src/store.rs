//! What trace storage must answer, independent of the engine behind it.

use std::{cmp::Ordering, ops::Range};

use serde::{Deserialize, Serialize};

use crate::search::{RunField, RunFilter};

#[derive(Clone, Debug, PartialEq)]
pub enum RunSelection {
    Matching(RunFilter),
    /// Every run with this trace id, whenever it happened.
    TraceId(String),
}

/// The last row of a page in its order: the row's sort value and its reference.
#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunCursor {
    pub value: i64,
    pub trace_ref: String,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RunSortKey {
    #[default]
    StartMs,
    DurationMs,
    SpanCount,
    ErrorCount,
    TraceRef,
}

/// Runs by `key`, ties broken by `trace_ref` in the same direction.
#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RunOrder {
    pub key: RunSortKey,
    pub descending: bool,
}

impl RunOrder {
    pub const NEWEST: Self = Self {
        key: RunSortKey::StartMs,
        descending: true,
    };
    pub const BY_REFERENCE: Self = Self {
        key: RunSortKey::TraceRef,
        descending: false,
    };

    pub fn value(self, row: &RunRow) -> i64 {
        let count = |count: u64| i64::try_from(count).unwrap_or(i64::MAX);
        match self.key {
            RunSortKey::StartMs => row.start_ms,
            RunSortKey::DurationMs => row.duration_ms,
            RunSortKey::SpanCount => count(row.span_count),
            RunSortKey::ErrorCount => count(row.error_count),
            RunSortKey::TraceRef => 0,
        }
    }

    pub fn compare(self, left: &RunRow, right: &RunRow) -> Ordering {
        let ascending =
            (self.value(left), &left.trace_ref).cmp(&(self.value(right), &right.trace_ref));
        if self.descending {
            ascending.reverse()
        } else {
            ascending
        }
    }

    pub fn cursor(self, row: &RunRow) -> RunCursor {
        RunCursor {
            value: self.value(row),
            trace_ref: row.trace_ref.clone(),
        }
    }
}

impl Default for RunOrder {
    fn default() -> Self {
        Self::NEWEST
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunQuery {
    pub selection: RunSelection,
    pub order: RunOrder,
    pub after: Option<RunCursor>,
    pub limit: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunRow {
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
}

/// What a run is counted under.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CountValue {
    Field(RunField),
    /// The run's alphabetically first agent label, or its service when it has none.
    PrimaryAgent,
    /// Keys of the span and resource attributes its spans carry.
    AttributeKey,
    /// Values of one attribute across its spans.
    Attribute(String),
}

/// Each dimension left unset collapses to one group: bucket 0, not failed, or an empty value.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CountBy {
    /// Equal-width slices of the filter window; run `i` lands in
    /// `(start_ms - window.start) * buckets / window.len()`.
    pub buckets: Option<u32>,
    pub failed: bool,
    pub value: Option<CountValue>,
}

/// Matching runs per group, most runs first. A run with several values for a field, such as
/// several models, counts once under each; empty values are not counted.
#[derive(Clone, Debug, PartialEq)]
pub struct RunCountQuery {
    pub filter: RunFilter,
    pub by: CountBy,
    /// Keeps values containing this text, ignoring case.
    pub contains: String,
    /// `None` returns every group.
    pub limit: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct RunCount {
    pub bucket: u32,
    #[serde(
        deserialize_with = "crate::wire::flag",
        serialize_with = "crate::wire::serialize_flag"
    )]
    pub failed: bool,
    pub value: String,
    pub runs: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpanSelection {
    Trace {
        trace_id: String,
        trace_ref: String,
    },
    /// Spans of several runs that started within `window`. `trace_ids` are those runs' trace ids,
    /// which narrow the read before references are checked.
    Runs {
        trace_ids: Vec<String>,
        trace_refs: Vec<String>,
        window: Range<i64>,
    },
}

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct SpanCursor {
    pub team_id: String,
    pub api_key_hash: String,
    pub trace_id: String,
    pub span_id: String,
}

/// One row per span, the earliest received copy when a span was exported twice, ordered by
/// [`SpanCursor`] ascending. Spans received after `as_of_ms` are left out.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanQuery {
    pub selection: SpanSelection,
    pub as_of_ms: u64,
    pub after: Option<SpanCursor>,
    pub limit: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpanRow {
    #[serde(default)]
    pub trace_id: String,
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
    /// The first 128 characters; [`SpanPart::Error`] reads the rest.
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
    pub team_id: String,
    pub api_key_hash: String,
    pub user_id: String,
}

impl SpanRow {
    pub fn cursor(&self) -> SpanCursor {
        SpanCursor {
            team_id: self.team_id.clone(),
            api_key_hash: self.api_key_hash.clone(),
            trace_id: self.trace_id.clone(),
            span_id: self.span_id.clone(),
        }
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    Deserialize,
    Serialize,
    strum::EnumString,
    strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum SpanPart {
    Input,
    Output,
    Error,
    /// The span's attributes as one JSON object of strings.
    Attributes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextRange {
    /// Up to `max_chars` characters from `offset`; `None` reads to the end.
    From { offset: u64, max_chars: Option<u64> },
    /// The last `chars` characters.
    Last { chars: u64 },
}

impl TextRange {
    pub const ALL: Self = Self::From {
        offset: 0,
        max_chars: None,
    };
}

/// One part of each listed span of one run, read from the copy [`SpanQuery`] would return.
/// Spans that are not visible to the reader, or do not exist, are left out.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanTextQuery {
    pub trace_id: String,
    pub trace_ref: String,
    pub span_ids: Vec<String>,
    pub part: SpanPart,
    pub range: TextRange,
    /// Reports whether the whole part contains this text, case-sensitive.
    pub contains: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SpanText {
    pub span_id: String,
    pub text: String,
    pub total_chars: u64,
    /// Uppercase hex SHA-256 of the whole part, so a reader can tell when it changed.
    pub version: String,
    pub contains: bool,
}

/// Gateway calls that can be priced against spans: those whose response id, call id or trace id
/// is listed, oldest first by `(team_id, start_ms, request_id)`.
#[derive(Clone, Debug, PartialEq)]
pub struct CallQuery {
    pub window: Range<i64>,
    /// Also matches the upstream id a managed `resp_` id wraps.
    pub response_ids: Vec<String>,
    /// Matches the gateway call id, or the request id when a call has none.
    pub request_ids: Vec<String>,
    pub trace_ids: Vec<String>,
    pub after: Option<CallCursor>,
    pub limit: u32,
}

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct CallCursor {
    pub team_id: String,
    pub start_ms: i64,
    pub request_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CallRow {
    pub request_id: String,
    pub litellm_call_id: String,
    pub response_id: String,
    pub upstream_response_id: String,
    pub trace_id: String,
    pub span_id: String,
    pub team_id: String,
    pub api_key: String,
    pub user: String,
    pub spend: Option<f64>,
    pub start_ms: i64,
}

impl CallRow {
    pub fn cursor(&self) -> CallCursor {
        CallCursor {
            team_id: self.team_id.clone(),
            start_ms: self.start_ms,
            request_id: self.request_id.clone(),
        }
    }

    pub(crate) fn identity(&self) -> (&str, i64, &str) {
        (&self.team_id, self.start_ms, &self.request_id)
    }
}
