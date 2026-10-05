use std::collections::BTreeMap;

use serde::{
    Deserialize, Deserializer,
    de::{Error, Unexpected},
};
use serde_json::Value;

use crate::{AgentNode, Span, TraceSummary};

#[cfg(feature = "schema")]
pub mod openapi;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TraceQueryWindow {
    pub start_ms: i64,
    pub end_ms: i64,
    pub as_of_ms: u64,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TraceSortField {
    #[default]
    StartMs,
    DurationMs,
    SpanCount,
    ErrorCount,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TraceSortDirection {
    Asc,
    #[default]
    Desc,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceNoQueryRequest {}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceListRequest {
    #[serde(default)]
    pub as_of_ms: Option<u64>,
    #[serde(default)]
    pub start_ms: Option<i64>,
    #[serde(default)]
    pub end_ms: Option<i64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 1000)))]
    #[serde(deserialize_with = "text::<_, 1000>")]
    pub q: String,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 512)))]
    #[serde(deserialize_with = "cursor")]
    pub cursor: Option<String>,
    #[serde(default = "list_page_size")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 500)))]
    #[serde(deserialize_with = "integer::<_, 1, 500>")]
    pub page_size: u16,
    #[serde(default)]
    pub sort_by: TraceSortField,
    #[serde(default)]
    pub sort_dir: TraceSortDirection,
}

const fn list_page_size() -> u16 {
    50
}

const fn span_page_size() -> u16 {
    100
}

const fn histogram_buckets() -> u16 {
    60
}

const fn value_limit() -> u16 {
    20
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceHistogramRequest {
    #[serde(default)]
    pub start_ms: Option<i64>,
    #[serde(default)]
    pub end_ms: Option<i64>,
    #[serde(default)]
    pub as_of_ms: Option<u64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 1000)))]
    #[serde(deserialize_with = "text::<_, 1000>")]
    pub q: String,
    #[serde(default = "histogram_buckets")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 240)))]
    #[serde(deserialize_with = "integer::<_, 1, 240>")]
    pub buckets: u16,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceValuesRequest {
    #[serde(default)]
    pub start_ms: Option<i64>,
    #[serde(default)]
    pub end_ms: Option<i64>,
    #[serde(default)]
    pub as_of_ms: Option<u64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 1000)))]
    #[serde(deserialize_with = "text::<_, 1000>")]
    pub q: String,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 200)))]
    #[serde(deserialize_with = "text::<_, 200>")]
    pub contains: String,
    #[serde(default = "value_limit")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 100)))]
    #[serde(deserialize_with = "integer::<_, 1, 100>")]
    pub limit: u16,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceSpanPageRequest {
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 512)))]
    #[serde(deserialize_with = "cursor")]
    pub cursor: Option<String>,
    #[serde(default = "span_page_size")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 500)))]
    #[serde(deserialize_with = "integer::<_, 1, 500>")]
    pub page_size: u16,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceErrorPageRequest {
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(length(max = 512)))]
    #[serde(deserialize_with = "cursor")]
    pub cursor: Option<String>,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct TraceMetadata {
    pub summary: TraceSummary,
    pub agents: Vec<AgentNode>,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct TraceSpansPage {
    pub data: Vec<Span>,
    pub next_cursor: Option<String>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum SqlParameter {
    String(String),
    Integer(i64),
    Unsigned(u64),
    Number(f64),
    Boolean(bool),
    Null,
    Strings(Vec<String>),
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceQueryRequest {
    #[cfg_attr(feature = "schema", schemars(length(min = 1)))]
    #[serde(deserialize_with = "sql")]
    pub sql: String,
    #[serde(default)]
    pub params: BTreeMap<String, SqlParameter>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(untagged)]
pub enum UnsignedCount {
    Integer(u64),
    String(String),
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
pub struct TraceSQLColumn {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
pub struct TraceQueryStatistics {
    pub elapsed: f64,
    pub rows_read: UnsignedCount,
    pub bytes_read: UnsignedCount,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
pub struct TraceSQLResponse {
    pub meta: Vec<TraceSQLColumn>,
    #[cfg_attr(feature = "schema", schemars(extend("x-python-normalized" = {"type": "tuple[Mapping[str, JsonValue], ...]"})))]
    pub data: Vec<BTreeMap<String, Value>>,
    pub rows: UnsignedCount,
    pub statistics: TraceQueryStatistics,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TraceProblemCode {
    InvalidRequest,
    Unauthorized,
    Forbidden,
    NotFound,
    TraceChanged,
    TooLarge,
    Unavailable,
    QueryRejected,
    QueryLimitExceeded,
    QueryUnavailable,
    InternalError,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceInvalidParam {
    pub location: String,
    pub reason: String,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TraceProblem {
    #[serde(rename = "type")]
    pub kind: String,
    pub title: String,
    pub status: u16,
    pub detail: String,
    pub code: TraceProblemCode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub database_code: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[cfg_attr(feature = "schema", schemars(extend("default" = [])))]
    pub errors: Vec<TraceInvalidParam>,
}

fn integer<'de, D: Deserializer<'de>, const MIN: u16, const MAX: u16>(
    deserializer: D,
) -> Result<u16, D::Error> {
    let value = u16::deserialize(deserializer)?;
    if (MIN..=MAX).contains(&value) {
        return Ok(value);
    }
    Err(D::Error::invalid_value(
        Unexpected::Unsigned(u64::from(value)),
        &"integer within the documented range",
    ))
}

fn text<'de, D: Deserializer<'de>, const MAX: usize>(deserializer: D) -> Result<String, D::Error> {
    let value = String::deserialize(deserializer)?;
    if value.chars().count() <= MAX {
        return Ok(value);
    }
    Err(D::Error::invalid_length(
        value.chars().count(),
        &"string within the documented length",
    ))
}

fn cursor<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    let value = Option::<String>::deserialize(deserializer)?;
    if value
        .as_ref()
        .is_none_or(|cursor| cursor.chars().count() <= 512)
    {
        return Ok(value);
    }
    Err(D::Error::invalid_length(
        value.as_ref().map_or(0, |cursor| cursor.chars().count()),
        &"cursor within the documented length",
    ))
}

fn sql<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    let value = String::deserialize(deserializer)?;
    if !value.is_empty() {
        return Ok(value);
    }
    Err(D::Error::invalid_value(
        Unexpected::Str(&value),
        &"nonempty SQL",
    ))
}
