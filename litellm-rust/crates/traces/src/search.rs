use serde::{Deserialize, Serialize};

use crate::store::RunCount;

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
    Deserialize,
    Serialize,
    strum::EnumIter,
    strum::EnumString,
    strum::IntoStaticStr,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case", ascii_case_insensitive)]
pub enum RunField {
    Name,
    Agent,
    RootStatus,
    HasError,
    Model,
    Input,
    TraceId,
    Service,
    Team,
}

/// What a `key:value` filter matches: a run field, or `attr.<key>`, a span or resource
/// attribute that any span of the run carries.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum SearchKey {
    Field(RunField),
    Attribute(String),
}

const ATTRIBUTE_PREFIX: &str = "attr.";

impl SearchKey {
    pub fn parse(key: &str) -> Option<Self> {
        if let Some(attribute) = key.strip_prefix(ATTRIBUTE_PREFIX) {
            return (!attribute.is_empty()).then(|| Self::Attribute(attribute.to_owned()));
        }
        let named = !key.is_empty() && key.chars().all(|c| c.is_ascii_alphabetic() || c == '_');
        named.then(|| key.parse().ok().map(Self::Field)).flatten()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RunFilter {
    pub start_ms: i64,
    pub end_ms: i64,
    pub as_of_ms: u64,
    pub search: RunSearch,
    /// When not empty, only these runs can match.
    pub trace_refs: Vec<String>,
}

impl RunFilter {
    pub fn window(&self) -> crate::api::TraceQueryWindow {
        crate::api::TraceQueryWindow {
            start_ms: self.start_ms,
            end_ms: self.end_ms,
            as_of_ms: self.as_of_ms,
        }
    }
}

impl Default for RunFilter {
    fn default() -> Self {
        Self {
            start_ms: 0,
            end_ms: 0,
            as_of_ms: u64::MAX,
            search: RunSearch::default(),
            trace_refs: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FieldFilter {
    pub key: SearchKey,
    /// Matched against the whole value, ignoring case; `*` matches any run of characters.
    pub pattern: String,
    pub exclude: bool,
}

/// The parsed `q` of the runs list. Every text term and every filter must hold.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RunSearch {
    /// Each must appear in the trace id, input or name, ignoring case.
    pub text: Vec<String>,
    pub filters: Vec<FieldFilter>,
}

impl RunSearch {
    pub fn parse(q: &str) -> Result<Self, crate::error::InvalidQuery> {
        if q.chars().count() > 1000 {
            return Err(crate::error::InvalidQuery);
        }
        let clauses = tokens(q)
            .map(|token| token.and_then(clause))
            .collect::<Result<Vec<_>, _>>()?;
        let (text, filters): (Vec<_>, Vec<_>) = clauses
            .into_iter()
            .partition(|clause| matches!(clause, Clause::Text(_)));
        Ok(Self {
            text: text
                .into_iter()
                .filter_map(|clause| match clause {
                    Clause::Text(text) => Some(text),
                    _ => None,
                })
                .collect(),
            filters: filters
                .into_iter()
                .filter_map(|clause| match clause {
                    Clause::Field(filter) => Some(filter),
                    _ => None,
                })
                .collect(),
        })
    }
}

enum Clause {
    Text(String),
    Field(FieldFilter),
}

fn tokens(q: &str) -> impl Iterator<Item = Result<&str, crate::error::InvalidQuery>> {
    let mut rest = q;
    std::iter::from_fn(move || {
        rest = rest.trim_start();
        if rest.is_empty() {
            return None;
        }
        let mut quoted = false;
        let end = rest
            .char_indices()
            .find(|&(_, ch)| {
                if ch == '"' {
                    quoted = !quoted;
                }
                !quoted && ch.is_whitespace()
            })
            .map_or(rest.len(), |(index, _)| index);
        let (token, tail) = rest.split_at(end);
        rest = tail;
        Some(if quoted {
            Err(crate::error::InvalidQuery)
        } else {
            Ok(token)
        })
    })
}

fn value(raw: &str) -> Result<String, crate::error::InvalidQuery> {
    let value = if raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2 {
        &raw[1..raw.len() - 1]
    } else {
        raw
    };
    if value.is_empty() || value.contains('"') {
        return Err(crate::error::InvalidQuery);
    }
    Ok(value.to_owned())
}

fn clause(raw: &str) -> Result<Clause, crate::error::InvalidQuery> {
    if raw.starts_with('"') {
        return value(raw).map(Clause::Text);
    }
    let (exclude, body) = raw
        .strip_prefix('-')
        .map_or((false, raw), |body| (true, body));
    let Some((key, raw_value)) = body.split_once(':') else {
        return value(raw).map(Clause::Text);
    };
    let key = SearchKey::parse(key).ok_or(crate::error::InvalidQuery)?;
    let pattern = value(raw_value)?;
    let pattern = match key {
        SearchKey::Field(RunField::RootStatus) => match pattern.to_ascii_lowercase().as_str() {
            "ok" | "error" | "unset" => pattern.to_ascii_lowercase(),
            _ => return Err(crate::error::InvalidQuery),
        },
        SearchKey::Field(RunField::HasError) => match pattern.to_ascii_lowercase().as_str() {
            "true" | "false" => pattern.to_ascii_lowercase(),
            _ => return Err(crate::error::InvalidQuery),
        },
        _ => pattern,
    };
    Ok(Clause::Field(FieldFilter {
        key,
        pattern,
        exclude,
    }))
}

pub const MAX_HISTOGRAM_BUCKETS: u32 = 240;
pub const MAX_RUN_VALUES: u32 = 100;

/// Matching runs per equal-width slice of the window.
#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct TraceHistogram {
    pub window: crate::api::TraceQueryWindow,
    pub buckets: Vec<HistogramBucket>,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramBucket {
    pub start_ms: i64,
    pub end_ms: i64,
    pub total: u64,
    pub failed: u64,
    /// Runs that did not fail, by their alphabetically first agent label, or service when unlabelled.
    pub agents: Vec<AgentRuns>,
}

#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct AgentRuns {
    pub agent: String,
    pub runs: u64,
}

/// Distinct values of one run field, most common first.
#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct RunValues {
    pub window: crate::api::TraceQueryWindow,
    pub values: Vec<String>,
}

/// Bucket `i` covers `[start + span * i / buckets, start + span * (i + 1) / buckets)`.
pub fn histogram(
    rows: &[RunCount],
    window: crate::api::TraceQueryWindow,
    buckets: u32,
) -> TraceHistogram {
    let start_ms = window.start_ms;
    let end_ms = window.end_ms;
    let start = i128::from(start_ms);
    let span = i128::from(end_ms) - start;
    let edge = |index: u32| (start + span * i128::from(index) / i128::from(buckets)) as i64;
    TraceHistogram {
        window,
        buckets: (0..buckets)
            .map(|index| {
                let hits = rows.iter().filter(|row| row.bucket == index);
                let mut agents: Vec<AgentRuns> = hits
                    .clone()
                    .filter(|row| !row.failed)
                    .map(|row| AgentRuns {
                        agent: row.value.clone(),
                        runs: row.runs,
                    })
                    .collect();
                agents.sort_by(|left, right| left.agent.cmp(&right.agent));
                HistogramBucket {
                    start_ms: edge(index),
                    end_ms: edge(index + 1),
                    total: hits.clone().map(|row| row.runs).sum(),
                    failed: hits.filter(|row| row.failed).map(|row| row.runs).sum(),
                    agents,
                }
            })
            .collect(),
    }
}
