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
    Status,
    Model,
    Input,
    TraceId,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RunFilter {
    pub start_ms: i64,
    pub end_ms: i64,
    pub search: RunSearch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldFilter {
    pub field: RunField,
    /// Matched against the whole value, ignoring case; `*` matches any run of characters.
    pub pattern: String,
    pub exclude: bool,
}

/// The parsed `q` of the runs list. Every text term and every filter must hold.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RunSearch {
    /// Each must appear in the trace id, input or name, ignoring case.
    pub text: Vec<String>,
    pub filters: Vec<FieldFilter>,
}

impl RunSearch {
    /// Mirrors the dashboard's search box: `key:value` filters on known keys, `-key:value` negates,
    /// `*` globs, double quotes keep spaces, and anything else is a free-text term.
    /// A key typed without a value yet narrows nothing.
    pub fn parse(q: &str) -> Self {
        let (text, filters): (Vec<_>, Vec<_>) = tokens(q)
            .map(clause)
            .filter(|clause| !clause.value().is_empty())
            .partition(|clause| matches!(clause, Clause::Text(_)));
        Self {
            text: text
                .into_iter()
                .map(|clause| clause.value().to_owned())
                .collect(),
            filters: filters
                .into_iter()
                .filter_map(|clause| match clause {
                    Clause::Field {
                        field,
                        exclude,
                        value,
                    } => Some(FieldFilter {
                        field,
                        pattern: value,
                        exclude,
                    }),
                    Clause::Text(_) => None,
                })
                .collect(),
        }
    }
}

enum Clause {
    Text(String),
    Field {
        field: RunField,
        exclude: bool,
        value: String,
    },
}

impl Clause {
    fn value(&self) -> &str {
        match self {
            Self::Text(value) | Self::Field { value, .. } => value,
        }
    }
}

/// Whitespace-separated tokens; a double-quoted stretch keeps its spaces, and an unclosed quote runs to the end.
fn tokens(q: &str) -> impl Iterator<Item = &str> {
    let mut rest = q;
    std::iter::from_fn(move || {
        rest = rest.trim_start();
        if rest.is_empty() {
            return None;
        }
        let mut quoted = false;
        let end = rest
            .char_indices()
            .find(|&(_, char)| {
                if char == '"' {
                    quoted = !quoted;
                }
                !quoted && char.is_whitespace()
            })
            .map_or(rest.len(), |(index, _)| index);
        let (token, tail) = rest.split_at(end);
        rest = tail;
        Some(token)
    })
}

fn unquote(raw: &str) -> String {
    raw.strip_prefix('"')
        .map(|inner| inner.strip_suffix('"').unwrap_or(inner))
        .filter(|inner| !inner.contains('"'))
        .unwrap_or(raw)
        .to_owned()
}

fn clause(raw: &str) -> Clause {
    let (exclude, body) = raw
        .strip_prefix('-')
        .map_or((false, raw), |body| (true, body));
    let field = body.split_once(':').and_then(|(key, value)| {
        let named = !key.is_empty() && key.chars().all(|c| c.is_ascii_alphabetic() || c == '_');
        named
            .then(|| key.parse::<RunField>().ok())
            .flatten()
            .map(|field| (field, value))
    });
    match field {
        Some((field, value)) => Clause::Field {
            field,
            exclude,
            value: unquote(value),
        },
        None => Clause::Text(unquote(raw)),
    }
}

pub const MAX_HISTOGRAM_BUCKETS: u32 = 240;
pub const MAX_RUN_VALUES: u32 = 100;

/// Matching runs per equal-width slice of the window.
#[macro_rules_attribute::apply(response_type)]
#[derive(Clone, Debug, PartialEq)]
pub struct TraceHistogram {
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
    pub values: Vec<String>,
}

/// Bucket `i` covers `[start + span * i / buckets, start + span * (i + 1) / buckets)`.
pub fn histogram(rows: &[RunCount], start_ms: i64, end_ms: i64, buckets: u32) -> TraceHistogram {
    let span = i128::from(end_ms - start_ms);
    let edge = |index: u32| start_ms + (span * i128::from(index) / i128::from(buckets)) as i64;
    TraceHistogram {
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
