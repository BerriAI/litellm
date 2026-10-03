use std::collections::BTreeSet;

use indexmap::IndexMap;

use crate::{
    CallEvidence, CallEvidenceKind, CallKey,
    query::named::{SpendByResponseIdsRow as SpendRow, TraceSpansRow},
};

/// The spend records to fetch for a set of spans.
#[derive(Debug, Default, PartialEq)]
pub struct SpendLookup {
    pub response_ids: Vec<String>,
    pub request_ids: Vec<String>,
    /// Traces whose transport spans LiteLLM logged by `traceparent`.
    pub trace_ids: Vec<String>,
}

impl SpendLookup {
    pub fn new(rows: &[TraceSpansRow]) -> Self {
        let evidence: Vec<_> = rows
            .iter()
            .map(|row| (row, CallEvidence::row_keys(row)))
            .collect();
        let keys = || {
            evidence
                .iter()
                .flat_map(|(row, calls)| calls.iter().map(move |key| (*row, key)))
        };
        let sorted = |values: BTreeSet<String>| values.into_iter().collect();
        Self {
            response_ids: sorted(
                keys()
                    .filter_map(|(_, key)| match key {
                        CallKey::ProviderResponse(id) => Some(id.clone()),
                        _ => None,
                    })
                    .collect(),
            ),
            request_ids: sorted(
                keys()
                    .filter_map(|(_, key)| match key {
                        CallKey::LiteLlmRequest(id) => Some(id.clone()),
                        _ => None,
                    })
                    .collect(),
            ),
            trace_ids: sorted(
                keys()
                    .filter_map(|(row, key)| match key {
                        CallKey::Transport if !row.trace_id.is_empty() => {
                            Some(row.trace_id.clone())
                        }
                        _ => None,
                    })
                    .collect(),
            ),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.response_ids.is_empty() && self.request_ids.is_empty() && self.trace_ids.is_empty()
    }
}

/// Who a trace's spend records must belong to.
pub(super) struct Ownership<'a> {
    pub(super) team_id: &'a str,
    pub(super) api_key_hash: &'a str,
    pub(super) user_id: &'a str,
}

impl Ownership<'_> {
    fn owns(&self, spend: &SpendRow) -> bool {
        spend.team_id == self.team_id
            && ((!self.user_id.is_empty() && spend.user == self.user_id)
                || (!self.api_key_hash.is_empty() && spend.api_key == self.api_key_hash))
    }
}

pub(super) type Requests<'a> = Vec<&'a SpendRow>;

pub(super) enum KeyMatch<'a> {
    Missing,
    Unique(&'a SpendRow),
    Ambiguous(Requests<'a>),
}

impl<'a> KeyMatch<'a> {
    fn new(requests: Requests<'a>) -> Self {
        match requests.as_slice() {
            [] => Self::Missing,
            [request] => Self::Unique(request),
            _ => Self::Ambiguous(requests),
        }
    }

    fn unique(&self) -> Option<&'a SpendRow> {
        match self {
            Self::Unique(request) => Some(request),
            Self::Missing | Self::Ambiguous(_) => None,
        }
    }

    fn agrees_with(&self, selected: &[&SpendRow]) -> bool {
        match self {
            Self::Missing => false,
            Self::Unique(request) => selected
                .iter()
                .any(|row| row.request_id == request.request_id),
            Self::Ambiguous(requests) => {
                requests
                    .iter()
                    .filter(|request| {
                        selected
                            .iter()
                            .any(|row| row.request_id == request.request_id)
                    })
                    .count()
                    == 1
            }
        }
    }
}

pub(super) enum SpendEvidence<'a> {
    Unknown,
    Partial(Vec<KeyMatch<'a>>),
    Complete(Vec<KeyMatch<'a>>),
}

impl<'a> SpendEvidence<'a> {
    pub(super) fn complete_requests(&self) -> Option<Requests<'a>> {
        match self {
            Self::Complete(matches) if !matches.is_empty() => {
                let requests: Requests<'a> = matches
                    .iter()
                    .filter_map(KeyMatch::unique)
                    .map(|request| (request.request_id.as_str(), request))
                    .collect::<IndexMap<_, _>>()
                    .into_values()
                    .collect();
                matches
                    .iter()
                    .all(|matched| matched.agrees_with(&requests))
                    .then_some(requests)
            }
            Self::Unknown | Self::Partial(_) | Self::Complete(_) => None,
        }
    }

    pub(super) fn agrees_with(&self, selected: &[&SpendRow]) -> bool {
        match self {
            Self::Unknown => true,
            Self::Partial(matches) | Self::Complete(matches) => matches
                .iter()
                .all(|evidence| evidence.agrees_with(selected)),
        }
    }
}

fn matches<'a>(
    ownership: &Ownership<'_>,
    spend_rows: &'a [SpendRow],
    key: &CallKey,
    row: &TraceSpansRow,
) -> IndexMap<&'a str, &'a SpendRow> {
    let matches = |spend: &SpendRow| match key {
        CallKey::ProviderResponse(id) => {
            !id.is_empty() && (spend.response_id == *id || spend.upstream_response_id == *id)
        }
        CallKey::LiteLlmRequest(id) => !id.is_empty() && spend.request_id == *id,
        CallKey::Transport => {
            !row.trace_id.is_empty()
                && !row.span_id.is_empty()
                && spend.trace_id == row.trace_id
                && spend.span_id == row.span_id
        }
    };
    spend_rows
        .iter()
        .filter(|spend| ownership.owns(spend) && matches(spend))
        .map(|spend| (spend.request_id.as_str(), spend))
        .collect()
}

pub(super) fn requests<'a>(
    row: &TraceSpansRow,
    ownership: &Ownership<'_>,
    spend_rows: &'a [SpendRow],
) -> SpendEvidence<'a> {
    let evidence = CallEvidence::from_row(row);
    let matches = evidence
        .key_set()
        .into_iter()
        .flatten()
        .map(|key| {
            KeyMatch::new(
                matches(ownership, spend_rows, key, row)
                    .into_values()
                    .collect(),
            )
        })
        .collect();
    match evidence.kind() {
        CallEvidenceKind::Complete => SpendEvidence::Complete(matches),
        CallEvidenceKind::Partial => SpendEvidence::Partial(matches),
        CallEvidenceKind::Unknown => SpendEvidence::Unknown,
    }
}

pub(super) fn request_cost(requests: &[&SpendRow]) -> Option<f64> {
    requests.iter().try_fold(0.0, |total, request| {
        let cost = request.spend.filter(|cost| cost.is_finite())?;
        let sum = total + cost;
        sum.is_finite().then_some(sum)
    })
}

pub(super) fn total(calls: &[Option<Requests<'_>>]) -> Option<f64> {
    if calls.is_empty() {
        return None;
    }
    let requests: Option<Vec<&SpendRow>> = calls
        .iter()
        .map(|requests| requests.as_ref())
        .collect::<Option<Vec<_>>>()
        .map(|calls| calls.into_iter().flatten().copied().collect());
    let unique: IndexMap<&str, &SpendRow> = requests?
        .into_iter()
        .map(|request| (request.request_id.as_str(), request))
        .collect();
    request_cost(&unique.into_values().collect::<Vec<_>>())
}
