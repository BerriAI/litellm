use crate::{
    normalize::CallKey,
    query::named::{SpendByResponseIdsRow as SpendRow, TraceSpansRow},
};
use indexmap::IndexMap;
use std::collections::BTreeSet;
fn parse_call_key(encoded: &str) -> Option<CallKey> {
    let (kind, id) = encoded.split_once(':')?;
    match kind {
        "provider_response" if !id.is_empty() => Some(CallKey::ProviderResponse(id.to_owned())),
        "litellm_request" if !id.is_empty() => Some(CallKey::LiteLlmRequest(id.to_owned())),
        "transport" if id.is_empty() => Some(CallKey::Transport),
        _ => None,
    }
}

pub(super) fn call_keys(row: &TraceSpansRow) -> Vec<Option<CallKey>> {
    if row.call_keys.is_empty() && !row.litellm_request_id.is_empty() {
        return vec![Some(CallKey::ProviderResponse(
            row.litellm_request_id.clone(),
        ))];
    }
    row.call_keys
        .iter()
        .map(|key| parse_call_key(key))
        .collect()
}

fn call_evidence(row: &TraceSpansRow) -> &str {
    match (
        row.call_evidence.as_str(),
        row.litellm_request_id.is_empty(),
    ) {
        ("", false) => "complete",
        ("", true) => "unknown",
        (recorded, _) => recorded,
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct SpendLookup {
    pub response_ids: Vec<String>,
    pub request_ids: Vec<String>,
    pub trace_ids: Vec<String>,
}

impl SpendLookup {
    pub fn new(rows: &[TraceSpansRow]) -> Self {
        let mut response_ids = BTreeSet::new();
        let mut request_ids = BTreeSet::new();
        let mut trace_ids = BTreeSet::new();
        for row in rows {
            for key in call_keys(row) {
                match key {
                    Some(CallKey::ProviderResponse(id)) => {
                        response_ids.insert(id.to_owned());
                    }
                    Some(CallKey::LiteLlmRequest(id)) => {
                        request_ids.insert(id.to_owned());
                    }
                    Some(CallKey::Transport) if !row.trace_id.is_empty() => {
                        trace_ids.insert(row.trace_id.clone());
                    }
                    _ => {}
                }
            }
        }
        Self {
            response_ids: response_ids.into_iter().collect(),
            request_ids: request_ids.into_iter().collect(),
            trace_ids: trace_ids.into_iter().collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.response_ids.is_empty() && self.request_ids.is_empty() && self.trace_ids.is_empty()
    }
}

pub(super) struct Ownership<'a> {
    pub(super) team_id: &'a str,
    pub(super) api_key_hash: &'a str,
    pub(super) user_id: &'a str,
}

impl Ownership<'_> {
    pub(super) fn owns(&self, spend: &SpendRow) -> bool {
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
    pub(super) fn new(requests: Requests<'a>) -> Self {
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

    pub(super) fn agrees_with(&self, selected: &[&SpendRow]) -> bool {
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

fn matches<'a>(
    ownership: &Ownership<'_>,
    spend_rows: &'a [SpendRow],
    key: &Option<CallKey>,
    row: &TraceSpansRow,
) -> IndexMap<&'a str, &'a SpendRow> {
    let matches = |spend: &SpendRow| match key {
        Some(CallKey::ProviderResponse(id)) => {
            !id.is_empty() && (spend.response_id == *id || spend.upstream_response_id == *id)
        }
        Some(CallKey::LiteLlmRequest(id)) => !id.is_empty() && spend.request_id == *id,
        Some(CallKey::Transport) => {
            !row.trace_id.is_empty()
                && !row.span_id.is_empty()
                && spend.trace_id == row.trace_id
                && spend.span_id == row.span_id
        }
        None => false,
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
    let matches = call_keys(row)
        .iter()
        .map(|key| {
            KeyMatch::new(
                matches(ownership, spend_rows, key, row)
                    .into_values()
                    .collect(),
            )
        })
        .collect();
    match call_evidence(row) {
        "complete" => SpendEvidence::Complete(matches),
        "partial" => SpendEvidence::Partial(matches),
        _ => SpendEvidence::Unknown,
    }
}
