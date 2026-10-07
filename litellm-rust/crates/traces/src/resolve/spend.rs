use std::collections::BTreeSet;

use indexmap::IndexMap;

use crate::{
    CallEvidence, CallEvidenceKind, CallKey, SpendMatch,
    query::named::{SpendByResponseIdsRow as SpendRow, TraceSpansRow},
};

/// The spend records to fetch for a set of spans.
#[derive(Debug, Default, PartialEq)]
pub struct SpendLookup {
    pub response_ids: Vec<String>,
    pub request_ids: Vec<String>,
    pub provider_request_ids: Vec<String>,
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
            provider_request_ids: sorted(
                keys()
                    .filter_map(|(_, key)| match key {
                        CallKey::ProviderRequest(id) => Some(id.clone()),
                        _ => None,
                    })
                    .collect(),
            ),
            trace_ids: sorted(
                keys()
                    .filter_map(|(row, key)| match key {
                        CallKey::Transport | CallKey::GatewayAttempt
                            if !row.transport_trace_id().is_empty() =>
                        {
                            Some(row.transport_trace_id().to_owned())
                        }
                        _ => None,
                    })
                    .collect(),
            ),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.response_ids.is_empty()
            && self.request_ids.is_empty()
            && self.provider_request_ids.is_empty()
            && self.trace_ids.is_empty()
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

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum KeyFamily {
    GatewayCall,
    ProviderResponse,
    ProviderRequest,
    Transport,
}

fn key_family(key: &CallKey) -> KeyFamily {
    match key {
        CallKey::LiteLlmRequest(_) => KeyFamily::GatewayCall,
        CallKey::ProviderResponse(_) => KeyFamily::ProviderResponse,
        CallKey::ProviderRequest(_) => KeyFamily::ProviderRequest,
        CallKey::Transport | CallKey::GatewayAttempt => KeyFamily::Transport,
    }
}

pub(super) enum KeyMatch<'a> {
    Missing,
    Conflicting,
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
            Self::Missing | Self::Conflicting | Self::Ambiguous(_) => None,
        }
    }

    fn agrees_with(&self, selected: &[&SpendRow]) -> bool {
        match self {
            Self::Missing | Self::Conflicting => false,
            Self::Unique(request) => selected
                .iter()
                .any(|row| row.identity() == request.identity()),
            Self::Ambiguous(requests) => {
                requests
                    .iter()
                    .filter(|request| {
                        selected
                            .iter()
                            .any(|row| row.identity() == request.identity())
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
    pub(super) fn unmatched_reason(&self) -> SpendMatch {
        match self {
            Self::Unknown => SpendMatch::NoCallId,
            Self::Partial(_) => SpendMatch::IncompleteEvidence,
            Self::Complete(matches) if matches.is_empty() => SpendMatch::NoCallId,
            Self::Complete(matches)
                if matches.iter().any(|evidence| {
                    matches!(evidence, KeyMatch::Conflicting | KeyMatch::Ambiguous(_))
                }) =>
            {
                SpendMatch::Ambiguous
            }
            Self::Complete(matches)
                if matches
                    .iter()
                    .any(|evidence| matches!(evidence, KeyMatch::Missing)) =>
            {
                SpendMatch::NoSpendLog
            }
            Self::Complete(_) => SpendMatch::Ambiguous,
        }
    }

    pub(super) fn complete_requests(&self) -> Option<Requests<'a>> {
        match self {
            Self::Complete(matches) if !matches.is_empty() => {
                let requests: Requests<'a> = matches
                    .iter()
                    .filter_map(KeyMatch::unique)
                    .map(|request| (request.identity(), request))
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
) -> IndexMap<(&'a str, i64, &'a str), &'a SpendRow> {
    let matches = |spend: &SpendRow| match key {
        CallKey::ProviderResponse(id) => {
            !id.is_empty() && (spend.response_id == *id || spend.upstream_response_id == *id)
        }
        CallKey::ProviderRequest(id) => !id.is_empty() && spend.provider_request_id == *id,
        CallKey::LiteLlmRequest(id) => {
            !id.is_empty()
                && (spend.litellm_call_id == *id
                    || (spend.litellm_call_id.is_empty() && spend.request_id == *id))
        }
        CallKey::Transport | CallKey::GatewayAttempt => {
            !row.transport_trace_id().is_empty()
                && !row.span_id.is_empty()
                && spend.trace_id == row.transport_trace_id()
                && spend.span_id == row.span_id
        }
    };
    spend_rows
        .iter()
        .filter(|spend| ownership.owns(spend) && matches(spend))
        .map(|spend| (spend.identity(), spend))
        .collect()
}

pub(super) fn requests<'a>(
    row: &TraceSpansRow,
    ownership: &Ownership<'_>,
    spend_rows: &'a [SpendRow],
) -> SpendEvidence<'a> {
    let evidence = CallEvidence::from_row(row);
    let keyed: Vec<(&CallKey, Requests<'a>)> = evidence
        .key_set()
        .into_iter()
        .flatten()
        .map(|key| {
            (
                key,
                matches(ownership, spend_rows, key, row)
                    .into_values()
                    .collect(),
            )
        })
        .collect();
    let anchored: Vec<&SpendRow> = keyed
        .iter()
        .filter(|(key, _)| !matches!(key, CallKey::LiteLlmRequest(_)))
        .flat_map(|(_, requests)| requests.iter().copied())
        .collect();
    let legacy_rows = !anchored.is_empty()
        && anchored
            .iter()
            .all(|request| request.litellm_call_id.is_empty());
    let aliases: Vec<_> = keyed
        .into_iter()
        .filter(|(key, requests)| {
            !(legacy_rows && requests.is_empty() && matches!(key, CallKey::LiteLlmRequest(_)))
        })
        .collect();
    let families: BTreeSet<_> = aliases.iter().map(|(key, _)| key_family(key)).collect();
    let compatible_rows: Vec<BTreeSet<_>> = families
        .into_iter()
        .map(|family| {
            aliases
                .iter()
                .filter(|(key, _)| key_family(key) == family)
                .flat_map(|(_, requests)| requests.iter().map(|request| request.identity()))
                .collect()
        })
        .collect();
    let matches = aliases
        .into_iter()
        .map(|(_, requests)| {
            let had_candidates = !requests.is_empty();
            let matched = KeyMatch::new(
                requests
                    .into_iter()
                    .filter(|request| {
                        compatible_rows
                            .iter()
                            .all(|family| family.contains(&request.identity()))
                    })
                    .collect(),
            );
            if had_candidates && matches!(matched, KeyMatch::Missing) {
                KeyMatch::Conflicting
            } else {
                matched
            }
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

pub(super) fn unique<'a>(requests: impl IntoIterator<Item = &'a SpendRow>) -> Requests<'a> {
    requests
        .into_iter()
        .map(|request| (request.identity(), request))
        .collect::<IndexMap<_, _>>()
        .into_values()
        .collect()
}

pub(super) struct Priced {
    pub(super) spend: Option<f64>,
    pub(super) priced_calls: u64,
}

pub(super) fn total(calls: &[Option<Requests<'_>>]) -> Priced {
    let priced: Vec<&Requests<'_>> = calls
        .iter()
        .flatten()
        .filter(|requests| request_cost(requests).is_some())
        .collect();
    let spend = if priced.is_empty() {
        None
    } else {
        request_cost(&unique(
            priced.iter().flat_map(|requests| requests.iter().copied()),
        ))
    };
    Priced {
        spend,
        priced_calls: priced.len() as u64,
    }
}
