use std::collections::BTreeSet;

use indexmap::IndexMap;

use crate::{
    CallEvidence, CallKey, SpendMatch,
    query::named::{SpendByResponseIdsRow as SpendRow, TraceSpansRow},
};

#[derive(Debug, Default, PartialEq)]
pub struct SpendLookup {
    pub response_ids: Vec<String>,
    pub call_ids: Vec<String>,
}

impl SpendLookup {
    pub fn new(rows: &[TraceSpansRow]) -> Self {
        let keys: BTreeSet<CallKey> = rows.iter().flat_map(CallEvidence::row_keys).collect();
        let ids = |pick: fn(&CallKey) -> Option<&String>| {
            keys.iter()
                .filter_map(pick)
                .filter(|id| !id.is_empty())
                .cloned()
                .collect()
        };
        Self {
            response_ids: ids(|key| match key {
                CallKey::ProviderResponse(id) => Some(id),
                _ => None,
            }),
            call_ids: ids(|key| match key {
                CallKey::LiteLlmRequest(id) => Some(id),
                _ => None,
            }),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.response_ids.is_empty() && self.call_ids.is_empty()
    }
}

fn assigned_id(key: &CallKey) -> bool {
    match key {
        CallKey::ProviderResponse(id) | CallKey::LiteLlmRequest(id) => !id.is_empty(),
        CallKey::Transport | CallKey::GatewayAttempt => false,
    }
}

fn names(key: &CallKey, spend: &SpendRow) -> bool {
    match key {
        CallKey::ProviderResponse(id) => {
            spend.response_id == *id || spend.upstream_response_id == *id
        }
        CallKey::LiteLlmRequest(id) => spend.litellm_call_id == *id,
        CallKey::Transport | CallKey::GatewayAttempt => false,
    }
}

pub(super) type Requests<'a> = Vec<&'a SpendRow>;

pub(super) fn unique<'a>(requests: impl IntoIterator<Item = &'a SpendRow>) -> Requests<'a> {
    requests
        .into_iter()
        .map(|request| (request.identity(), request))
        .collect::<IndexMap<_, _>>()
        .into_values()
        .collect()
}

pub(super) fn call_ids(row: &TraceSpansRow) -> BTreeSet<CallKey> {
    CallEvidence::from_row(row)
        .key_set()
        .into_iter()
        .flatten()
        .filter(|key| assigned_id(key))
        .cloned()
        .collect()
}

pub(super) fn match_ids<'a>(
    ids: &BTreeSet<CallKey>,
    spend_rows: &[&'a SpendRow],
) -> (Option<Requests<'a>>, SpendMatch) {
    if ids.is_empty() {
        return (None, SpendMatch::NoCallId);
    }
    let named: Vec<Requests<'a>> = ids
        .iter()
        .map(|id| unique(spend_rows.iter().copied().filter(|spend| names(id, spend))))
        .collect();
    if named.iter().any(|rows| rows.len() > 1) {
        return (None, SpendMatch::Ambiguous);
    }
    let matched = unique(named.into_iter().flatten());
    if matched.is_empty() {
        return (None, SpendMatch::NoSpendLog);
    }
    (Some(matched), SpendMatch::Matched)
}

pub(super) fn request_cost(requests: &[&SpendRow]) -> Option<f64> {
    requests.iter().try_fold(0.0, |total, request| {
        let cost = request.spend.filter(|cost| cost.is_finite())?;
        let sum = total + cost;
        sum.is_finite().then_some(sum)
    })
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
