use std::collections::BTreeSet;

use indexmap::IndexMap;

use crate::{
    CallEvidence, CallKey,
    query::named::{SpendByResponseIdsRow as SpendRow, TraceSpansRow},
};

/// The `spend_logs.response_id` values a set of spans names as `gen_ai.response.id`.
#[derive(Debug, Default, PartialEq)]
pub struct SpendLookup {
    pub response_ids: Vec<String>,
}

impl SpendLookup {
    pub fn new(rows: &[TraceSpansRow]) -> Self {
        let ids: BTreeSet<String> = rows
            .iter()
            .flat_map(CallEvidence::row_keys)
            .filter_map(response_id)
            .collect();
        Self {
            response_ids: ids.into_iter().collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.response_ids.is_empty()
    }
}

fn response_id(key: CallKey) -> Option<String> {
    match key {
        CallKey::ProviderResponse(id) if !id.is_empty() => Some(id),
        CallKey::ProviderResponse(_)
        | CallKey::LiteLlmRequest(_)
        | CallKey::Transport
        | CallKey::GatewayAttempt => None,
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

fn unique_match<'a>(id: &str, spend_rows: &'a [SpendRow]) -> Option<&'a SpendRow> {
    match unique(
        spend_rows
            .iter()
            .filter(|spend| spend.response_id == id || spend.upstream_response_id == id),
    )
    .as_slice()
    {
        [request] => Some(request),
        _ => None,
    }
}

pub(super) fn response_ids(row: &TraceSpansRow) -> BTreeSet<String> {
    CallEvidence::from_row(row)
        .key_set()
        .into_iter()
        .flatten()
        .cloned()
        .filter_map(response_id)
        .collect()
}

/// The spend rows that `ids` name, or `None` when there are none or any id matches no row or
/// more than one.
pub(super) fn requests<'a>(
    ids: &BTreeSet<String>,
    spend_rows: &'a [SpendRow],
) -> Option<Requests<'a>> {
    if ids.is_empty() {
        return None;
    }
    ids.iter().map(|id| unique_match(id, spend_rows)).collect()
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
