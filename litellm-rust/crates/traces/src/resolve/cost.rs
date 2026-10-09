use std::collections::{BTreeMap, BTreeSet};

use crate::{
    CallEvidence, CallKey, CostSource, SpendMatch,
    query::named::{SpendByResponseIdsRow, TraceSpansRow},
};

use super::{
    resolution::Resolution,
    spend::{Requests, request_cost, unique},
};

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct TraceCostInput {
    pub start_ns: i64,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum EstimateIdentity {
    Response(String),
    Request(String),
    Gateway(String),
    Span(String),
}

pub(super) enum CallCost<'a> {
    Gateway(Requests<'a>),
    Estimated(EstimateIdentity, f64),
}

impl CallCost<'_> {
    pub(super) fn source(&self) -> CostSource {
        match self {
            Self::Gateway(_) => CostSource::Gateway,
            Self::Estimated(_, _) => CostSource::Estimated,
        }
    }

    pub(super) fn amount(&self) -> Option<f64> {
        match self {
            Self::Gateway(requests) => request_cost(requests),
            Self::Estimated(_, amount) => Some(*amount),
        }
    }
}

fn identity(resolution: &Resolution<'_>, call: usize) -> Option<EstimateIdentity> {
    let matched = resolution.call_match(call)?;
    if matches!(
        matched.state,
        SpendMatch::Ambiguous | SpendMatch::IncompleteEvidence
    ) || matched
        .requests
        .as_ref()
        .is_some_and(|requests| requests.len() > 1)
    {
        return None;
    }
    let transports = resolution.transports(call);
    if transports.len() > 1 {
        return None;
    }
    let mut responses = BTreeSet::new();
    let mut requests = BTreeSet::new();
    let mut gateway = BTreeSet::new();
    for index in resolution.call_sources(call).into_iter().chain(transports) {
        let keys = match CallEvidence::from_row(resolution.row(index)) {
            CallEvidence::Partial(_) => return None,
            CallEvidence::Complete(keys) => keys,
            CallEvidence::Unknown => continue,
        };
        for key in keys {
            match key {
                CallKey::ProviderResponse(id) => {
                    responses.insert(id);
                }
                CallKey::ProviderRequest(id) => {
                    requests.insert(id);
                }
                CallKey::LiteLlmRequest(id) => {
                    gateway.insert(id);
                }
                CallKey::Transport | CallKey::GatewayAttempt => {}
            }
        }
    }
    if responses.len() > 1 || requests.len() > 1 || gateway.len() > 1 {
        return None;
    }
    Some(if let Some(id) = responses.into_iter().next() {
        EstimateIdentity::Response(id)
    } else if let Some(id) = requests.into_iter().next() {
        EstimateIdentity::Request(id)
    } else if let Some(id) = gateway.into_iter().next() {
        EstimateIdentity::Gateway(id)
    } else {
        EstimateIdentity::Span(resolution.row(call).span_id.clone())
    })
}

fn candidates(resolution: &Resolution<'_>) -> BTreeMap<EstimateIdentity, Vec<usize>> {
    let mut groups: BTreeMap<EstimateIdentity, Vec<usize>> = BTreeMap::new();
    for call in &resolution.model_calls {
        if let Some(key) = identity(resolution, *call) {
            groups.entry(key).or_default().push(*call);
        }
    }
    groups.retain(|_, calls| {
        let first = &resolution.row(calls[0]).pricing_attributes;
        !first.is_empty()
            && calls.iter().all(|call| {
                &resolution.row(*call).pricing_attributes == first
                    && resolution
                        .call_requests(*call)
                        .as_ref()
                        .and_then(|requests| request_cost(requests))
                        .is_none()
            })
    });
    groups
}

pub fn trace_cost_inputs(
    rows: &[TraceSpansRow],
    spend: &[SpendByResponseIdsRow],
) -> Vec<(Vec<String>, TraceCostInput)> {
    if rows.iter().all(|row| row.pricing_attributes.is_empty()) {
        return Vec::new();
    }
    let resolution = Resolution::new(rows, spend);
    candidates(&resolution)
        .into_values()
        .map(|calls| {
            let first = resolution.row(calls[0]);
            (
                calls
                    .iter()
                    .map(|call| resolution.row(*call).span_id.clone())
                    .collect(),
                TraceCostInput {
                    start_ns: calls
                        .iter()
                        .map(|call| resolution.row(*call).start_ns)
                        .min()
                        .unwrap_or(first.start_ns),
                    attributes: first.pricing_attributes.clone(),
                },
            )
        })
        .collect()
}

pub(super) fn selected<'a>(
    resolution: &Resolution<'a>,
    estimates: &BTreeMap<String, f64>,
) -> BTreeMap<usize, CallCost<'a>> {
    let mut costs: BTreeMap<_, _> = resolution
        .model_calls
        .iter()
        .filter_map(|call| {
            let requests = resolution.call_requests(*call)?;
            request_cost(&requests)?;
            Some((*call, CallCost::Gateway(requests)))
        })
        .collect();
    if estimates.is_empty() {
        return costs;
    }
    for (key, calls) in candidates(resolution) {
        let amounts: Option<Vec<_>> = calls
            .iter()
            .map(|call| {
                estimates
                    .get(&resolution.row(*call).span_id)
                    .copied()
                    .filter(|cost| cost.is_finite() && *cost >= 0.0)
            })
            .collect();
        let Some(amounts) = amounts else {
            continue;
        };
        let amount = amounts[0];
        if amounts.iter().any(|value| *value != amount) {
            continue;
        }
        for call in calls {
            costs.insert(call, CallCost::Estimated(key.clone(), amount));
        }
    }
    costs
}

pub(super) struct Priced {
    pub(super) spend: Option<f64>,
    pub(super) priced_calls: u64,
    pub(super) estimated_calls: u64,
}

pub(super) fn total<'a>(calls: impl IntoIterator<Item = &'a CallCost<'a>>) -> Priced {
    let mut requests = Vec::new();
    let mut estimates = BTreeMap::new();
    let mut priced_calls = 0;
    let mut estimated_calls = 0;
    for call in calls {
        priced_calls += 1;
        match call {
            CallCost::Gateway(rows) => requests.extend(rows.iter().copied()),
            CallCost::Estimated(key, amount) => {
                estimates.insert(key, amount);
                estimated_calls += 1;
            }
        }
    }
    let spend = (priced_calls > 0)
        .then(|| {
            estimates
                .into_values()
                .try_fold(request_cost(&unique(requests))?, |total, amount| {
                    let sum = total + amount;
                    sum.is_finite().then_some(sum)
                })
        })
        .flatten();
    Priced {
        spend,
        priced_calls,
        estimated_calls,
    }
}
