use std::collections::{BTreeMap, BTreeSet};

use crate::{
    CallEvidence, CallKey, SpendMatch,
    query::named::{SpendByResponseIdsRow, TraceSpansRow},
};

use super::{
    resolution::Resolution,
    spend::{self, Priced},
};

fn eligible(resolution: &Resolution<'_>, call: usize) -> bool {
    match resolution
        .call_match(call)
        .map(|matched| (matched.requests.as_ref(), matched.state))
    {
        Some((None, SpendMatch::NoCallId | SpendMatch::NoSpendLog)) => true,
        Some((None, SpendMatch::IncompleteEvidence)) => {
            let row = resolution.row(call);
            let operation = row
                .pricing_attributes
                .get("gen_ai.operation.name")
                .map(String::as_str);
            matches!(
                operation,
                Some("chat" | "text_completion" | "generate_content")
            ) && matches!(
                CallEvidence::row_keys(row)
                    .iter()
                    .collect::<Vec<_>>()
                    .as_slice(),
                [CallKey::ProviderResponse(_)]
            )
        }
        _ => false,
    }
}

pub fn estimate_candidates<'a>(
    rows: &'a [TraceSpansRow],
    spend: &[SpendByResponseIdsRow],
) -> Vec<&'a TraceSpansRow> {
    if rows.is_empty() {
        return Vec::new();
    }
    let resolution = Resolution::new(rows, spend);
    eligible_groups(&resolution)
        .into_iter()
        .flatten()
        .map(|call| &rows[call])
        .filter(|row| !row.pricing_attributes.is_empty())
        .collect()
}

fn eligible_groups(resolution: &Resolution<'_>) -> Vec<Vec<usize>> {
    resolution
        .graph
        .call_groups(&resolution.model_calls)
        .into_iter()
        .filter(|group| {
            let first = resolution.row(group[0]);
            let mut keys = BTreeSet::new();
            group.iter().all(|call| {
                let row = resolution.row(*call);
                keys.extend(CallEvidence::row_keys(row));
                eligible(resolution, *call) && row.pricing_attributes == first.pricing_attributes
            }) && [
                keys.iter()
                    .filter(|key| matches!(key, CallKey::ProviderResponse(_)))
                    .count(),
                keys.iter()
                    .filter(|key| matches!(key, CallKey::ProviderRequest(_)))
                    .count(),
                keys.iter()
                    .filter(|key| matches!(key, CallKey::LiteLlmRequest(_)))
                    .count(),
            ]
            .into_iter()
            .all(|count| count <= 1)
        })
        .collect()
}

pub(super) type ValidatedEstimates = BTreeMap<usize, (usize, f64)>;

pub(super) fn validated(
    resolution: &Resolution<'_>,
    estimates: &BTreeMap<String, f64>,
) -> ValidatedEstimates {
    eligible_groups(resolution)
        .into_iter()
        .filter_map(|group| {
            let amount = estimates.get(&resolution.row(group[0]).span_id).copied()?;
            if !amount.is_finite()
                || amount < 0.0
                || group.iter().any(|call| {
                    estimates.get(&resolution.row(*call).span_id).copied() != Some(amount)
                })
            {
                return None;
            }
            let identity = group[0];
            Some(
                group
                    .into_iter()
                    .map(move |call| (call, (identity, amount))),
            )
        })
        .flatten()
        .collect()
}

pub(super) fn cost(call: usize, estimates: &ValidatedEstimates) -> Option<f64> {
    estimates.get(&call).map(|(_, amount)| *amount)
}

pub(super) fn total(
    resolution: &Resolution<'_>,
    calls: &[usize],
    estimates: &ValidatedEstimates,
) -> (Priced, u64) {
    let gateway = spend::total(
        &calls
            .iter()
            .map(|call| resolution.call_requests(*call))
            .collect::<Vec<_>>(),
    );
    let estimated: Vec<_> = calls
        .iter()
        .filter_map(|call| estimates.get(call).copied())
        .collect();
    let unique: BTreeMap<_, _> = estimated.iter().copied().collect();
    let sum = gateway.spend.unwrap_or_default() + unique.values().sum::<f64>();
    (
        Priced {
            spend: (sum.is_finite() && (gateway.spend.is_some() || !estimated.is_empty()))
                .then_some(sum),
            priced_calls: gateway.priced_calls + estimated.len() as u64,
        },
        estimated.len() as u64,
    )
}
