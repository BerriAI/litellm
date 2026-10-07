use std::{collections::BTreeMap, future::Future, pin::Pin};

use litellm_traces::{
    Trace, estimate_candidates,
    query::named::{SpendByResponseIdsRow, TraceSpansRow},
    resolve_trace_with_estimates,
};

use crate::TraceReader;

pub type CostEstimates = Pin<Box<dyn Future<Output = Vec<Option<f64>>> + Send>>;

pub trait CostEstimator: Send + Sync {
    fn estimate(&self, attributes: Vec<BTreeMap<String, String>>) -> CostEstimates;
}

pub struct CatalogCostEstimator;

impl CostEstimator for CatalogCostEstimator {
    fn estimate(&self, attributes: Vec<BTreeMap<String, String>>) -> CostEstimates {
        Box::pin(async move {
            let catalog = litellm_model_catalog::bundled_pricing_catalog().ok();
            attributes
                .iter()
                .map(|attributes| {
                    catalog.and_then(|catalog| litellm_traces::estimate_cost(attributes, catalog))
                })
                .collect()
        })
    }
}

pub(super) struct ResolutionInput<'a> {
    pub trace_id: &'a str,
    pub trace_ref: &'a str,
    pub rows: &'a [TraceSpansRow],
    pub spend: &'a [SpendByResponseIdsRow],
}

impl TraceReader {
    pub(super) async fn resolve(
        &self,
        trace_id: &str,
        trace_ref: &str,
        rows: &[TraceSpansRow],
        spend: &[SpendByResponseIdsRow],
    ) -> Option<Trace> {
        self.resolve_batch(&[ResolutionInput {
            trace_id,
            trace_ref,
            rows,
            spend,
        }])
        .await
        .pop()
        .flatten()
    }

    pub(super) async fn resolve_batch(&self, runs: &[ResolutionInput<'_>]) -> Vec<Option<Trace>> {
        let candidates: Vec<_> = runs
            .iter()
            .map(|run| estimate_candidates(run.rows, run.spend))
            .collect();
        let attributes: Vec<_> = candidates
            .iter()
            .flatten()
            .map(|row| {
                let mut attributes = row.pricing_attributes.clone();
                attributes.insert("litellm.trace.start_ns".into(), row.start_ns.to_string());
                attributes
            })
            .collect();
        let prices = match &self.estimator {
            Some(estimator) if !attributes.is_empty() => estimator.estimate(attributes).await,
            _ => Vec::new(),
        };
        let valid = prices.len() == candidates.iter().map(Vec::len).sum::<usize>();
        let mut prices = prices.into_iter();
        runs.iter()
            .zip(candidates)
            .map(|(run, candidates)| {
                let estimates = if valid {
                    candidates
                        .iter()
                        .zip(prices.by_ref().take(candidates.len()))
                        .filter_map(|(row, price)| price.map(|price| (row.span_id.clone(), price)))
                        .collect()
                } else {
                    BTreeMap::new()
                };
                resolve_trace_with_estimates(
                    run.trace_id,
                    run.trace_ref,
                    run.rows,
                    run.spend,
                    &estimates,
                )
            })
            .collect()
    }
}
