use std::{collections::BTreeMap, future::Future, pin::Pin, time::Duration};

use litellm_traces::{
    TraceCostInput,
    query::named::{SpendByResponseIdsRow, TraceSpansRow},
    trace_cost_inputs,
};

use crate::{EstimateUnavailable, TraceReader};

pub type EstimateFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<Option<f64>>, EstimateUnavailable>> + Send + 'a>>;

pub trait TraceCostEstimator: Send + Sync {
    fn estimate(&self, calls: Vec<TraceCostInput>) -> EstimateFuture<'_>;
}

const BATCH_SIZE: usize = 128;
const ESTIMATE_TIMEOUT: Duration = Duration::from_secs(3);

fn bounded(input: &TraceCostInput) -> bool {
    input.attributes.len() <= 64
        && input
            .attributes
            .iter()
            .all(|(key, value)| key.chars().count() <= 128 && value.chars().count() <= 512)
}

pub(super) async fn estimates(
    reader: &TraceReader,
    runs: &[(&[TraceSpansRow], &[SpendByResponseIdsRow])],
) -> (Vec<BTreeMap<String, f64>>, bool) {
    let mut estimates = vec![BTreeMap::new(); runs.len()];
    let Some(estimator) = &reader.estimator else {
        return (estimates, false);
    };
    let calls: Vec<_> = runs
        .iter()
        .enumerate()
        .flat_map(|(run, (spans, spend))| {
            trace_cost_inputs(spans, spend)
                .into_iter()
                .filter(|(_, input)| bounded(input))
                .map(move |(spans, input)| (run, spans, input))
        })
        .collect();
    let result = tokio::time::timeout(ESTIMATE_TIMEOUT, async {
        for batch in calls.chunks(BATCH_SIZE) {
            let prices = estimator
                .estimate(batch.iter().map(|(_, _, input)| input.clone()).collect())
                .await?;
            if prices.len() != batch.len()
                || prices
                    .iter()
                    .flatten()
                    .any(|cost| !cost.is_finite() || *cost < 0.0)
            {
                return Err(EstimateUnavailable);
            }
            for ((run, spans, _), price) in batch.iter().zip(prices) {
                if let Some(price) = price {
                    for span in spans {
                        estimates[*run].insert(span.clone(), price);
                    }
                }
            }
        }
        Ok(())
    })
    .await;
    (estimates, !matches!(result, Ok(Ok(()))))
}
