use crate::control::Control;
use http::Method;
use litellm_traces::TraceCostInput;
use litellm_traces_cache::{EstimateFuture, EstimateUnavailable, TraceCostEstimator};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::Semaphore;

const TIMEOUT: Duration = Duration::from_secs(2);

pub struct GatewayTraceCosts {
    control: Control,
    slots: Semaphore,
}

impl GatewayTraceCosts {
    pub fn new(control: Control) -> Self {
        Self {
            control,
            slots: Semaphore::new(4),
        }
    }
}

#[derive(Serialize)]
struct Request {
    calls: Vec<TraceCostInput>,
}

#[derive(Deserialize)]
struct Response {
    costs: Vec<Option<f64>>,
}

impl TraceCostEstimator for GatewayTraceCosts {
    fn estimate(&self, calls: Vec<TraceCostInput>) -> EstimateFuture<'_> {
        Box::pin(async move {
            tokio::time::timeout(TIMEOUT, async {
                let _permit = self
                    .slots
                    .acquire()
                    .await
                    .map_err(|_| EstimateUnavailable)?;
                let url = self
                    .control
                    .url("lens/internal/trace-costs")
                    .map_err(|_| EstimateUnavailable)?;
                let response: Response = self
                    .control
                    .request(Method::POST, url, Some(&Request { calls }), TIMEOUT)
                    .await
                    .map_err(|_| EstimateUnavailable)?;
                Ok(response.costs)
            })
            .await
            .map_err(|_| EstimateUnavailable)?
        })
    }
}
