#![forbid(unsafe_code)]

use std::fmt;

use gwbench_core::{LatencySummary, Scenario};
use hdrhistogram::Histogram;

#[derive(Debug)]
pub struct DriverError {
    message: String,
}

impl fmt::Display for DriverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for DriverError {}

/// Runs a stub open-loop constant-arrival-rate engine against `target`.
pub async fn run(_scenario: &Scenario, _target: &str) -> Result<LatencySummary, DriverError> {
    let histogram = Histogram::<u64>::new(3).map_err(|error| DriverError {
        message: error.to_string(),
    })?;
    Ok(LatencySummary::from_histogram(&histogram))
}
