use litellm_model_catalog_refresh::{AfterRetryable, after_retryable};
use pyo3::intern;
use pyo3::prelude::*;

/// Seconds to wait before the next attempt, or `None` to give up. `rng.uniform(0.0, 1.0)` is drawn only for a backoff,
/// as main draws it
#[pyfunction]
pub(crate) fn retry_wait_seconds(
    attempt: i64,
    max_attempts: i64,
    retry_after_seconds: Option<f64>,
    rng: &Bound<'_, PyAny>,
) -> PyResult<Option<f64>> {
    let mut failed = None;
    let after = after_retryable(attempt, max_attempts, retry_after_seconds, || {
        rng.call_method1(intern!(rng.py(), "uniform"), (0.0, 1.0))
            .and_then(|jitter| jitter.extract::<f64>())
            .unwrap_or_else(|error| {
                failed = Some(error);
                0.0
            })
    });
    if let Some(error) = failed {
        return Err(error);
    }
    Ok(match after {
        AfterRetryable::GiveUp => None,
        AfterRetryable::Wait { seconds } => Some(seconds),
    })
}
