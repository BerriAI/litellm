//! The standalone account delivery schedule and its last pass after the spend-log writer drains.

use std::time::Duration;

use litellm_config::Environment;
use litellm_env::Lookup;
use litellm_repositories::SpendLogRepository;
use tokio::time::Instant;

use crate::periodic::{self, PeriodicJob};

const PASS_DEADLINE: Duration = Duration::from_secs(5);

/// Settings used by each pass, including the last pass on shutdown.
#[derive(Clone, Copy)]
struct Settings {
    batch_size: i32,
    retention_days: i32,
    lock_timeout_ms: u64,
}

pub(crate) fn start(
    repository: SpendLogRepository,
    variables: &Environment,
    environment: &dyn Lookup,
) -> PeriodicJob {
    let positive = |name, fallback| {
        environment
            .get(name)
            .and_then(|raw| raw.parse::<i32>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(fallback)
    };
    let settings = Settings {
        batch_size: positive("LITELLM_ACCOUNTING_DELIVERY_BATCH_SIZE", 500),
        retention_days: positive("LITELLM_ACCOUNTING_DELIVERY_RETENTION_DAYS", 1),
        lock_timeout_ms: u64::try_from(positive("SPEND_ROLLUP_LOCK_TIMEOUT_MS", 5000))
            .unwrap_or(5000),
    };
    PeriodicJob::start(periodic::batch_write_interval(variables), move || {
        let repository = repository.clone();
        async move {
            if let Err(error) = flush(&repository, settings).await {
                tracing::error!(error = %litellm_error::Report::new(&error), "Accounting delivery pass failed");
            }
        }
    })
}

async fn flush(
    repository: &SpendLogRepository,
    settings: Settings,
) -> Result<(), litellm_db::Error> {
    let size = usize::try_from(settings.batch_size).unwrap_or(1);
    for retry in [false, true] {
        let deadline = Instant::now() + PASS_DEADLINE;
        for _ in 0..size {
            if Instant::now() >= deadline {
                break;
            }
            let batch = repository
                .apply_accounting_batch(settings.batch_size, retry, settings.lock_timeout_ms)
                .await?;
            if batch.parked != 0 {
                tracing::error!(
                    deliveries = batch.parked,
                    "Accounting deliveries parked, persisted spend may require repair"
                );
            }
            if if retry {
                batch.applied == 0 && batch.failed == 0
            } else {
                batch.applied + batch.parked < size
            } {
                break;
            }
        }
    }
    let limit = settings.batch_size.saturating_mul(2);
    let deadline = Instant::now() + PASS_DEADLINE;
    for _ in 0..size {
        if Instant::now() >= deadline
            || repository
                .cleanup_accounting_deliveries(settings.retention_days, limit)
                .await?
                < u64::try_from(limit).unwrap_or(1)
        {
            break;
        }
    }
    Ok(())
}
