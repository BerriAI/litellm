use std::time::Duration;

use time::OffsetDateTime;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::{
    Bounds, PartitionPlan, PartitionStore, Policy, RetentionStore, StatementBudget, Target,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Exhausted,
    BatchCap,
    BudgetSpent,
    Aborted,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetReport {
    pub target: Target,
    pub deleted: u64,
    pub stop: Stop,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RetentionReport {
    pub targets: Vec<TargetReport>,
    pub dropped_partitions: Vec<String>,
}

pub async fn run<R, P>(
    rows: &R,
    partitions: Option<(&P, PartitionPlan)>,
    policies: &[Policy],
    bounds: &Bounds,
    now: OffsetDateTime,
    cancel: CancellationToken,
) -> RetentionReport
where
    R: RetentionStore,
    P: PartitionStore,
{
    let deadline = Instant::now() + bounds.run_budget;
    let dropped_partitions = match partitions {
        Some((store, plan)) => {
            maintain_partitions(store, plan, policies, bounds, now, deadline, &cancel).await
        }
        None => Vec::new(),
    };
    let mut targets = Vec::with_capacity(policies.len());
    for (done, policy) in policies.iter().enumerate() {
        let budget =
            StatementBudget::new(share(deadline, policies.len() - done), bounds.batch_timeout);
        let report = prune(rows, policy, bounds, now, budget, &cancel).await;
        targets.push(report);
        if report.stop == Stop::Cancelled {
            break;
        }
    }
    RetentionReport {
        targets,
        dropped_partitions,
    }
}

fn share(deadline: Instant, remaining: usize) -> Instant {
    let now = Instant::now();
    if remaining <= 1 || now >= deadline {
        return deadline;
    }
    now + (deadline - now) / remaining as u32
}

fn cutoff(now: OffsetDateTime, max_age: Duration) -> Option<OffsetDateTime> {
    now.checked_sub(time::Duration::try_from(max_age).ok()?)
}

async fn maintain_partitions<P: PartitionStore>(
    store: &P,
    plan: PartitionPlan,
    policies: &[Policy],
    bounds: &Bounds,
    now: OffsetDateTime,
    deadline: Instant,
    cancel: &CancellationToken,
) -> Vec<String> {
    let budget = StatementBudget::new(deadline, bounds.batch_timeout);
    let ensured = cancel
        .run_until_cancelled(store.ensure_ahead(now, plan.interval, plan.periods_ahead, budget))
        .await;
    if let Some(Err(error)) = ensured {
        tracing::warn!(%error, "creating upcoming spend log partitions failed; the next run retries");
    }
    let Some(cutoff) = policies
        .iter()
        .find(|policy| policy.target == Target::SpendLogs)
        .and_then(|policy| cutoff(now, policy.max_age))
    else {
        return Vec::new();
    };
    match cancel
        .run_until_cancelled(store.drop_before(cutoff, budget))
        .await
    {
        Some(Ok(dropped)) => dropped,
        Some(Err(error)) => {
            tracing::warn!(%error, "dropping expired spend log partitions failed; the next run retries");
            Vec::new()
        }
        None => Vec::new(),
    }
}

async fn prune<R: RetentionStore>(
    rows: &R,
    policy: &Policy,
    bounds: &Bounds,
    now: OffsetDateTime,
    budget: StatementBudget,
    cancel: &CancellationToken,
) -> TargetReport {
    let report = |deleted, stop| TargetReport {
        target: policy.target,
        deleted,
        stop,
    };
    let Some(cutoff) = cutoff(now, policy.max_age) else {
        return report(0, Stop::Exhausted);
    };
    let (mut deleted, mut batches, mut failures) = (0, 0, 0);
    loop {
        if batches >= bounds.max_batches {
            return report(deleted, Stop::BatchCap);
        }
        let Some(timeout) = budget.timeout() else {
            return report(deleted, Stop::BudgetSpent);
        };
        let attempt = rows.delete_batch(policy.target, cutoff, bounds.batch_size, timeout);
        let Some(outcome) = cancel.run_until_cancelled(attempt).await else {
            return report(deleted, Stop::Cancelled);
        };
        let pause = match outcome {
            Ok(count) => {
                deleted += count;
                batches += 1;
                failures = 0;
                if count < u64::from(bounds.batch_size) {
                    return report(deleted, Stop::Exhausted);
                }
                bounds.batch_pause
            }
            Err(_) if budget.timeout().is_none() => return report(deleted, Stop::BudgetSpent),
            Err(error) => {
                failures += 1;
                tracing::warn!(target = ?policy.target, failures, %error, "retention batch failed");
                if failures >= bounds.max_consecutive_failures {
                    return report(deleted, Stop::Aborted);
                }
                bounds.failure_backoff
            }
        };
        if cancel
            .run_until_cancelled(tokio::time::sleep_until(
                (Instant::now() + pause).min(budget.deadline()),
            ))
            .await
            .is_none()
        {
            return report(deleted, Stop::Cancelled);
        }
    }
}
