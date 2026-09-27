use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
    time::Duration,
};

use litellm_retention::{
    Bounds, PartitionInterval, PartitionPlan, PartitionStore, Policy, RetentionReport,
    RetentionStore, StatementBudget, Stop, Target, TargetReport, run,
};
use rstest::rstest;
use time::{OffsetDateTime, macros::datetime};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

const NOW: OffsetDateTime = datetime!(2026-09-26 12:00 UTC);
const HOUR: Duration = Duration::from_secs(3_600);
const DAY: Duration = Duration::from_secs(86_400);

#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error("injected failure")]
    Injected,
    #[error("statement timeout")]
    TimedOut,
}

#[derive(Debug)]
struct Call {
    target: Target,
    cutoff: OffsetDateTime,
    timeout: Duration,
    at: Instant,
}

#[derive(Default)]
struct Rows {
    latency: Duration,
    rows: Mutex<HashMap<Target, Vec<OffsetDateTime>>>,
    failures: Mutex<VecDeque<bool>>,
    calls: Mutex<Vec<Call>>,
}

impl Rows {
    fn with(
        latency: Duration,
        rows: impl IntoIterator<Item = (Target, Vec<OffsetDateTime>)>,
    ) -> Self {
        Self {
            latency,
            rows: Mutex::new(rows.into_iter().collect()),
            ..Self::default()
        }
    }

    fn failing(self, script: impl IntoIterator<Item = bool>) -> Self {
        *self.failures.lock().unwrap() = script.into_iter().collect();
        self
    }

    fn left(&self, target: Target) -> usize {
        self.rows.lock().unwrap().get(&target).map_or(0, Vec::len)
    }

    fn calls_for(&self, target: Target) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|call| call.target == target)
            .count()
    }
}

impl RetentionStore for Rows {
    type Error = Failure;

    async fn delete_batch(
        &self,
        target: Target,
        cutoff: OffsetDateTime,
        limit: u32,
        timeout: Duration,
    ) -> Result<u64, Failure> {
        self.calls.lock().unwrap().push(Call {
            target,
            cutoff,
            timeout,
            at: Instant::now(),
        });
        if self.latency > timeout {
            tokio::time::sleep(timeout).await;
            return Err(Failure::TimedOut);
        }
        tokio::time::sleep(self.latency).await;
        if self.failures.lock().unwrap().pop_front() == Some(true) {
            return Err(Failure::Injected);
        }
        let mut rows = self.rows.lock().unwrap();
        let Some(rows) = rows.get_mut(&target) else {
            return Ok(0);
        };
        let before = rows.len();
        let mut expired = 0;
        rows.retain(|at| {
            let delete = *at < cutoff && expired < limit;
            expired += u32::from(delete);
            !delete
        });
        Ok((before - rows.len()) as u64)
    }
}

#[derive(Default)]
struct Partitions {
    fails: bool,
    ensured: Mutex<Vec<(OffsetDateTime, PartitionInterval, u32)>>,
    dropped_before: Mutex<Vec<OffsetDateTime>>,
}

impl PartitionStore for Partitions {
    type Error = Failure;

    async fn ensure_ahead(
        &self,
        now: OffsetDateTime,
        interval: PartitionInterval,
        periods_ahead: u32,
        _: StatementBudget,
    ) -> Result<(), Failure> {
        self.ensured
            .lock()
            .unwrap()
            .push((now, interval, periods_ahead));
        if self.fails {
            Err(Failure::Injected)
        } else {
            Ok(())
        }
    }

    async fn drop_before(
        &self,
        cutoff: OffsetDateTime,
        _: StatementBudget,
    ) -> Result<Vec<String>, Failure> {
        self.dropped_before.lock().unwrap().push(cutoff);
        if self.fails {
            return Err(Failure::Injected);
        }
        Ok(vec!["LiteLLM_SpendLogs_p20260901".to_owned()])
    }
}

fn bounds() -> Bounds {
    Bounds {
        batch_size: 2,
        max_batches: 100,
        run_budget: Duration::from_secs(600),
        batch_timeout: Duration::from_secs(30),
        batch_pause: Duration::ZERO,
        max_consecutive_failures: 3,
        failure_backoff: Duration::from_millis(500),
    }
}

fn aged(age: Duration, count: usize) -> Vec<OffsetDateTime> {
    vec![NOW - age; count]
}

fn policy(target: Target, max_age: Duration) -> Policy {
    Policy { target, max_age }
}

fn report(target: Target, deleted: u64, stop: Stop) -> TargetReport {
    TargetReport {
        target,
        deleted,
        stop,
    }
}

async fn prune(rows: &Rows, policies: &[Policy], bounds: &Bounds) -> RetentionReport {
    run::<_, Partitions>(rows, None, policies, bounds, NOW, CancellationToken::new()).await
}

#[tokio::test(start_paused = true)]
async fn each_target_deletes_only_rows_older_than_its_own_max_age() {
    let rows = Rows::with(
        Duration::from_millis(10),
        [
            (
                Target::SpendLogs,
                [aged(2 * DAY, 5), aged(DAY, 1), aged(HOUR, 1)].concat(),
            ),
            (
                Target::HealthCheck,
                [aged(2 * HOUR, 2), aged(HOUR / 2, 1)].concat(),
            ),
        ],
    );

    let outcome = prune(
        &rows,
        &[
            policy(Target::SpendLogs, DAY),
            policy(Target::HealthCheck, HOUR),
        ],
        &bounds(),
    )
    .await;

    assert_eq!(
        outcome.targets,
        vec![
            report(Target::SpendLogs, 5, Stop::Exhausted),
            report(Target::HealthCheck, 2, Stop::Exhausted),
        ]
    );
    assert_eq!(
        (rows.left(Target::SpendLogs), rows.left(Target::HealthCheck)),
        (2, 1)
    );
    assert!(
        rows.calls
            .lock()
            .unwrap()
            .iter()
            .all(|call| match call.target {
                Target::SpendLogs => call.cutoff == NOW - DAY,
                _ => call.cutoff == NOW - HOUR,
            })
    );
    assert_eq!(
        rows.calls_for(Target::SpendLogs),
        3,
        "a short batch means the backlog is gone, so no empty batch follows it"
    );
}

#[tokio::test(start_paused = true)]
async fn a_backlog_larger_than_the_batch_cap_is_left_for_the_next_run() {
    let rows = Rows::with(Duration::ZERO, [(Target::SpendLogs, aged(DAY, 10))]);

    let outcome = prune(
        &rows,
        &[policy(Target::SpendLogs, HOUR)],
        &Bounds {
            max_batches: 2,
            ..bounds()
        },
    )
    .await;

    assert_eq!(
        outcome.targets,
        vec![report(Target::SpendLogs, 4, Stop::BatchCap)]
    );
    assert_eq!(rows.left(Target::SpendLogs), 6);
}

#[tokio::test(start_paused = true)]
async fn no_statement_runs_past_the_run_budget_and_a_batch_cut_by_it_is_not_a_failure() {
    let rows = Rows::with(
        Duration::from_secs(1),
        [(Target::SpendLogs, aged(DAY, 100))],
    );
    let started = Instant::now();
    let budget = Duration::from_millis(2_500);

    let outcome = prune(
        &rows,
        &[policy(Target::SpendLogs, HOUR)],
        &Bounds {
            run_budget: budget,
            max_consecutive_failures: 1,
            ..bounds()
        },
    )
    .await;

    assert_eq!(
        outcome.targets,
        vec![report(Target::SpendLogs, 4, Stop::BudgetSpent)]
    );
    let calls = rows.calls.lock().unwrap();
    assert!(
        calls
            .iter()
            .all(|call| call.at + call.timeout <= started + budget)
    );
    assert_eq!(
        calls.iter().map(|call| call.timeout).collect::<Vec<_>>(),
        [2_500, 1_500, 500].map(Duration::from_millis)
    );
}

#[tokio::test(start_paused = true)]
async fn the_per_statement_timeout_never_exceeds_the_batch_timeout() {
    let rows = Rows::with(Duration::ZERO, [(Target::SpendLogs, aged(DAY, 3))]);

    prune(&rows, &[policy(Target::SpendLogs, HOUR)], &bounds()).await;

    let calls = rows.calls.lock().unwrap();
    assert!(
        calls
            .iter()
            .all(|call| call.timeout == bounds().batch_timeout)
    );
}

#[tokio::test(start_paused = true)]
async fn a_large_backlog_on_one_target_cannot_starve_the_next() {
    let rows = Rows::with(
        Duration::from_secs(1),
        [
            (Target::SpendLogs, aged(DAY, 1_000)),
            (Target::HealthCheck, aged(DAY, 1_000)),
        ],
    );

    let outcome = prune(
        &rows,
        &[
            policy(Target::SpendLogs, HOUR),
            policy(Target::HealthCheck, HOUR),
        ],
        &Bounds {
            run_budget: Duration::from_secs(10),
            ..bounds()
        },
    )
    .await;

    assert_eq!(
        outcome.targets,
        vec![
            report(Target::SpendLogs, 10, Stop::BudgetSpent),
            report(Target::HealthCheck, 10, Stop::BudgetSpent),
        ]
    );
}

#[rstest]
#[case::three_in_a_row([true, true, true].to_vec(), report(Target::SpendLogs, 0, Stop::Aborted))]
#[case::a_success_resets_the_streak(
    [true, true, false, true, true, false].to_vec(),
    report(Target::SpendLogs, 3, Stop::Exhausted)
)]
#[tokio::test(start_paused = true)]
async fn only_consecutive_failures_abort_a_target(
    #[case] failures: Vec<bool>,
    #[case] expected: TargetReport,
) {
    let rows = Rows::with(Duration::ZERO, [(Target::SpendLogs, aged(DAY, 3))]).failing(failures);

    let outcome = prune(&rows, &[policy(Target::SpendLogs, HOUR)], &bounds()).await;

    assert_eq!(outcome.targets, vec![expected]);
}

#[tokio::test(start_paused = true)]
async fn a_failed_batch_is_retried_only_after_the_backoff() {
    let rows = Rows::with(Duration::ZERO, [(Target::SpendLogs, aged(DAY, 1))]).failing([true]);

    prune(&rows, &[policy(Target::SpendLogs, HOUR)], &bounds()).await;

    let calls = rows.calls.lock().unwrap();
    assert_eq!(calls[1].at - calls[0].at, bounds().failure_backoff);
}

#[tokio::test(start_paused = true)]
async fn cancelling_mid_batch_stops_the_run_and_skips_the_remaining_targets() {
    let rows = Rows::with(
        Duration::from_secs(1),
        [
            (Target::SpendLogs, aged(DAY, 100)),
            (Target::HealthCheck, aged(DAY, 100)),
        ],
    );
    let cancel = CancellationToken::new();
    let canceller = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        canceller.cancel();
    });

    let outcome = run::<_, Partitions>(
        &rows,
        None,
        &[
            policy(Target::SpendLogs, HOUR),
            policy(Target::HealthCheck, HOUR),
        ],
        &bounds(),
        NOW,
        cancel,
    )
    .await;

    assert_eq!(
        outcome.targets,
        vec![report(Target::SpendLogs, 2, Stop::Cancelled)]
    );
    assert_eq!(rows.calls_for(Target::HealthCheck), 0);
}

const PLAN: PartitionPlan = PartitionPlan {
    interval: PartitionInterval::Week,
    periods_ahead: 7,
};

#[rstest]
#[case::with_spend_log_retention(
    vec![policy(Target::SpendLogs, DAY)],
    vec![NOW - DAY],
    vec!["LiteLLM_SpendLogs_p20260901".to_owned()]
)]
#[case::without_spend_log_retention(vec![policy(Target::HealthCheck, DAY)], vec![], vec![])]
#[tokio::test(start_paused = true)]
async fn partitions_are_created_ahead_every_run_and_dropped_only_under_spend_log_retention(
    #[case] policies: Vec<Policy>,
    #[case] dropped_before: Vec<OffsetDateTime>,
    #[case] dropped: Vec<String>,
) {
    let partitions = Partitions::default();

    let outcome = run(
        &Rows::default(),
        Some((&partitions, PLAN)),
        &policies,
        &bounds(),
        NOW,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(
        *partitions.ensured.lock().unwrap(),
        [(NOW, PLAN.interval, PLAN.periods_ahead)]
    );
    assert_eq!(*partitions.dropped_before.lock().unwrap(), dropped_before);
    assert_eq!(outcome.dropped_partitions, dropped);
}

#[tokio::test(start_paused = true)]
async fn failing_partition_maintenance_still_deletes_expired_rows() {
    let rows = Rows::with(Duration::ZERO, [(Target::SpendLogs, aged(2 * DAY, 3))]);
    let partitions = Partitions {
        fails: true,
        ..Partitions::default()
    };

    let outcome = run(
        &rows,
        Some((&partitions, PLAN)),
        &[policy(Target::SpendLogs, DAY)],
        &bounds(),
        NOW,
        CancellationToken::new(),
    )
    .await;

    assert_eq!(
        outcome.targets,
        vec![report(Target::SpendLogs, 3, Stop::Exhausted)]
    );
    assert!(outcome.dropped_partitions.is_empty());
}
