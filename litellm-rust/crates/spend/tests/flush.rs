mod support;

use std::sync::atomic::Ordering;

use litellm_spend::{
    Batch, BatchId, Buffer, Claimed, Cost, EntityKey, FlushError, FlushOutcome, Totals, flush,
};
use rstest::rstest;
use support::{CLAIM_TTL, Database, Faults, Injected, buffer};

struct StuckBuffer {
    release_fails: bool,
    ack_fails: bool,
}

impl Buffer<EntityKey, Cost> for StuckBuffer {
    type Error = Injected;

    async fn push(&self, _: Batch<EntityKey, Cost>) -> Result<(), Injected> {
        Err(Injected)
    }

    async fn claim(&self, _: usize) -> Result<Option<Claimed<EntityKey, Cost>>, Injected> {
        Ok(Some(Claimed::from_buffer(
            BatchId::random(),
            batch(&[("a", 0.25)]),
        )))
    }

    async fn ack(&self, _: Claimed<EntityKey, Cost>) -> Result<(), Injected> {
        if self.ack_fails {
            Err(Injected)
        } else {
            Ok(())
        }
    }

    async fn release(&self, _: Claimed<EntityKey, Cost>) -> Result<(), Injected> {
        if self.release_fails {
            Err(Injected)
        } else {
            Ok(())
        }
    }
}

fn batch(entries: &[(&str, f64)]) -> Totals {
    Totals::from_entries(
        entries
            .iter()
            .map(|(id, cost)| (EntityKey::Key((*id).to_owned()), Cost(*cost))),
    )
}

fn total(database: &Database, id: &str) -> Option<f64> {
    database
        .totals()
        .get(&EntityKey::Key(id.to_owned()))
        .copied()
}

fn failing_every_commit(after_applying: bool) -> Database {
    Database::new(Faults {
        fail_without_applying_every: (!after_applying).then_some(1),
        fail_after_applying_every: after_applying.then_some(1),
        ..Faults::default()
    })
}

#[tokio::test(start_paused = true)]
async fn an_empty_buffer_commits_nothing() {
    let database = failing_every_commit(false);

    let outcome = flush(&buffer(), &database, 10).await;

    assert!(matches!(outcome, Ok(FlushOutcome::Empty)));
    assert_eq!(database.failed_without_applying.load(Ordering::SeqCst), 0);
}

#[tokio::test(start_paused = true)]
async fn a_commit_moves_at_most_max_entries_out_of_the_buffer() {
    let buffer = buffer();
    buffer
        .push(batch(&[("a", 0.25), ("b", 0.5), ("c", 1.0)]))
        .await
        .unwrap();
    let database = Database::default();

    let outcome = flush(&buffer, &database, 2).await;

    assert!(matches!(
        outcome,
        Ok(FlushOutcome::Committed { entries: 2 })
    ));
    assert_eq!(database.totals().len(), 2);
    assert_eq!(
        buffer
            .claim(usize::MAX)
            .await
            .unwrap()
            .unwrap()
            .batch()
            .len(),
        1
    );
}

#[rstest]
#[case::nothing_landed(false, None)]
#[case::it_landed_but_reported_an_error(true, Some(0.25))]
#[tokio::test(start_paused = true)]
async fn a_failed_commit_is_retried_by_the_next_flush_and_counts_once(
    #[case] after_applying: bool,
    #[case] total_after_the_failure: Option<f64>,
) {
    let buffer = buffer();
    buffer.push(batch(&[("a", 0.25)])).await.unwrap();
    let failing = failing_every_commit(after_applying);
    let healthy = Database::default();

    let outcome = flush(&buffer, &failing, 10).await;
    buffer.push(batch(&[("a", 0.5)])).await.unwrap();

    assert!(matches!(outcome, Err(FlushError::Commit(_))));
    assert_eq!(total(&failing, "a"), total_after_the_failure);
    assert!(matches!(
        flush(&buffer, &failing, 10).await,
        Err(FlushError::Commit(_))
    ));
    assert_eq!(total(&failing, "a"), total_after_the_failure);
    flush(&buffer, &healthy, 10).await.unwrap();
    assert_eq!(total(&healthy, "a"), Some(0.25));
}

#[tokio::test(start_paused = true)]
async fn a_flush_abandoned_after_the_commit_landed_is_redelivered_and_skipped() {
    let buffer = buffer();
    buffer.push(batch(&[("a", 0.25)])).await.unwrap();
    let database = Database::new(Faults {
        latency: std::time::Duration::from_secs(2),
        ..Faults::default()
    });

    let crashed = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        flush(&buffer, &database, 10),
    )
    .await;
    assert!(crashed.is_err());
    assert!(matches!(
        flush(&buffer, &database, 10).await,
        Ok(FlushOutcome::Empty)
    ));
    tokio::time::advance(CLAIM_TTL).await;
    let retried = flush(&buffer, &database, 10).await;

    assert!(matches!(
        retried,
        Ok(FlushOutcome::Committed { entries: 1 })
    ));
    assert_eq!(total(&database, "a"), Some(0.25));
    assert_eq!(database.repeats_skipped.load(Ordering::SeqCst), 1);
}

#[rstest]
#[case::release_fails(StuckBuffer { release_fails: true, ack_fails: false }, true)]
#[case::ack_fails(StuckBuffer { release_fails: false, ack_fails: true }, false)]
#[tokio::test(start_paused = true)]
async fn a_buffer_failure_after_the_commit_attempt_is_reported_as_its_own_error(
    #[case] stuck: StuckBuffer,
    #[case] commit_fails: bool,
) {
    let database = if commit_fails {
        failing_every_commit(false)
    } else {
        Database::default()
    };

    let outcome = flush(&stuck, &database, 10).await;

    assert_eq!(
        (
            matches!(outcome, Err(FlushError::Unreleased { .. })),
            matches!(outcome, Err(FlushError::Ack(_)))
        ),
        (commit_fails, !commit_fails)
    );
}
