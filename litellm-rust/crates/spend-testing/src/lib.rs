//! Backend-neutral contract checks every spend `Buffer` and `Store` backend runs from its
//! own `rstest` suite.
//!
//! Each check takes the backend under test and a `prefix` that keeps runs apart on shared
//! servers. Checks that wait for a claim to expire take `expire`, which the backend's suite
//! implements as a clock advance or a real sleep. A check panics with the violated contract.

use std::future::Future;

use litellm_spend::{BatchId, Buffer, Claimed, Cost, EntityKey, Store, Totals};

pub trait Ledger: Store<EntityKey, Cost> {
    fn total(&self, entity: &EntityKey) -> impl Future<Output = Option<Cost>> + Send;
}

fn team(prefix: &str, name: &str) -> EntityKey {
    EntityKey::Team(format!("{prefix}{name}"))
}

fn batch(prefix: &str, entries: &[(&str, f64)]) -> Totals {
    Totals::from_entries(
        entries
            .iter()
            .map(|(name, cost)| (team(prefix, name), Cost(*cost))),
    )
}

fn claimed(prefix: &str, entries: &[(&str, f64)]) -> Claimed<EntityKey, Cost> {
    Claimed::from_buffer(BatchId::random(), batch(prefix, entries))
}

async fn claim<B: Buffer<EntityKey, Cost>>(buffer: &B) -> Option<Claimed<EntityKey, Cost>> {
    buffer.claim(usize::MAX).await.expect("claim must succeed")
}

/// Committing one claimed batch again, as a retry after an unknown outcome does, counts once.
pub async fn a_batch_committed_twice_counts_once<L: Ledger>(ledger: &L, prefix: &str) {
    let retried = claimed(prefix, &[("a", 0.25), ("b", 0.5)]);

    ledger
        .commit(&retried)
        .await
        .expect("first commit must succeed");
    ledger
        .commit(&retried)
        .await
        .expect("a repeated commit must succeed");

    assert_eq!(ledger.total(&team(prefix, "a")).await, Some(Cost(0.25)));
    assert_eq!(ledger.total(&team(prefix, "b")).await, Some(Cost(0.5)));
}

/// Two batches with equal contents are two charges: only the id makes a commit a repeat.
pub async fn equal_batches_with_different_ids_both_count<L: Ledger>(ledger: &L, prefix: &str) {
    for _ in 0..2 {
        ledger
            .commit(&claimed(prefix, &[("a", 0.25)]))
            .await
            .expect("commit must succeed");
    }

    assert_eq!(ledger.total(&team(prefix, "a")).await, Some(Cost(0.5)));
}

/// A batch the store refuses leaves every entity in it untouched, and can land whole later.
///
/// `poison` is an entity the backend under test is arranged to reject until `cure` runs.
pub async fn a_refused_batch_changes_nothing<L, F>(
    ledger: &L,
    prefix: &str,
    poison: EntityKey,
    cure: impl FnOnce() -> F,
) where
    L: Ledger,
    F: Future<Output = ()>,
{
    let refused = Claimed::from_buffer(
        BatchId::random(),
        batch(prefix, &[("a", 0.25), ("z", 0.5)])
            .merge(Totals::from_entries([(poison.clone(), Cost(1.0))])),
    );

    ledger
        .commit(&refused)
        .await
        .expect_err("a batch holding the poison entity must be refused");
    assert_eq!(ledger.total(&team(prefix, "a")).await, None);
    assert_eq!(ledger.total(&team(prefix, "z")).await, None);

    cure().await;
    ledger
        .commit(&refused)
        .await
        .expect("the cured batch must land");
    assert_eq!(ledger.total(&team(prefix, "a")).await, Some(Cost(0.25)));
    assert_eq!(ledger.total(&team(prefix, "z")).await, Some(Cost(0.5)));
    assert_eq!(ledger.total(&poison).await, Some(Cost(1.0)));
}

/// While a claim is live nobody else can claim the same batch, even concurrently.
pub async fn a_claimed_batch_is_hidden_from_other_claimers<B>(buffer: &B, prefix: &str)
where
    B: Buffer<EntityKey, Cost>,
{
    buffer
        .push(batch(prefix, &[("a", 0.25)]))
        .await
        .expect("push must succeed");

    let (first, second) = tokio::join!(claim(buffer), claim(buffer));

    assert!(
        first.is_some() != second.is_some(),
        "exactly one claimer must get the batch"
    );
    assert_eq!(claim(buffer).await, None);
}

/// A claimer that dies without acking loses nothing: the batch returns with its id and contents.
pub async fn an_unacked_batch_returns_unchanged_once_its_claim_expires<B, F>(
    buffer: &B,
    prefix: &str,
    expire: impl FnOnce() -> F,
) where
    B: Buffer<EntityKey, Cost>,
    F: Future<Output = ()>,
{
    buffer
        .push(batch(prefix, &[("a", 0.25)]))
        .await
        .expect("push must succeed");
    let abandoned = claim(buffer)
        .await
        .expect("the pushed batch must be claimable");
    buffer
        .push(batch(prefix, &[("a", 0.5), ("b", 1.0)]))
        .await
        .expect("push must succeed");

    expire().await;
    let redelivered = claim(buffer)
        .await
        .expect("the abandoned batch must return");

    assert_eq!(redelivered, abandoned);
    assert_eq!(
        claim(buffer).await.map(|later| later.batch().clone()),
        Some(batch(prefix, &[("a", 0.5), ("b", 1.0)])),
        "entries pushed after a claim must form their own batch"
    );
}

/// A released batch is claimable at once, under the id it was first claimed with.
pub async fn a_released_batch_returns_at_once_with_its_id<B>(buffer: &B, prefix: &str)
where
    B: Buffer<EntityKey, Cost>,
{
    buffer
        .push(batch(prefix, &[("a", 0.25)]))
        .await
        .expect("push must succeed");
    let first = claim(buffer)
        .await
        .expect("the pushed batch must be claimable");
    let (id, contents) = (first.id(), first.batch().clone());

    buffer.release(first).await.expect("release must succeed");
    let again = claim(buffer)
        .await
        .expect("a released batch must be claimable");

    assert_eq!((again.id(), again.batch()), (id, &contents));
}

/// An acked batch never returns, and acking it again after a redelivery is harmless.
pub async fn an_acked_batch_is_gone_and_a_second_ack_is_harmless<B, F>(
    buffer: &B,
    prefix: &str,
    expire: impl Fn() -> F,
) where
    B: Buffer<EntityKey, Cost>,
    F: Future<Output = ()>,
{
    buffer
        .push(batch(prefix, &[("a", 0.25)]))
        .await
        .expect("push must succeed");
    let slow = claim(buffer)
        .await
        .expect("the pushed batch must be claimable");
    expire().await;
    let takeover = claim(buffer)
        .await
        .expect("the expired claim must be redelivered");
    buffer
        .push(batch(prefix, &[("b", 0.5)]))
        .await
        .expect("push must succeed");

    buffer.ack(takeover).await.expect("ack must succeed");
    buffer.ack(slow).await.expect("a second ack must succeed");

    assert_eq!(
        claim(buffer).await.map(|next| next.batch().clone()),
        Some(batch(prefix, &[("b", 0.5)])),
        "a second ack must not remove another batch"
    );
    expire().await;
    assert_eq!(
        claim(buffer).await.map(|next| next.batch().clone()),
        Some(batch(prefix, &[("b", 0.5)])),
        "only the unacked batch may return after every claim has expired"
    );
    assert_eq!(claim(buffer).await, None);
}
