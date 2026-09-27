mod support;

use litellm_spend::EntityKey;
use litellm_spend_testing as contract;
use support::{BufferLedger, CLAIM_TTL, Database, Faults, buffer};

async fn expire() {
    tokio::time::advance(CLAIM_TTL + std::time::Duration::from_secs(1)).await;
}

#[tokio::test(start_paused = true)]
async fn memory_buffer_hides_a_claimed_batch_from_other_claimers() {
    contract::a_claimed_batch_is_hidden_from_other_claimers(&buffer(), "").await;
}

#[tokio::test(start_paused = true)]
async fn memory_buffer_returns_an_unacked_batch_unchanged_once_its_claim_expires() {
    contract::an_unacked_batch_returns_unchanged_once_its_claim_expires(&buffer(), "", expire)
        .await;
}

#[tokio::test(start_paused = true)]
async fn memory_buffer_returns_a_released_batch_at_once_with_its_id() {
    contract::a_released_batch_returns_at_once_with_its_id(&buffer(), "").await;
}

#[tokio::test(start_paused = true)]
async fn memory_buffer_forgets_an_acked_batch_and_tolerates_a_second_ack() {
    contract::an_acked_batch_is_gone_and_a_second_ack_is_harmless(&buffer(), "", expire).await;
}

#[tokio::test(start_paused = true)]
async fn memory_buffer_as_a_store_counts_a_batch_committed_twice_once() {
    contract::a_batch_committed_twice_counts_once(&BufferLedger(buffer()), "").await;
}

#[tokio::test(start_paused = true)]
async fn memory_buffer_as_a_store_counts_equal_batches_with_different_ids_twice() {
    contract::equal_batches_with_different_ids_both_count(&BufferLedger(buffer()), "").await;
}

#[tokio::test(start_paused = true)]
async fn simulated_database_counts_a_batch_committed_twice_once() {
    contract::a_batch_committed_twice_counts_once(&Database::default(), "").await;
}

#[tokio::test(start_paused = true)]
async fn simulated_database_counts_equal_batches_with_different_ids_twice() {
    contract::equal_batches_with_different_ids_both_count(&Database::default(), "").await;
}

#[tokio::test(start_paused = true)]
async fn simulated_database_leaves_a_refused_batch_unapplied() {
    let poison = EntityKey::Team("m".to_owned());
    let faults = Faults::default();
    *faults.poison.lock().unwrap() = Some(poison.clone());
    let database = Database::new(faults.clone());

    contract::a_refused_batch_changes_nothing(&database, "", poison, async || {
        *faults.poison.lock().unwrap() = None;
    })
    .await;
}
