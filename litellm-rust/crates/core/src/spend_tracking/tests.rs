use std::sync::Arc;
use std::time::Duration;

use tokio::time;

use super::types::*;
use super::worker::*;

fn make_spend_entry(request_id: &str, cost: f64) -> SpendEntry {
    SpendEntry {
        request_id: request_id.to_string(),
        call_type: "completion".to_string(),
        api_key: "hashed-key-123".to_string(),
        spend: cost,
        total_tokens: 150,
        prompt_tokens: 100,
        completion_tokens: 50,
        model: "gpt-4o".to_string(),
        user: Some("user-1".to_string()),
        team_id: Some("team-1".to_string()),
        organization_id: None,
        end_user: None,
        custom_llm_provider: Some("openai".to_string()),
        status: SpendStatus::Success,
    }
}

fn make_spend_update(entity_type: EntityType, entity_id: &str, cost: f64) -> SpendUpdateItem {
    SpendUpdateItem {
        entity_type,
        entity_id: entity_id.to_string(),
        cost,
    }
}

#[tokio::test]
async fn worker_flushes_on_batch_size() {
    let flush = MemoryFlush::new();
    let worker = SpendWorker::spawn(3, Duration::from_secs(10), flush.clone());

    worker.record_log(make_spend_entry("req-1", 0.01)).await;
    worker.record_log(make_spend_entry("req-2", 0.02)).await;
    worker.record_log(make_spend_entry("req-3", 0.03)).await;

    time::sleep(Duration::from_millis(50)).await;

    let batches = flush.get_batches().await;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].spend_logs.len(), 3);
}

#[tokio::test]
async fn worker_flushes_on_interval() {
    let flush = MemoryFlush::new();
    let worker = SpendWorker::spawn(1000, Duration::from_millis(50), flush.clone());

    worker.record_log(make_spend_entry("req-1", 0.01)).await;

    time::sleep(Duration::from_millis(100)).await;

    let batches = flush.get_batches().await;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].spend_logs.len(), 1);
}

#[tokio::test]
async fn worker_aggregates_updates_by_entity() {
    let flush = MemoryFlush::new();
    let worker = SpendWorker::spawn(100, Duration::from_millis(50), flush.clone());

    worker
        .record_update(make_spend_update(EntityType::Key, "key-1", 0.01))
        .await;
    worker
        .record_update(make_spend_update(EntityType::Key, "key-1", 0.02))
        .await;
    worker
        .record_update(make_spend_update(EntityType::Key, "key-2", 0.03))
        .await;
    worker
        .record_update(make_spend_update(EntityType::User, "user-1", 0.05))
        .await;

    time::sleep(Duration::from_millis(100)).await;

    let batches = flush.get_batches().await;
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    assert_eq!(batch.key_updates.len(), 2);
    assert!((batch.key_updates["key-1"] - 0.03).abs() < 1e-10);
    assert!((batch.key_updates["key-2"] - 0.03).abs() < 1e-10);
    assert_eq!(batch.user_updates.len(), 1);
    assert!((batch.user_updates["user-1"] - 0.05).abs() < 1e-10);
}

#[tokio::test]
async fn worker_handles_all_entity_types() {
    let flush = MemoryFlush::new();
    let worker = SpendWorker::spawn(100, Duration::from_millis(50), flush.clone());

    worker
        .record_update(make_spend_update(EntityType::Key, "k1", 1.0))
        .await;
    worker
        .record_update(make_spend_update(EntityType::User, "u1", 2.0))
        .await;
    worker
        .record_update(make_spend_update(EntityType::EndUser, "eu1", 3.0))
        .await;
    worker
        .record_update(make_spend_update(EntityType::Team, "t1", 4.0))
        .await;
    worker
        .record_update(make_spend_update(EntityType::TeamMember, "tm1", 5.0))
        .await;
    worker
        .record_update(make_spend_update(EntityType::Organization, "o1", 6.0))
        .await;
    worker
        .record_update(make_spend_update(EntityType::Tag, "tag1", 7.0))
        .await;
    worker
        .record_update(make_spend_update(EntityType::Agent, "a1", 8.0))
        .await;

    time::sleep(Duration::from_millis(100)).await;

    let batches = flush.get_batches().await;
    let batch = &batches[0];

    assert_eq!(batch.key_updates.len(), 1);
    assert_eq!(batch.user_updates.len(), 1);
    assert_eq!(batch.end_user_updates.len(), 1);
    assert_eq!(batch.team_updates.len(), 1);
    assert_eq!(batch.team_member_updates.len(), 1);
    assert_eq!(batch.org_updates.len(), 1);
    assert_eq!(batch.tag_updates.len(), 1);
    assert_eq!(batch.agent_updates.len(), 1);
}

#[tokio::test]
async fn worker_flushes_remaining_on_drop() {
    let flush = MemoryFlush::new();
    {
        let worker = SpendWorker::spawn(1000, Duration::from_secs(60), flush.clone());
        worker.record_log(make_spend_entry("req-1", 0.01)).await;
        worker.record_log(make_spend_entry("req-2", 0.02)).await;
    }

    time::sleep(Duration::from_millis(50)).await;

    let total = flush.total_entries().await;
    assert_eq!(total, 2);
}

#[tokio::test]
async fn null_flush_discards_data() {
    let worker = SpendWorker::spawn(100, Duration::from_millis(50), NullFlush);

    worker.record_log(make_spend_entry("req-1", 0.01)).await;
    worker
        .record_update(make_spend_update(EntityType::Key, "key-1", 0.01))
        .await;

    time::sleep(Duration::from_millis(100)).await;
}

#[derive(Clone)]
struct SlowFirstFlush {
    delay: Duration,
    stalled: Arc<std::sync::atomic::AtomicBool>,
    inner: MemoryFlush,
}

impl SpendFlush for SlowFirstFlush {
    async fn flush(&self, batch: SpendUpdateBatch) {
        if !self.stalled.swap(true, std::sync::atomic::Ordering::SeqCst) {
            time::sleep(self.delay).await;
        }
        self.inner.flush(batch).await;
    }
}

#[tokio::test]
async fn worker_applies_backpressure_instead_of_dropping_records() {
    let flush = SlowFirstFlush {
        delay: Duration::from_secs(2),
        stalled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        inner: MemoryFlush::new(),
    };
    // Channel capacity is 10_000; the stalled first flush guarantees it fills.
    let worker = SpendWorker::spawn(5, Duration::from_secs(60), flush.clone());

    let sender = tokio::spawn(async move {
        for i in 0..12_000u32 {
            worker
                .record_update(make_spend_update(
                    EntityType::Key,
                    &format!("key-{i}"),
                    0.001,
                ))
                .await;
        }
        drop(worker);
    });
    sender.await.unwrap();

    time::sleep(Duration::from_millis(200)).await;
    assert_eq!(flush.inner.total_entries().await, 12_000);
}

#[test]
fn spend_update_batch_is_empty_initially() {
    let batch = SpendUpdateBatch::new();
    assert!(batch.is_empty());
    assert_eq!(batch.total_entries(), 0);
}

#[test]
fn spend_update_batch_tracks_entries() {
    let mut batch = SpendUpdateBatch::new();
    batch.add_update(SpendUpdateItem {
        entity_type: EntityType::Key,
        entity_id: "k1".to_string(),
        cost: 0.01,
    });
    batch.add_spend_log(make_spend_entry("req-1", 0.01));

    assert!(!batch.is_empty());
    assert_eq!(batch.total_entries(), 2);
}

#[test]
fn spend_update_batch_aggregates_costs() {
    let mut batch = SpendUpdateBatch::new();
    batch.add_update(SpendUpdateItem {
        entity_type: EntityType::User,
        entity_id: "u1".to_string(),
        cost: 0.01,
    });
    batch.add_update(SpendUpdateItem {
        entity_type: EntityType::User,
        entity_id: "u1".to_string(),
        cost: 0.02,
    });

    assert_eq!(batch.user_updates.len(), 1);
    assert!((batch.user_updates["u1"] - 0.03).abs() < 1e-10);
}
