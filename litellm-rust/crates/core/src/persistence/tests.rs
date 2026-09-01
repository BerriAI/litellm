use std::sync::Arc;
use tokio::sync::Mutex;

use super::*;
use crate::spend_tracking::SpendUpdateBatch;

#[test]
fn persistence_error_display() {
    let err = PersistenceError::Redis("connection refused".to_string());
    assert_eq!(err.to_string(), "Redis error: connection refused");

    let err = PersistenceError::Postgres("table not found".to_string());
    assert_eq!(err.to_string(), "Postgres error: table not found");

    let err = PersistenceError::Serialization("invalid JSON".to_string());
    assert_eq!(err.to_string(), "Serialization error: invalid JSON");

    let err = PersistenceError::Connection("timeout".to_string());
    assert_eq!(err.to_string(), "Connection error: timeout");
}

#[test]
fn spend_update_batch_has_correct_counts() {
    let mut batch = SpendUpdateBatch::new();
    assert!(batch.is_empty());
    assert_eq!(batch.total_entries(), 0);

    batch.key_updates.insert("key-1".to_string(), 0.01);
    batch.user_updates.insert("user-1".to_string(), 0.02);
    assert!(!batch.is_empty());
    assert_eq!(batch.total_entries(), 2);
}

/// Mock CacheStore for testing the trait contract.
struct MockCacheStore {
    data: Arc<Mutex<std::collections::HashMap<String, serde_json::Value>>>,
    counters: Arc<Mutex<std::collections::HashMap<String, f64>>>,
}

impl MockCacheStore {
    fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(std::collections::HashMap::new())),
            counters: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

impl CacheStore for MockCacheStore {
    async fn get(&self, key: &str) -> Result<Option<serde_json::Value>, PersistenceError> {
        Ok(self.data.lock().await.get(key).cloned())
    }

    async fn set(
        &self,
        key: &str,
        value: &serde_json::Value,
        _ttl_seconds: Option<u64>,
    ) -> Result<(), PersistenceError> {
        self.data
            .lock()
            .await
            .insert(key.to_string(), value.clone());
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<(), PersistenceError> {
        self.data.lock().await.remove(key);
        Ok(())
    }

    async fn incr_by_float(&self, key: &str, amount: f64) -> Result<f64, PersistenceError> {
        let mut counters = self.counters.lock().await;
        let entry = counters.entry(key.to_string()).or_insert(0.0);
        *entry += amount;
        Ok(*entry)
    }
}

/// Mock DatabaseStore for testing the trait contract.
struct MockDatabaseStore {
    spend_logs: Arc<Mutex<Vec<serde_json::Value>>>,
    entity_spend: Arc<Mutex<std::collections::HashMap<(String, String), f64>>>,
}

impl MockDatabaseStore {
    fn new() -> Self {
        Self {
            spend_logs: Arc::new(Mutex::new(Vec::new())),
            entity_spend: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

impl DatabaseStore for MockDatabaseStore {
    async fn insert_spend_log(&self, log: &serde_json::Value) -> Result<(), PersistenceError> {
        self.spend_logs.lock().await.push(log.clone());
        Ok(())
    }

    async fn batch_insert_spend_logs(
        &self,
        logs: &[serde_json::Value],
    ) -> Result<(), PersistenceError> {
        self.spend_logs.lock().await.extend(logs.iter().cloned());
        Ok(())
    }

    async fn update_entity_spend(
        &self,
        entity_type: &str,
        entity_id: &str,
        amount: f64,
    ) -> Result<(), PersistenceError> {
        let key = (entity_type.to_string(), entity_id.to_string());
        let mut spend = self.entity_spend.lock().await;
        let entry = spend.entry(key).or_insert(0.0);
        *entry += amount;
        Ok(())
    }

    async fn batch_update_entity_spend(
        &self,
        updates: &[(String, String, f64)],
    ) -> Result<(), PersistenceError> {
        for (entity_type, entity_id, amount) in updates {
            self.update_entity_spend(entity_type, entity_id, *amount)
                .await?;
        }
        Ok(())
    }
}

#[tokio::test]
async fn mock_cache_store_get_set_delete() {
    let store = MockCacheStore::new();

    assert!(store.get("missing").await.unwrap().is_none());

    store
        .set("key1", &serde_json::json!({"value": 42}), None)
        .await
        .unwrap();
    let val = store.get("key1").await.unwrap().unwrap();
    assert_eq!(val["value"], 42);

    store.delete("key1").await.unwrap();
    assert!(store.get("key1").await.unwrap().is_none());
}

#[tokio::test]
async fn mock_cache_store_incr_by_float() {
    let store = MockCacheStore::new();

    let val = store.incr_by_float("counter", 1.5).await.unwrap();
    assert!((val - 1.5).abs() < 1e-10);

    let val = store.incr_by_float("counter", 2.3).await.unwrap();
    assert!((val - 3.8).abs() < 1e-10);
}

#[tokio::test]
async fn mock_database_store_insert_spend_log() {
    let store = MockDatabaseStore::new();

    let log = serde_json::json!({"request_id": "req-1", "spend": 0.01});
    store.insert_spend_log(&log).await.unwrap();

    let logs = store.spend_logs.lock().await;
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0]["request_id"], "req-1");
}

#[tokio::test]
async fn mock_database_store_batch_insert_spend_logs() {
    let store = MockDatabaseStore::new();

    let logs = vec![
        serde_json::json!({"request_id": "req-1", "spend": 0.01}),
        serde_json::json!({"request_id": "req-2", "spend": 0.02}),
    ];
    store.batch_insert_spend_logs(&logs).await.unwrap();

    let stored = store.spend_logs.lock().await;
    assert_eq!(stored.len(), 2);
}

#[tokio::test]
async fn mock_database_store_update_entity_spend() {
    let store = MockDatabaseStore::new();

    store
        .update_entity_spend("key", "key-1", 0.05)
        .await
        .unwrap();
    store
        .update_entity_spend("key", "key-1", 0.03)
        .await
        .unwrap();

    let spend = store.entity_spend.lock().await;
    let total = spend[&("key".to_string(), "key-1".to_string())];
    assert!((total - 0.08).abs() < 1e-10);
}

#[tokio::test]
async fn mock_database_store_batch_update_entity_spend() {
    let store = MockDatabaseStore::new();

    let updates = vec![
        ("key".to_string(), "key-1".to_string(), 0.05),
        ("user".to_string(), "user-1".to_string(), 0.03),
        ("key".to_string(), "key-1".to_string(), 0.02),
    ];
    store.batch_update_entity_spend(&updates).await.unwrap();

    let spend = store.entity_spend.lock().await;
    assert!((spend[&("key".to_string(), "key-1".to_string())] - 0.07).abs() < 1e-10);
    assert!((spend[&("user".to_string(), "user-1".to_string())] - 0.03).abs() < 1e-10);
}

#[tokio::test]
async fn strip_provider_prefix_strips_known_prefixes() {
    use crate::cost_calculator::pricing::strip_provider_prefix;
    assert_eq!(
        strip_provider_prefix("anthropic/claude-3-opus"),
        "claude-3-opus"
    );
    assert_eq!(strip_provider_prefix("openai/gpt-4o"), "gpt-4o");
    assert_eq!(
        strip_provider_prefix("bedrock/anthropic.claude-3"),
        "anthropic.claude-3"
    );
    assert_eq!(strip_provider_prefix("azure/gpt-4"), "gpt-4");
    assert_eq!(strip_provider_prefix("gemini/gemini-pro"), "gemini-pro");
}

#[tokio::test]
async fn strip_provider_prefix_preserves_unknown_prefixes() {
    use crate::cost_calculator::pricing::strip_provider_prefix;
    assert_eq!(strip_provider_prefix("claude-3-opus"), "claude-3-opus");
    assert_eq!(strip_provider_prefix("gpt-4o"), "gpt-4o");
    assert_eq!(strip_provider_prefix("unknown/model"), "unknown/model");
}
