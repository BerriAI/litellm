//! Persistence layer for LiteLLM.
//!
//! Provides trait-based abstractions for cache and database operations,
//! with Redis and PostgreSQL implementations.

mod postgres_store;
mod redis_store;
#[cfg(test)]
mod tests;

use std::future::Future;

use serde_json::Value;

use crate::spend_tracking::SpendUpdateBatch;

pub use postgres_store::PostgresStore;
pub use redis_store::RedisStore;

/// Errors from persistence operations.
#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    #[error("Redis error: {0}")]
    Redis(String),
    #[error("Postgres error: {0}")]
    Postgres(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Connection error: {0}")]
    Connection(String),
}

/// Trait for key-value cache operations.
/// Implemented by Redis and in-memory caches.
pub trait CacheStore: Send + Sync + 'static {
    fn get(
        &self,
        key: &str,
    ) -> impl Future<Output = Result<Option<Value>, PersistenceError>> + Send;
    fn set(
        &self,
        key: &str,
        value: &Value,
        ttl_seconds: Option<u64>,
    ) -> impl Future<Output = Result<(), PersistenceError>> + Send;
    fn delete(&self, key: &str) -> impl Future<Output = Result<(), PersistenceError>> + Send;
    fn incr_by_float(
        &self,
        key: &str,
        amount: f64,
    ) -> impl Future<Output = Result<f64, PersistenceError>> + Send;
}

/// Trait for database operations.
/// Implemented by PostgreSQL (and potentially other SQL databases).
pub trait DatabaseStore: Send + Sync + 'static {
    fn insert_spend_log(
        &self,
        log: &Value,
    ) -> impl Future<Output = Result<(), PersistenceError>> + Send;
    fn batch_insert_spend_logs(
        &self,
        logs: &[Value],
    ) -> impl Future<Output = Result<(), PersistenceError>> + Send;
    fn update_entity_spend(
        &self,
        entity_type: &str,
        entity_id: &str,
        amount: f64,
    ) -> impl Future<Output = Result<(), PersistenceError>> + Send;
    fn batch_update_entity_spend(
        &self,
        updates: &[(String, String, f64)],
    ) -> impl Future<Output = Result<(), PersistenceError>> + Send;
}

/// A `SpendFlush` implementation that writes to both Redis and Postgres.
pub struct RedisPostgresSpendFlush {
    redis: RedisStore,
    postgres: PostgresStore,
}

impl RedisPostgresSpendFlush {
    pub fn new(redis: RedisStore, postgres: PostgresStore) -> Self {
        Self { redis, postgres }
    }
}

impl crate::spend_tracking::SpendFlush for RedisPostgresSpendFlush {
    async fn flush(&self, batch: SpendUpdateBatch) {
        for (entity_id, cost) in &batch.key_updates {
            let _ = self
                .redis
                .incr_by_float(&format!("spend:key:{entity_id}"), *cost)
                .await;
            let _ = self
                .postgres
                .update_entity_spend("key", entity_id, *cost)
                .await;
        }
        for (entity_id, cost) in &batch.user_updates {
            let _ = self
                .redis
                .incr_by_float(&format!("spend:user:{entity_id}"), *cost)
                .await;
            let _ = self
                .postgres
                .update_entity_spend("user", entity_id, *cost)
                .await;
        }
        for (entity_id, cost) in &batch.team_updates {
            let _ = self
                .redis
                .incr_by_float(&format!("spend:team:{entity_id}"), *cost)
                .await;
            let _ = self
                .postgres
                .update_entity_spend("team", entity_id, *cost)
                .await;
        }
        for (entity_id, cost) in &batch.org_updates {
            let _ = self
                .redis
                .incr_by_float(&format!("spend:org:{entity_id}"), *cost)
                .await;
            let _ = self
                .postgres
                .update_entity_spend("organization", entity_id, *cost)
                .await;
        }
        for (entity_id, cost) in &batch.end_user_updates {
            let _ = self
                .redis
                .incr_by_float(&format!("spend:end_user:{entity_id}"), *cost)
                .await;
            let _ = self
                .postgres
                .update_entity_spend("end_user", entity_id, *cost)
                .await;
        }
        for (entity_id, cost) in &batch.tag_updates {
            let _ = self
                .redis
                .incr_by_float(&format!("spend:tag:{entity_id}"), *cost)
                .await;
            let _ = self
                .postgres
                .update_entity_spend("tag", entity_id, *cost)
                .await;
        }
        for (entity_id, cost) in &batch.agent_updates {
            let _ = self
                .redis
                .incr_by_float(&format!("spend:agent:{entity_id}"), *cost)
                .await;
            let _ = self
                .postgres
                .update_entity_spend("agent", entity_id, *cost)
                .await;
        }
        for (entity_id, cost) in &batch.team_member_updates {
            let _ = self
                .redis
                .incr_by_float(&format!("spend:team_member:{entity_id}"), *cost)
                .await;
            let _ = self
                .postgres
                .update_entity_spend("team_member", entity_id, *cost)
                .await;
        }

        if !batch.spend_logs.is_empty() {
            let logs: Vec<Value> = batch
                .spend_logs
                .iter()
                .filter_map(|entry| serde_json::to_value(entry).ok())
                .collect();
            let _ = self.postgres.batch_insert_spend_logs(&logs).await;
        }
    }
}
