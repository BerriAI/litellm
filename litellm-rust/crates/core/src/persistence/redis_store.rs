use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use serde_json::Value;

use super::{CacheStore, PersistenceError};

/// Redis-backed cache store.
///
/// Uses `redis::aio::ConnectionManager` for automatic reconnection and
/// connection pooling.
pub struct RedisStore {
    client: ConnectionManager,
}

impl RedisStore {
    /// Connect to Redis at the given URL.
    ///
    /// URL format: `redis://[:password@]host[:port][/db]`
    pub async fn connect(url: &str) -> Result<Self, PersistenceError> {
        let client = redis::Client::open(url)
            .map_err(|e| PersistenceError::Connection(format!("invalid Redis URL: {e}")))?;
        let manager = ConnectionManager::new(client)
            .await
            .map_err(|e| PersistenceError::Connection(format!("Redis connection failed: {e}")))?;
        Ok(Self { client: manager })
    }

    /// Create from an existing connection manager (for testing/sharing).
    pub fn from_manager(manager: ConnectionManager) -> Self {
        Self { client: manager }
    }

    /// Clone the underlying connection manager (for sharing across components).
    pub fn clone_manager(&self) -> ConnectionManager {
        self.client.clone()
    }

    /// Ping Redis to verify connectivity.
    pub async fn ping(&self) -> Result<(), PersistenceError> {
        let mut conn = self.client.clone();
        let pong: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(|e| PersistenceError::Redis(format!("PING failed: {e}")))?;
        if pong == "PONG" {
            Ok(())
        } else {
            Err(PersistenceError::Redis(format!(
                "unexpected PING response: {pong}"
            )))
        }
    }

    /// Increment a counter and set TTL atomically (for rate limiting).
    pub async fn incr_with_ttl(
        &self,
        key: &str,
        amount: f64,
        ttl_seconds: u64,
    ) -> Result<f64, PersistenceError> {
        let mut conn = self.client.clone();

        // Use Lua script for atomic increment + TTL
        let script = redis::Script::new(
            r#"
            local current = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
            redis.call('EXPIRE', KEYS[1], ARGV[2])
            return current
            "#,
        );

        let result: f64 = script
            .key(key)
            .arg(amount)
            .arg(ttl_seconds)
            .invoke_async(&mut conn)
            .await
            .map_err(|e| PersistenceError::Redis(format!("INCR with TTL failed: {e}")))?;

        Ok(result)
    }
}

impl CacheStore for RedisStore {
    async fn get(&self, key: &str) -> Result<Option<Value>, PersistenceError> {
        let mut conn = self.client.clone();
        let result: Option<String> = conn
            .get(key)
            .await
            .map_err(|e| PersistenceError::Redis(format!("GET failed: {e}")))?;
        match result {
            Some(json_str) => {
                let value: Value = serde_json::from_str(&json_str).map_err(|e| {
                    PersistenceError::Serialization(format!("invalid JSON in Redis: {e}"))
                })?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    async fn set(
        &self,
        key: &str,
        value: &Value,
        ttl_seconds: Option<u64>,
    ) -> Result<(), PersistenceError> {
        let mut conn = self.client.clone();
        let json_str = serde_json::to_string(value)
            .map_err(|e| PersistenceError::Serialization(format!("failed to serialize: {e}")))?;

        if let Some(ttl) = ttl_seconds {
            conn.set_ex::<_, _, ()>(key, json_str, ttl)
                .await
                .map_err(|e| PersistenceError::Redis(format!("SET EX failed: {e}")))?;
        } else {
            conn.set::<_, _, ()>(key, json_str)
                .await
                .map_err(|e| PersistenceError::Redis(format!("SET failed: {e}")))?;
        }
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<(), PersistenceError> {
        let mut conn = self.client.clone();
        conn.del::<_, ()>(key)
            .await
            .map_err(|e| PersistenceError::Redis(format!("DEL failed: {e}")))?;
        Ok(())
    }

    async fn incr_by_float(&self, key: &str, amount: f64) -> Result<f64, PersistenceError> {
        let mut conn = self.client.clone();
        let result: f64 = conn
            .incr(key, amount)
            .await
            .map_err(|e| PersistenceError::Redis(format!("INCRBYFLOAT failed: {e}")))?;
        Ok(result)
    }
}
