//! [`RouterCache`]: the DualCache-style store all routing state lives in, under the keys,
//! encodings and TTLs the Python router uses, so Python and Rust workers share it. Method names
//! follow `litellm-cache`'s `BaseCache`, `BatchCache`, `CounterCache` and `DeleteCache`.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use litellm_cache::Error;
use parking_lot::Mutex;
use serde_json::Value;

#[async_trait]
pub trait RouterCache: Send + Sync {
    async fn async_get_cache(&self, key: &str) -> Result<Option<Value>, Error>;

    async fn async_set_cache(
        &self,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), Error>;

    /// Add `amount` to the number at `key` (0 when missing) and return the new value. The TTL
    /// applies when the key is created, as Redis `INCRBYFLOAT` + `EXPIRE NX` does.
    async fn async_increment(
        &self,
        key: &str,
        amount: f64,
        ttl: Option<Duration>,
    ) -> Result<f64, Error>;

    async fn async_batch_get_cache(&self, keys: &[String]) -> Result<Vec<Option<Value>>, Error>;

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error>;
}

#[derive(Default)]
struct Entry {
    value: Value,
    expires: Option<Instant>,
}

/// An in-process [`RouterCache`], for tests and for state Python keeps `local_only`.
#[derive(Clone, Default)]
pub struct InMemoryRouterCache {
    entries: Arc<Mutex<HashMap<String, Entry>>>,
}

impl InMemoryRouterCache {
    fn live(entries: &mut HashMap<String, Entry>, key: &str) -> Option<Value> {
        let expired = entries
            .get(key)
            .is_some_and(|entry| entry.expires.is_some_and(|at| at <= Instant::now()));
        if expired {
            entries.remove(key);
        }
        entries.get(key).map(|entry| entry.value.clone())
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        Self::live(&mut self.entries.lock(), key)
    }

    pub fn increment(&self, key: &str, amount: f64, ttl: Option<Duration>) -> f64 {
        let mut entries = self.entries.lock();
        let current = Self::live(&mut entries, key).and_then(|value| value.as_f64());
        let next = current.unwrap_or(0.0) + amount;
        let value = serde_json::Number::from_f64(next)
            .map(|number| {
                if next.fract() == 0.0 && next.abs() < i64::MAX as f64 {
                    Value::from(next as i64)
                } else {
                    Value::Number(number)
                }
            })
            .unwrap_or(Value::Null);
        let expires = match (current, entries.get(key)) {
            (Some(_), Some(entry)) => entry.expires,
            _ => ttl.map(|ttl| Instant::now() + ttl),
        };
        entries.insert(key.to_owned(), Entry { value, expires });
        next
    }
}

#[async_trait]
impl RouterCache for InMemoryRouterCache {
    async fn async_get_cache(&self, key: &str) -> Result<Option<Value>, Error> {
        Ok(self.get(key))
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Value,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let expires = ttl.map(|ttl| Instant::now() + ttl);
        self.entries
            .lock()
            .insert(key.to_owned(), Entry { value, expires });
        Ok(())
    }

    async fn async_increment(
        &self,
        key: &str,
        amount: f64,
        ttl: Option<Duration>,
    ) -> Result<f64, Error> {
        Ok(self.increment(key, amount, ttl))
    }

    async fn async_batch_get_cache(&self, keys: &[String]) -> Result<Vec<Option<Value>>, Error> {
        let mut entries = self.entries.lock();
        Ok(keys
            .iter()
            .map(|key| Self::live(&mut entries, key))
            .collect())
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error> {
        self.entries.lock().remove(key);
        Ok(())
    }
}
