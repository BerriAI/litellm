//! [`RouterCache`]: the routing state router-core reads and writes, named by what it is (a
//! deployment's cooldown, its allowed-fails count, its per-minute usage, the health checks'
//! states). Where that state lives in Redis, and under which keys, is the host's store's business:
//! router-core never spells a Redis key.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use litellm_cache::Error;
use parking_lot::Mutex;
use serde_json::Value;

/// One deployment's per-minute usage counters: `model` is the model its usage is counted under,
/// and `minute` is `HH-MM` in UTC.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UsageWindow {
    pub deployment_id: String,
    pub model: String,
    pub minute: String,
}

/// A [`UsageWindow`]'s counts, `None` for a counter nothing has written yet.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UsageCounts {
    pub tokens: Option<f64>,
    pub requests: Option<f64>,
}

#[async_trait]
pub trait RouterCache: Send + Sync {
    /// The stored cooldown of each of `ids`, in order, `None` for one without.
    async fn cooldowns(&self, ids: &[String]) -> Result<Vec<Option<Value>>, Error>;

    async fn put_cooldown(&self, id: &str, cooldown: Value, ttl: Duration) -> Result<(), Error>;

    /// Count one more failure against `id`'s allowed fails (per `suffix` when one is given) and
    /// return the count. `ttl` applies when the counter is created.
    async fn increment_allowed_fails(
        &self,
        id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> Result<f64, Error>;

    /// `_increment_deployment_usage`'s write: tokens and requests in one round trip.
    async fn add_usage(
        &self,
        window: &UsageWindow,
        tokens: f64,
        requests: f64,
    ) -> Result<(), Error>;

    /// One more request on `window`, whose counter expires `ttl` after it is created.
    async fn add_request(&self, window: &UsageWindow, ttl: Duration) -> Result<(), Error>;

    async fn usage(&self, window: &UsageWindow) -> Result<UsageCounts, Error>;

    /// The background health checks' per-deployment states, as `DeploymentHealthCache` keeps them.
    async fn health_states(&self) -> Result<Option<Value>, Error>;

    /// Read this worker's in-memory tier only (`local_only=True`).
    async fn get_local(&self, key: &str) -> Result<Option<Value>, Error>;

    /// Add to a counter in this worker's in-memory tier only (`local_only=True`).
    async fn increment_local(
        &self,
        key: &str,
        amount: f64,
        ttl: Option<Duration>,
    ) -> Result<f64, Error>;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Slot {
    Cooldown(String),
    AllowedFails(String, Option<String>),
    Tokens(UsageWindow),
    Requests(UsageWindow),
    HealthStates,
    Local(String),
}

struct Entry {
    value: Value,
    expires: Option<Instant>,
}

/// An in-process [`RouterCache`], for tests.
#[derive(Clone, Default)]
pub struct InMemoryRouterCache {
    entries: Arc<Mutex<HashMap<Slot, Entry>>>,
}

impl InMemoryRouterCache {
    fn live(entries: &mut HashMap<Slot, Entry>, slot: &Slot) -> Option<Value> {
        let expired = entries
            .get(slot)
            .is_some_and(|entry| entry.expires.is_some_and(|at| at <= Instant::now()));
        if expired {
            entries.remove(slot);
        }
        entries.get(slot).map(|entry| entry.value.clone())
    }

    fn get(&self, slot: &Slot) -> Option<Value> {
        Self::live(&mut self.entries.lock(), slot)
    }

    fn set(&self, slot: Slot, value: Value, ttl: Option<Duration>) {
        let expires = ttl.map(|ttl| Instant::now() + ttl);
        self.entries.lock().insert(slot, Entry { value, expires });
    }

    fn increment(&self, slot: Slot, amount: f64, ttl: Option<Duration>) -> f64 {
        let mut entries = self.entries.lock();
        let current = Self::live(&mut entries, &slot).and_then(|value| value.as_f64());
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
        let expires = match (current, entries.get(&slot)) {
            (Some(_), Some(entry)) => entry.expires,
            _ => ttl.map(|ttl| Instant::now() + ttl),
        };
        entries.insert(slot, Entry { value, expires });
        next
    }

    /// `window`'s counts as stored now.
    pub fn counted(&self, window: &UsageWindow) -> UsageCounts {
        let count = |slot: Slot| self.get(&slot).and_then(|value| value.as_f64());
        UsageCounts {
            tokens: count(Slot::Tokens(window.clone())),
            requests: count(Slot::Requests(window.clone())),
        }
    }

    /// A local counter as stored now.
    pub fn local(&self, key: &str) -> Option<Value> {
        self.get(&Slot::Local(key.to_owned()))
    }

    pub fn set_health_states(&self, states: Value) {
        self.set(Slot::HealthStates, states, None);
    }
}

#[async_trait]
impl RouterCache for InMemoryRouterCache {
    async fn cooldowns(&self, ids: &[String]) -> Result<Vec<Option<Value>>, Error> {
        let mut entries = self.entries.lock();
        Ok(ids
            .iter()
            .map(|id| Self::live(&mut entries, &Slot::Cooldown(id.clone())))
            .collect())
    }

    async fn put_cooldown(&self, id: &str, cooldown: Value, ttl: Duration) -> Result<(), Error> {
        self.set(Slot::Cooldown(id.to_owned()), cooldown, Some(ttl));
        Ok(())
    }

    async fn increment_allowed_fails(
        &self,
        id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> Result<f64, Error> {
        let slot = Slot::AllowedFails(id.to_owned(), suffix.map(str::to_owned));
        Ok(self.increment(slot, 1.0, Some(ttl)))
    }

    async fn add_usage(
        &self,
        window: &UsageWindow,
        tokens: f64,
        requests: f64,
    ) -> Result<(), Error> {
        self.increment(Slot::Tokens(window.clone()), tokens, None);
        self.increment(Slot::Requests(window.clone()), requests, None);
        Ok(())
    }

    async fn add_request(&self, window: &UsageWindow, ttl: Duration) -> Result<(), Error> {
        self.increment(Slot::Requests(window.clone()), 1.0, Some(ttl));
        Ok(())
    }

    async fn usage(&self, window: &UsageWindow) -> Result<UsageCounts, Error> {
        Ok(self.counted(window))
    }

    async fn health_states(&self) -> Result<Option<Value>, Error> {
        Ok(self.get(&Slot::HealthStates))
    }

    async fn get_local(&self, key: &str) -> Result<Option<Value>, Error> {
        Ok(self.local(key))
    }

    async fn increment_local(
        &self,
        key: &str,
        amount: f64,
        ttl: Option<Duration>,
    ) -> Result<f64, Error> {
        Ok(self.increment(Slot::Local(key.to_owned()), amount, ttl))
    }
}
