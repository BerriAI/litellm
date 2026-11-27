//! [`RouterCache`]: the routing state router-core reads and writes, named by what it is (a
//! deployment's cooldown, its allowed-fails count, its per-minute usage, the health checks'
//! states). Where that state lives in Redis, and under which keys, is the host's store's business:
//! router-core never spells a Redis key.

use std::{
    collections::HashMap,
    future::Future,
    hash::Hash,
    sync::Arc,
    time::{Duration, Instant},
};

use litellm_cache::Error;
use parking_lot::Mutex;
use serde::Deserialize;

use crate::health::CooldownValue;

/// `RoutingArgs.ttl`: a per-minute counter lives for its minute.
const MINUTE: Duration = Duration::from_secs(60);

/// One deployment's per-minute usage counters: `model` is the model its usage is counted under,
/// and `minute` is `HH-MM` in UTC.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
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

/// One background health check's result (`DeploymentHealthStateValue`): `timestamp` is when it
/// ran, in Unix seconds.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct HealthState {
    pub is_healthy: bool,
    pub timestamp: f64,
}

/// What this worker saw of a deployment in the current minute.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcomes {
    pub successes: u64,
    pub fails: u64,
}

/// What a [`LimitCounter`] counts, as the model rate limit check names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr, strum::VariantArray)]
pub enum LimitKind {
    #[strum(serialize = "tpm")]
    Tpm,
    #[strum(serialize = "rpm")]
    Rpm,
    #[strum(serialize = "tpd")]
    Tpd,
    #[strum(serialize = "rpd")]
    Rpd,
}

/// A deployment's counter the model rate limit check reads (`UsageCounter.for_deployment`):
/// `window` is `HH-MM` in UTC for a per-minute kind, the day for a per-day one.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LimitCounter {
    pub deployment_id: String,
    pub model: String,
    pub kind: LimitKind,
    pub window: String,
}

pub trait RouterCache: Send + Sync {
    /// The stored cooldown of each of `ids`, in order, `None` for one without.
    fn cooldowns(
        &self,
        ids: &[String],
    ) -> impl Future<Output = Result<Vec<Option<CooldownValue>>, Error>> + Send;

    fn put_cooldown(
        &self,
        id: &str,
        cooldown: &CooldownValue,
        ttl: Duration,
    ) -> impl Future<Output = Result<(), Error>> + Send;

    /// Count one more failure against `id`'s allowed fails (per `suffix` when one is given) and
    /// return the count. `ttl` applies when the counter is created.
    fn increment_allowed_fails(
        &self,
        id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> impl Future<Output = Result<f64, Error>> + Send;

    /// `_increment_deployment_usage`'s write: tokens and requests in one round trip.
    fn add_usage(
        &self,
        window: &UsageWindow,
        tokens: f64,
        requests: f64,
    ) -> impl Future<Output = Result<(), Error>> + Send;

    /// One more request on `window`.
    fn add_request(&self, window: &UsageWindow) -> impl Future<Output = Result<(), Error>> + Send;

    fn usage(
        &self,
        window: &UsageWindow,
    ) -> impl Future<Output = Result<UsageCounts, Error>> + Send;

    /// The background health checks' latest state of each deployment they checked, by id.
    fn health_states(
        &self,
    ) -> impl Future<Output = Result<HashMap<String, HealthState>, Error>> + Send;

    /// `id`'s outcomes in this worker, which no other worker sees.
    fn outcomes(&self, id: &str) -> impl Future<Output = Result<Outcomes, Error>> + Send;

    fn count_success(&self, id: &str) -> impl Future<Output = Result<(), Error>> + Send;

    fn count_failure(&self, id: &str) -> impl Future<Output = Result<(), Error>> + Send;

    /// The model rate limit check's current count (`RouterUsage.current`): this worker's when it
    /// already reaches `limit`, else every worker's. `None` for a counter nothing has written.
    fn limit_usage(
        &self,
        counter: &LimitCounter,
        limit: f64,
    ) -> impl Future<Output = Result<Option<f64>, Error>> + Send;

    /// Add `amount` to `counter`, which lives for `ttl` from its first write, and answer its
    /// total, `None` when it could not be counted.
    fn add_limit_usage(
        &self,
        counter: &LimitCounter,
        amount: i64,
        ttl: Duration,
    ) -> impl Future<Output = Result<Option<f64>, Error>> + Send;

    /// Add every increment in one round trip, answering their totals. Redis batches refresh each
    /// counter's TTL on every write; memory keeps the deadline from its first write.
    fn add_limit_usages(
        &self,
        increments: &[(LimitCounter, i64, Duration)],
    ) -> impl Future<Output = Result<Option<Vec<f64>>, Error>> + Send;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Counter {
    AllowedFails(String, Option<String>),
    Tokens(UsageWindow),
    Requests(UsageWindow),
    Successes(String),
    Fails(String),
    Limit(LimitCounter),
}

struct Expiring<T> {
    value: T,
    expires: Instant,
}

fn live<K: Eq + Hash, T: Clone>(entries: &mut HashMap<K, Expiring<T>>, key: &K) -> Option<T> {
    if entries
        .get(key)
        .is_some_and(|entry| entry.expires <= Instant::now())
    {
        entries.remove(key);
    }
    entries.get(key).map(|entry| entry.value.clone())
}

#[derive(Default)]
struct State {
    cooldowns: HashMap<String, Expiring<CooldownValue>>,
    counters: HashMap<Counter, Expiring<f64>>,
    health_states: HashMap<String, HealthState>,
}

/// An in-process [`RouterCache`]: one process's routing state, shared with no other worker.
#[derive(Clone, Default)]
pub struct InMemoryRouterCache {
    state: Arc<Mutex<State>>,
}

impl InMemoryRouterCache {
    fn count(&self, counter: &Counter) -> Option<f64> {
        live(&mut self.state.lock().counters, counter)
    }

    fn increment(&self, counter: Counter, amount: f64, ttl: Duration) -> f64 {
        let mut state = self.state.lock();
        let next = live(&mut state.counters, &counter).unwrap_or(0.0) + amount;
        let expires = state
            .counters
            .get(&counter)
            .map_or_else(|| Instant::now() + ttl, |entry| entry.expires);
        state.counters.insert(
            counter,
            Expiring {
                value: next,
                expires,
            },
        );
        next
    }

    pub fn counted(&self, window: &UsageWindow) -> UsageCounts {
        UsageCounts {
            tokens: self.count(&Counter::Tokens(window.clone())),
            requests: self.count(&Counter::Requests(window.clone())),
        }
    }

    pub fn set_health_states(&self, states: HashMap<String, HealthState>) {
        self.state.lock().health_states = states;
    }
}

impl RouterCache for InMemoryRouterCache {
    async fn cooldowns(&self, ids: &[String]) -> Result<Vec<Option<CooldownValue>>, Error> {
        let mut state = self.state.lock();
        Ok(ids
            .iter()
            .map(|id| live(&mut state.cooldowns, id))
            .collect())
    }

    async fn put_cooldown(
        &self,
        id: &str,
        cooldown: &CooldownValue,
        ttl: Duration,
    ) -> Result<(), Error> {
        self.state.lock().cooldowns.insert(
            id.to_owned(),
            Expiring {
                value: cooldown.clone(),
                expires: Instant::now() + ttl,
            },
        );
        Ok(())
    }

    async fn increment_allowed_fails(
        &self,
        id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> Result<f64, Error> {
        let counter = Counter::AllowedFails(id.to_owned(), suffix.map(str::to_owned));
        Ok(self.increment(counter, 1.0, ttl))
    }

    async fn add_usage(
        &self,
        window: &UsageWindow,
        tokens: f64,
        requests: f64,
    ) -> Result<(), Error> {
        self.increment(Counter::Tokens(window.clone()), tokens, MINUTE);
        self.increment(Counter::Requests(window.clone()), requests, MINUTE);
        Ok(())
    }

    async fn add_request(&self, window: &UsageWindow) -> Result<(), Error> {
        self.increment(Counter::Requests(window.clone()), 1.0, MINUTE);
        Ok(())
    }

    async fn usage(&self, window: &UsageWindow) -> Result<UsageCounts, Error> {
        Ok(self.counted(window))
    }

    async fn health_states(&self) -> Result<HashMap<String, HealthState>, Error> {
        Ok(self.state.lock().health_states.clone())
    }

    async fn outcomes(&self, id: &str) -> Result<Outcomes, Error> {
        let count = |counter: Counter| self.count(&counter).map_or(0, |count| count as u64);
        Ok(Outcomes {
            successes: count(Counter::Successes(id.to_owned())),
            fails: count(Counter::Fails(id.to_owned())),
        })
    }

    async fn count_success(&self, id: &str) -> Result<(), Error> {
        self.increment(Counter::Successes(id.to_owned()), 1.0, MINUTE);
        Ok(())
    }

    async fn count_failure(&self, id: &str) -> Result<(), Error> {
        self.increment(Counter::Fails(id.to_owned()), 1.0, MINUTE);
        Ok(())
    }

    async fn limit_usage(&self, counter: &LimitCounter, _limit: f64) -> Result<Option<f64>, Error> {
        Ok(self.count(&Counter::Limit(counter.clone())))
    }

    async fn add_limit_usage(
        &self,
        counter: &LimitCounter,
        amount: i64,
        ttl: Duration,
    ) -> Result<Option<f64>, Error> {
        Ok(Some(self.increment(
            Counter::Limit(counter.clone()),
            amount as f64,
            ttl,
        )))
    }

    async fn add_limit_usages(
        &self,
        increments: &[(LimitCounter, i64, Duration)],
    ) -> Result<Option<Vec<f64>>, Error> {
        Ok(Some(
            increments
                .iter()
                .map(|(counter, amount, ttl)| {
                    self.increment(Counter::Limit(counter.clone()), *amount as f64, *ttl)
                })
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[tokio::test]
    async fn usage_counters_expire_so_tomorrows_same_minute_starts_empty() {
        let cache = InMemoryRouterCache::default();
        let window = UsageWindow {
            deployment_id: "dep-1".to_owned(),
            model: "gpt".to_owned(),
            minute: "12-00".to_owned(),
        };

        cache.add_usage(&window, 90.0, 1.0).await.unwrap();

        let state = cache.state.lock();
        let within_a_minute = Instant::now() + MINUTE;
        let expires = |counter: Counter| state.counters[&counter].expires <= within_a_minute;
        assert!(expires(Counter::Tokens(window.clone())));
        assert!(expires(Counter::Requests(window)));
    }

    #[rstest]
    #[tokio::test]
    async fn limit_counters_add_up_apart_from_each_other() {
        let cache = InMemoryRouterCache::default();
        let counter = |kind, window: &str| LimitCounter {
            deployment_id: "dep-1".to_owned(),
            model: "gpt".to_owned(),
            kind,
            window: window.to_owned(),
        };
        let rpm = counter(LimitKind::Rpm, "12-00");
        let rpd = counter(LimitKind::Rpd, "2026-01-01");

        let charged = cache
            .add_limit_usages(&[(rpm.clone(), 1, MINUTE), (rpd.clone(), 1, 1440 * MINUTE)])
            .await
            .unwrap();
        let recharged = cache.add_limit_usage(&rpm, 2, MINUTE).await.unwrap();

        assert_eq!((charged, recharged), (Some(vec![1.0, 1.0]), Some(3.0)));
        assert_eq!(cache.limit_usage(&rpd, 5.0).await.unwrap(), Some(1.0));
        let unwritten = counter(LimitKind::Tpm, "12-00");
        assert_eq!(cache.limit_usage(&unwritten, 5.0).await.unwrap(), None);
    }
}
