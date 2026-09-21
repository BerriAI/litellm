use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::{
    BaseCache, CacheConnectionResult, CacheConnectionStatus, CacheKwargs, ClaimCache, CounterCache,
    Error, IncrementOperation,
};

const DEFAULT_MAX_SIZE_IN_MEMORY: usize = 200;
const DEFAULT_TTL: Duration = Duration::from_secs(600);

type ValueMeasure<V> = Arc<dyn Fn(&V) -> Result<usize, Error> + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheWrite {
    Stored,
    Disabled,
    TooLarge,
}

struct CacheState<V> {
    values: HashMap<String, V>,
    expirations: HashMap<String, Duration>,
    expiration_heap: BinaryHeap<Reverse<(Duration, String)>>,
}

pub struct InMemoryCache<V: Clone> {
    state: Mutex<CacheState<V>>,
    max_size_in_memory: usize,
    default_ttl: Duration,
    max_entry_bytes: Option<usize>,
    measure_value: Option<ValueMeasure<V>>,
    now: Arc<dyn Fn() -> Duration + Send + Sync>,
}

impl<V: Clone> Default for InMemoryCache<V> {
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl<V: Clone> InMemoryCache<V> {
    pub fn new(max_size_in_memory: Option<usize>, default_ttl: Option<Duration>) -> Self {
        Self::with_clock(max_size_in_memory, default_ttl, || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
        })
    }

    pub fn with_clock(
        max_size_in_memory: Option<usize>,
        default_ttl: Option<Duration>,
        now: impl Fn() -> Duration + Send + Sync + 'static,
    ) -> Self {
        Self::with_clock_and_size_measurement(max_size_in_memory, default_ttl, None, None, now)
    }

    pub fn with_clock_and_size_measurement(
        max_size_in_memory: Option<usize>,
        default_ttl: Option<Duration>,
        max_entry_bytes: Option<usize>,
        measure_value: Option<ValueMeasure<V>>,
        now: impl Fn() -> Duration + Send + Sync + 'static,
    ) -> Self {
        Self {
            state: Mutex::new(CacheState {
                values: HashMap::new(),
                expirations: HashMap::new(),
                expiration_heap: BinaryHeap::new(),
            }),
            max_size_in_memory: max_size_in_memory.unwrap_or(DEFAULT_MAX_SIZE_IN_MEMORY),
            default_ttl: default_ttl.unwrap_or(DEFAULT_TTL),
            max_entry_bytes,
            measure_value,
            now: Arc::new(now),
        }
    }

    pub fn set_cache(
        &self,
        key: impl Into<String>,
        value: V,
        ttl: Option<Duration>,
    ) -> Result<CacheWrite, Error> {
        if self.max_size_in_memory == 0 {
            return Ok(CacheWrite::Disabled);
        }
        if let (Some(limit), Some(measure)) = (self.max_entry_bytes, &self.measure_value)
            && measure(&value)? > limit
        {
            return Ok(CacheWrite::TooLarge);
        }
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        let key = key.into();
        Self::evict(&mut state, self.max_size_in_memory, now, &key);
        let expiration = state.expirations.get(&key).copied();
        if expiration.is_none_or(|expiration| expiration < now) {
            Self::set_expiration(&mut state, &key, now + ttl.unwrap_or(self.default_ttl));
        }
        state.values.insert(key, value);
        Ok(CacheWrite::Stored)
    }

    pub fn get_cache(&self, key: &str) -> Result<Option<V>, Error> {
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        if state
            .expirations
            .get(key)
            .is_some_and(|expiration| *expiration < now)
        {
            Self::remove(&mut state, key);
        }
        Ok(state.values.get(key).cloned())
    }

    pub fn max_size_in_memory(&self) -> usize {
        self.max_size_in_memory
    }

    pub fn expires_at(&self, key: &str) -> Result<Option<Duration>, Error> {
        Ok(self
            .state
            .lock()
            .map_err(|_| Error::Unavailable)?
            .expirations
            .get(key)
            .copied())
    }

    pub async fn async_get_ttl(&self, key: &str) -> Result<Option<Duration>, Error> {
        self.expires_at(key)
    }

    pub async fn async_get_oldest_n_keys(&self, count: usize) -> Result<Vec<String>, Error> {
        let state = self.state.lock().map_err(|_| Error::Unavailable)?;
        let mut expirations = state
            .expirations
            .iter()
            .map(|(key, expiration)| (key.clone(), *expiration))
            .collect::<Vec<_>>();
        expirations.sort_unstable_by_key(|(_, expiration)| *expiration);
        Ok(expirations
            .into_iter()
            .take(count)
            .map(|(key, _)| key)
            .collect())
    }

    pub fn delete_cache(&self, key: &str) -> Result<(), Error> {
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        Self::remove(&mut state, key);
        Ok(())
    }

    pub fn flush_cache(&self) -> Result<(), Error> {
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        state.values.clear();
        state.expirations.clear();
        state.expiration_heap.clear();
        Ok(())
    }

    fn evict(state: &mut CacheState<V>, capacity: usize, now: Duration, key: &str) {
        while let Some(Reverse((expiration, key))) = state.expiration_heap.peek().cloned() {
            if state.expirations.get(&key).copied() != Some(expiration) {
                state.expiration_heap.pop();
            } else if expiration <= now {
                state.expiration_heap.pop();
                Self::remove(state, &key);
            } else {
                break;
            }
        }
        if state.values.contains_key(key) {
            return;
        }
        while state.values.len() >= capacity {
            let Some(Reverse((expiration, key))) = state.expiration_heap.pop() else {
                break;
            };
            if state.expirations.get(&key).copied() == Some(expiration) {
                Self::remove(state, &key);
            }
        }
    }

    fn set_expiration(state: &mut CacheState<V>, key: &str, expiration: Duration) {
        if state.expirations.get(key).copied() != Some(expiration) {
            state.expirations.insert(key.into(), expiration);
            state
                .expiration_heap
                .push(Reverse((expiration, key.into())));
        }
    }

    fn remove(state: &mut CacheState<V>, key: &str) {
        state.values.remove(key);
        state.expirations.remove(key);
    }
}

impl<V> ClaimCache for InMemoryCache<V>
where
    V: Clone + PartialEq + Send + Sync + 'static,
{
    fn claim_cache(
        &self,
        key: &str,
        candidate: V,
        eligible: &[V],
        kwargs: CacheKwargs,
    ) -> Result<V, Error> {
        if self.max_size_in_memory == 0 {
            return Ok(candidate);
        }
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        Self::evict(&mut state, self.max_size_in_memory, now, key);
        let existing = state
            .values
            .get(key)
            .filter(|existing| eligible.is_empty() || eligible.contains(existing))
            .cloned();
        if let Some(existing) = &existing
            && eligible.is_empty()
            && *existing != candidate
        {
            return Ok(existing.clone());
        }
        let winner = existing.unwrap_or(candidate);
        Self::set_expiration(&mut state, key, now + self.get_ttl(&kwargs));
        state.values.insert(key.into(), winner.clone());
        Ok(winner)
    }
}

impl CounterCache for InMemoryCache<f64> {
    fn increment_cache(&self, key: &str, amount: f64, kwargs: CacheKwargs) -> Result<f64, Error> {
        if self.max_size_in_memory == 0 {
            return Ok(amount);
        }
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        Self::evict(&mut state, self.max_size_in_memory, now, key);
        let value = state.values.get(key).copied().unwrap_or_default() + amount;
        if !state.expirations.contains_key(key) {
            Self::set_expiration(&mut state, key, now + self.get_ttl(&kwargs));
        }
        state.values.insert(key.into(), value);
        Ok(value)
    }
}

impl InMemoryCache<f64> {
    pub async fn async_increment_pipeline(
        &self,
        operations: Vec<IncrementOperation>,
    ) -> Result<Vec<f64>, Error> {
        operations
            .into_iter()
            .map(|operation| {
                self.increment_cache(
                    &operation.key,
                    operation.amount,
                    CacheKwargs {
                        ttl: operation.ttl,
                        ..CacheKwargs::default()
                    },
                )
            })
            .collect()
    }
}

impl<V: Clone + Send + Sync + 'static> BaseCache for InMemoryCache<V> {
    type Value = V;

    fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    fn set_cache(&self, key: &str, value: Self::Value, kwargs: CacheKwargs) -> Result<(), Error> {
        let ttl = self.get_ttl(&kwargs);
        self.set_cache(key, value, Some(ttl)).map(|_| ())
    }

    fn get_cache(&self, key: &str, _: &CacheKwargs) -> Result<Option<Self::Value>, Error> {
        self.get_cache(key)
    }

    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        self.delete_cache(key)
    }

    fn flush_cache(&self) -> Result<(), Error> {
        self.flush_cache()
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        Ok(CacheConnectionResult {
            status: CacheConnectionStatus::Success,
            message: "In-memory cache connection test successful".into(),
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_increments_keep_one_heap_entry_per_expiration() {
        let cache = InMemoryCache::<f64>::new(Some(4), None);
        for _ in 0..100 {
            cache
                .increment_cache("counter", 1.0, CacheKwargs::default())
                .unwrap();
        }
        assert_eq!(cache.state.lock().unwrap().expiration_heap.len(), 1);
    }
}
