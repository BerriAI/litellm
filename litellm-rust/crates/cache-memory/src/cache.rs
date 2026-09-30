use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    hash::Hash,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::{
    BaseCache, BatchCache, ClaimCache, CounterCache, DeleteCache, DisconnectCache, Error,
    ExactCacheContext, FlushCache, SetCache, TtlCache,
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
            default_ttl: default_ttl
                .filter(|ttl| !ttl.is_zero())
                .unwrap_or(DEFAULT_TTL),
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
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        self.store(&mut state, key.into(), value, ttl, now)
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

    /// `check_value_size`: whether `value` fits `max_entry_bytes`. Always `true` without a
    /// limit and a measure, since typed values have no generic size.
    pub fn check_value_size(&self, value: &V) -> Result<bool, Error> {
        match (self.max_entry_bytes, &self.measure_value) {
            (Some(limit), Some(measure)) => Ok(measure(value)? <= limit),
            _ => Ok(true),
        }
    }

    /// `evict_cache`: drops expired entries, then the earliest-expiring ones until a new key
    /// fits.
    pub fn evict_cache(&self) -> Result<(), Error> {
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        Self::evict(&mut state, self.max_size_in_memory, now, None);
        Ok(())
    }

    /// `evict_element_if_expired`: `true` when `key` had expired and was removed.
    pub fn evict_element_if_expired(&self, key: &str) -> Result<bool, Error> {
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        let expired = state
            .expirations
            .get(key)
            .is_some_and(|expiration| *expiration < now);
        if expired {
            Self::remove(&mut state, key);
        }
        Ok(expired)
    }

    /// `allow_ttl_override`: a write may set the TTL when the key has none or it has passed.
    pub fn allow_ttl_override(&self, key: &str) -> Result<bool, Error> {
        let now = (self.now)();
        Ok(self
            .expires_at(key)?
            .is_none_or(|expiration| expiration < now))
    }

    /// The number of stored entries, expired ones included until they are evicted.
    pub fn len(&self) -> Result<usize, Error> {
        Ok(self
            .state
            .lock()
            .map_err(|_| Error::Unavailable)?
            .values
            .len())
    }

    pub fn is_empty(&self) -> Result<bool, Error> {
        Ok(self.len()? == 0)
    }

    /// Entries in the expiration heap, stale ones included; bounded by eviction.
    pub fn expiration_heap_len(&self) -> Result<usize, Error> {
        Ok(self
            .state
            .lock()
            .map_err(|_| Error::Unavailable)?
            .expiration_heap
            .len())
    }

    pub fn max_size_in_memory(&self) -> usize {
        self.max_size_in_memory
    }

    pub fn max_entry_bytes(&self) -> Option<usize> {
        self.max_entry_bytes
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

    /// Writing an existing `key` never evicts another entry, unlike Python, which pops the
    /// earliest-expiring entry whenever the cache is full.
    fn evict(state: &mut CacheState<V>, capacity: usize, now: Duration, key: Option<&str>) {
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
        if key.is_some_and(|key| state.values.contains_key(key)) {
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

    /// `get_cache` under the held lock: an expired entry is removed and reads as missing.
    fn live(state: &mut CacheState<V>, key: &str, now: Duration) -> Option<V> {
        if state
            .expirations
            .get(key)
            .is_some_and(|expiration| *expiration < now)
        {
            Self::remove(state, key);
        }
        state.values.get(key).cloned()
    }

    /// Python `set_cache` under the held lock: evict first (even when `key` already exists),
    /// then skip oversized values, then write, keeping a live key's expiry.
    fn store(
        &self,
        state: &mut CacheState<V>,
        key: String,
        value: V,
        ttl: Option<Duration>,
        now: Duration,
    ) -> Result<CacheWrite, Error> {
        Self::evict(state, self.max_size_in_memory, now, None);
        if !self.check_value_size(&value)? {
            return Ok(CacheWrite::TooLarge);
        }
        let expiration = state.expirations.get(&key).copied();
        if expiration.is_none_or(|expiration| expiration < now) {
            Self::set_expiration(state, &key, now + ttl.unwrap_or(self.default_ttl));
        }
        state.values.insert(key, value);
        Ok(CacheWrite::Stored)
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
        context: ExactCacheContext,
    ) -> Result<V, Error> {
        if self.max_size_in_memory == 0 {
            return Ok(candidate);
        }
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        Self::evict(&mut state, self.max_size_in_memory, now, Some(key));
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
        Self::set_expiration(
            &mut state,
            key,
            now + self.get_ttl(&context).unwrap_or(self.default_ttl),
        );
        state.values.insert(key.into(), winner.clone());
        Ok(winner)
    }
}

impl CounterCache for InMemoryCache<f64> {
    fn increment_cache(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
    ) -> Result<f64, Error> {
        if self.max_size_in_memory == 0 {
            return Ok(amount);
        }
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        let value = Self::live(&mut state, key, now).unwrap_or_default() + amount;
        self.store(&mut state, key.into(), value, self.get_ttl(&context), now)?;
        Ok(value)
    }
}

impl<V: Clone + Send + Sync + 'static> BaseCache for InMemoryCache<V> {
    type Value = V;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl.or(Some(self.default_ttl))
    }

    fn set_cache(
        &self,
        key: &str,
        value: Self::Value,
        context: &ExactCacheContext,
    ) -> Result<(), Error> {
        let ttl = self.get_ttl(context).unwrap_or(self.default_ttl);
        self.set_cache(key, value, Some(ttl)).map(|_| ())
    }

    fn get_cache(&self, key: &str, _: &ExactCacheContext) -> Result<Option<Self::Value>, Error> {
        self.get_cache(key)
    }
}

impl<V: Clone + Send + Sync + 'static> DisconnectCache for InMemoryCache<V> {
    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl<V: Clone + Send + Sync + 'static> BatchCache for InMemoryCache<V> {}

impl<V: Clone + Send + Sync + 'static> DeleteCache for InMemoryCache<V> {
    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        InMemoryCache::delete_cache(self, key)
    }
}

impl<V: Clone + Send + Sync + 'static> FlushCache for InMemoryCache<V> {
    fn flush_cache(&self) -> Result<(), Error> {
        InMemoryCache::flush_cache(self)
    }
}

impl<V: Clone + Send + Sync + 'static> TtlCache for InMemoryCache<V> {
    async fn async_get_ttl(&self, key: &str) -> Result<Option<Duration>, Error> {
        InMemoryCache::async_get_ttl(self, key).await
    }
}

impl<T> SetCache for InMemoryCache<HashSet<T>>
where
    T: Clone + Eq + Hash + Send + Sync + 'static,
{
    type SetValue = T;
    type SetResult = Vec<T>;

    async fn async_set_cache_sadd(
        &self,
        key: &str,
        values: Vec<Self::SetValue>,
        ttl: Option<Duration>,
    ) -> Result<Self::SetResult, Error> {
        if self.max_size_in_memory == 0 {
            return Ok(values);
        }
        let now = (self.now)();
        let mut state = self.state.lock().map_err(|_| Error::Unavailable)?;
        let mut stored = Self::live(&mut state, key, now).unwrap_or_default();
        stored.extend(values.iter().cloned());
        self.store(&mut state, key.into(), stored, ttl, now)?;
        Ok(values)
    }
}
